#!/usr/bin/env python3
from __future__ import annotations
"""
B1 BiLSTM-CTC ASR Model for Adja.

Architecture:
    Log-mel spectrogram (80 dims) -> SpecAugment -> 3-layer BiLSTM -> Linear -> CTC

References:
    - Graves et al. 2006: Connectionist Temporal Classification
    - Park et al. 2019: SpecAugment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class LogMelFrontend(nn.Module):
    """Extract log-mel spectrogram features from raw waveforms.

    Uses torchaudio's MelSpectrogram. Output shape: (batch, time, n_mels).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: float = 8000.0,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.hop_length = hop_length

    def forward(self, waveforms: torch.Tensor, wav_lengths: torch.Tensor):
        """
        Args:
            waveforms: (batch, samples) raw audio
            wav_lengths: (batch,) number of valid samples per utterance

        Returns:
            features: (batch, time, n_mels) log-mel features
            feat_lengths: (batch,) number of valid frames per utterance
        """
        # MelSpectrogram expects (batch, samples), returns (batch, n_mels, time)
        mel = self.mel_spec(waveforms)

        # Log compression (add small epsilon to avoid log(0))
        log_mel = torch.log(mel + 1e-9)

        # Transpose to (batch, time, n_mels)
        features = log_mel.transpose(1, 2)

        # Compute output lengths: each frame covers hop_length samples
        # torchaudio MelSpectrogram output length = floor(samples / hop_length) + 1
        feat_lengths = torch.div(wav_lengths, self.hop_length, rounding_mode="floor") + 1

        # Clamp to actual feature length (in case of rounding edge cases)
        feat_lengths = torch.clamp(feat_lengths, max=features.size(1))

        return features, feat_lengths


class SpecAugment(nn.Module):
    """SpecAugment data augmentation (Park et al. 2019).

    Applies frequency and time masking to log-mel spectrograms.
    Only active during training.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, time, n_mels)

        Returns:
            Augmented features with same shape.
        """
        if not self.training:
            return features

        # torchaudio masking expects (batch, freq, time), so transpose
        x = features.transpose(1, 2)  # (batch, n_mels, time)

        for _ in range(self.n_freq_masks):
            x = self.freq_mask(x)
        for _ in range(self.n_time_masks):
            x = self.time_mask(x)

        return x.transpose(1, 2)  # back to (batch, time, n_mels)


class BiLSTMCTCModel(nn.Module):
    """BiLSTM encoder with CTC loss for ASR.

    Full pipeline:
        waveform -> LogMel -> SpecAugment -> BiLSTM -> Linear -> CTC
    """

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        encoder_dim: int = 256,
        encoder_layers: int = 3,
        dropout: float = 0.3,
        blank_index: int = 0,
        # Frontend params
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        # SpecAugment params
        spec_augment: bool = True,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.blank_index = blank_index
        self.vocab_size = vocab_size

        # Feature extraction
        self.frontend = LogMelFrontend(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

        # SpecAugment
        self.spec_augment = None
        if spec_augment:
            self.spec_augment = SpecAugment(
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
                n_freq_masks=n_freq_masks,
                n_time_masks=n_time_masks,
            )

        # BiLSTM encoder
        self.encoder = nn.LSTM(
            input_size=n_mels,
            hidden_size=encoder_dim,
            num_layers=encoder_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if encoder_layers > 1 else 0.0,
        )

        self.encoder_dropout = nn.Dropout(dropout)

        # Projection: BiLSTM outputs 2*encoder_dim -> vocab_size
        self.output_proj = nn.Linear(2 * encoder_dim, vocab_size)

        # CTC loss
        self.ctc_loss = nn.CTCLoss(blank=blank_index, reduction="mean", zero_infinity=True)

    def forward(
        self,
        waveforms: torch.Tensor,
        wav_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: waveforms -> log-probs.

        Args:
            waveforms: (batch, max_samples) raw audio, zero-padded
            wav_lengths: (batch,) valid sample counts

        Returns:
            log_probs: (batch, time, vocab_size) log-softmax output
            feat_lengths: (batch,) valid frame counts
        """
        # Extract features
        features, feat_lengths = self.frontend(waveforms, wav_lengths)

        # SpecAugment (training only)
        if self.spec_augment is not None:
            features = self.spec_augment(features)

        # Pack padded sequences for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            features,
            feat_lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False,
        )

        # BiLSTM encoder
        packed_out, _ = self.encoder(packed)

        # Unpack
        encoder_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        encoder_out = self.encoder_dropout(encoder_out)

        # Project to vocabulary
        logits = self.output_proj(encoder_out)  # (batch, time, vocab_size)

        # Log-softmax for CTC
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, feat_lengths

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        feat_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.

        Args:
            log_probs: (batch, time, vocab_size) from forward()
            feat_lengths: (batch,) valid frame counts
            targets: (batch, max_label_len) token indices (no blanks)
            target_lengths: (batch,) valid label lengths

        Returns:
            CTC loss scalar
        """
        # CTCLoss expects (time, batch, vocab_size)
        log_probs_t = log_probs.transpose(0, 1)

        loss = self.ctc_loss(
            log_probs_t,
            targets,
            feat_lengths.long(),
            target_lengths.long(),
        )

        return loss

    def greedy_decode(
        self,
        log_probs: torch.Tensor,
        feat_lengths: torch.Tensor,
    ) -> list[list[int]]:
        """Greedy CTC decoding: argmax -> collapse repeats -> remove blanks.

        Args:
            log_probs: (batch, time, vocab_size)
            feat_lengths: (batch,) valid frame counts

        Returns:
            List of decoded token index sequences (one per utterance).
        """
        # Argmax over vocabulary
        predictions = torch.argmax(log_probs, dim=-1)  # (batch, time)

        decoded = []
        for i in range(predictions.size(0)):
            seq = predictions[i, : feat_lengths[i]].tolist()

            # Collapse consecutive duplicates, then remove blanks
            collapsed = []
            prev = None
            for token in seq:
                if token != prev:
                    if token != self.blank_index:
                        collapsed.append(token)
                    prev = token

            decoded.append(collapsed)

        return decoded
