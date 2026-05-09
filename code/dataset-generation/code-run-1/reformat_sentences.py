#!/usr/bin/env python3
"""
Reformat numbered sentences from one-per-line to paragraph format
This reduces tokens when sending to APIs (more cost effective)
"""

def reformat_to_paragraph(input_file, output_file):
    """
    Read sentences and join them with spaces instead of newlines
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Remove any empty lines and strip whitespace
    sentences = [line.strip() for line in lines if line.strip()]

    # Join with single space
    paragraph = ' '.join(sentences)

    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(paragraph)

    print(f"✅ Reformatted {len(sentences)} sentences")
    print(f"📥 Input:  {input_file}")
    print(f"📤 Output: {output_file}")
    print(f"\n📊 Stats:")
    print(f"   Original length: {sum(len(s) for s in sentences) + len(sentences) - 1} chars")
    print(f"   Paragraph length: {len(paragraph)} chars")
    print(f"   Token savings: ~{len(sentences)} newline tokens")

if __name__ == "__main__":
    input_file = "final-output/ADJA_FRENCH_SENTENCES.txt"
    output_file = "final-output/ADJA_FRENCH_SENTENCES_paragraph.txt"

    reformat_to_paragraph(input_file, output_file)
