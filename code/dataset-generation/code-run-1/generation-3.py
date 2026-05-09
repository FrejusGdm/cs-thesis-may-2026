





# transform_modules_FIXED.py
import pandas as pd
from openai import OpenAI
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv  # ADD THIS
load_dotenv()  # ADD THIS LINE


# Initialize OpenAI client (new way)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def batch_transform(sentences, transformation_type, batch_size=15):
    """
    Transform sentences in small batches using NEW OpenAI API
    """
    
    all_transformed = []
    failed_batches = []
    
    num_batches = (len(sentences) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(sentences), batch_size), 
                  desc=f"Transforming to {transformation_type}", 
                  total=num_batches):
        
        batch = sentences[i:i+batch_size]
        
        # Create numbered list
        batch_text = "\n".join([f"{idx}. {sent}" for idx, sent in enumerate(batch, start=1)])
        
        # Get the right prompt
        prompt = get_transformation_prompt(transformation_type, batch_text)
        
        # Try API call with retry logic
        success = False
        retries = 3
        
        for attempt in range(retries):
            try:
                # NEW API SYNTAX
                response = client.chat.completions.create(
                    model="gpt-4.1-mini-2025-04-14",  # Updated model name
                    messages=[
                        {"role": "system", "content": "You are a French grammar expert. Follow instructions precisely."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )
                
                # NEW way to access response
                result = response.choices[0].message.content.strip()
                
                # Parse numbered output
                parsed = parse_numbered_output(result, len(batch))
                
                # Validate we got the right number of sentences
                if len(parsed) == len(batch):
                    all_transformed.extend(parsed)
                    success = True
                    break
                else:
                    print(f"\n⚠️  Batch {i//batch_size + 1}: Expected {len(batch)} sentences, got {len(parsed)}")
                    if attempt < retries - 1:
                        print(f"   Retrying (attempt {attempt + 2}/{retries})...")
                        time.sleep(2)
                
            except Exception as e:
                print(f"\n❌ Error on batch {i//batch_size + 1}: {e}")
                if attempt < retries - 1:
                    print(f"   Retrying (attempt {attempt + 2}/{retries})...")
                    time.sleep(2)
        
        if not success:
            # If all retries failed, save the batch for manual review
            failed_batches.append((i, batch))
            all_transformed.extend(['ERROR'] * len(batch))
            print(f"   ⚠️  Batch {i//batch_size + 1} failed after {retries} attempts")
        
        # Rate limiting
        time.sleep(1.5)
    
    # Report failures
    if failed_batches:
        print(f"\n⚠️  {len(failed_batches)} batches failed. Saving for manual review...")
        save_failed_batches(failed_batches, transformation_type)
    
    return all_transformed

def parse_numbered_output(text, expected_count):
    """Parse GPT output that should be numbered sentences"""
    sentences = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Look for numbered format: "1. sentence" or "1) sentence"
        if line and line[0].isdigit():
            # Remove number and separator
            if '. ' in line:
                sentence = line.split('. ', 1)[1]
            elif ') ' in line:
                sentence = line.split(') ', 1)[1]
            else:
                sentence = line[2:].strip()
            
            sentences.append(sentence.strip())
    
    return sentences

def get_transformation_prompt(transformation_type, batch_text):
    """Get the appropriate prompt for each transformation"""
    
    if transformation_type == "past":
        return f"""Transform these French PRESENT tense sentences into PAST tense (passé composé).

CRITICAL RULES:
- Use correct auxiliary (avoir or être)
- Correct past participle
- Maintain pronoun and object EXACTLY
- Natural French only

Sentences to transform:
{batch_text}

RESPOND WITH:
Only numbered sentences (1. ... 2. ... 3. ...)
NO explanations, NO extra text
Match the input numbering exactly"""

    elif transformation_type == "future":
        return f"""Transform these French PRESENT tense sentences into FUTURE tense.

CRITICAL RULES:
- Use "aller + infinitive" form ONLY
- Conjugate "aller" correctly (je vais, tu vas, il va, nous allons, vous allez, ils vont, elles vont)
- Keep infinitive form of main verb
- Maintain object EXACTLY
- Natural French only

Examples:
- je mange du riz → je vais manger du riz
- nous voyons mon ami → nous allons voir mon ami

Sentences to transform:
{batch_text}

RESPOND WITH:
Only numbered sentences (1. ... 2. ... 3. ...)
NO explanations, NO extra text
Match the input numbering exactly"""

    elif transformation_type == "question_yn":
        return f"""Transform these French statements into YES/NO QUESTIONS.

CRITICAL RULES:
- Add "Est-ce que" at the beginning
- Keep everything else EXACTLY the same
- Add question mark at end
- Natural French only

Example:
- je mange du riz → Est-ce que je mange du riz ?

Sentences to transform:
{batch_text}

RESPOND WITH:
Only numbered questions (1. ... 2. ... 3. ...)
NO explanations, NO extra text
Match the input numbering exactly"""

    elif transformation_type == "question_wh":
        return f"""Transform these French statements into WH-QUESTIONS.

CRITICAL RULES:
- Use appropriate wh-word: Que/Qu'est-ce que (what), Qui (who), Où (where), Quand (when), Pourquoi (why), Comment (how)
- Vary the wh-words naturally based on sentence content
- Natural French only
- Add question mark

Examples:
- je mange du riz → Qu'est-ce que je mange ?
- je vais à l'école → Où vas-tu ?
- il voit mon ami → Qui voit-il ?

Sentences to transform:
{batch_text}

RESPOND WITH:
Only numbered questions (1. ... 2. ... 3. ...)
NO explanations, NO extra text
Match the input numbering exactly"""

def save_failed_batches(failed_batches, transformation_type):
    """Save failed batches for manual review"""
    with open(f'failed_batches_{transformation_type}.txt', 'w', encoding='utf-8') as f:
        for idx, batch in failed_batches:
            f.write(f"\n=== Batch starting at index {idx} ===\n")
            for sent in batch:
                f.write(f"{sent}\n")
    print(f"   Saved to: failed_batches_{transformation_type}.txt")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    
    print("📂 Loading Module 1...\n")
    df_module1 = pd.read_csv('module1_base.csv')
    base_sentences = df_module1['french'].tolist()
    
    print(f"Loaded {len(base_sentences)} sentences")
    print(f"Will process in batches of 15 sentences")
    print(f"Estimated: {(len(base_sentences) + 14) // 15} API calls per module\n")
    
    print("💰 Cost estimate:")
    print("   - GPT-4o: ~$0.50 per module")
    print("   - Total for 3 modules: ~$1.50")
    
    # Ask for confirmation
    response = input("\nPress Enter to start (or 'q' to quit)...")
    if response.lower() == 'q':
        print("Cancelled.")
        exit()
    
    # Module 3: Past Tense
    print("\n" + "="*60)
    print("MODULE 3: PAST TENSE")
    print("="*60)
    past = batch_transform(base_sentences, "past", batch_size=15)
    
    df_module3 = df_module1.copy()
    df_module3['sentence_id'] = df_module3['sentence_id'].str.replace('M1_', 'M3_')
    df_module3['module'] = 'module3_past'
    df_module3['french'] = past
    df_module3['structure'] = 'affirmative_past'
    df_module3['base_sentence_id'] = df_module1['sentence_id']
    df_module3.to_csv('module3_past.csv', index=False)
    
    print(f"\n✅ Module 3 saved: module3_past.csv")
    print(f"📝 Sample transformations:")
    for i in range(min(5, len(past))):
        if past[i] != 'ERROR':
            print(f"   {df_module1.iloc[i]['french']} → {past[i]}")
    
    # Module 4: Future Tense
    print("\n" + "="*60)
    print("MODULE 4: FUTURE TENSE")
    print("="*60)
    future = batch_transform(base_sentences, "future", batch_size=15)
    
    df_module4 = df_module1.copy()
    df_module4['sentence_id'] = df_module4['sentence_id'].str.replace('M1_', 'M4_')
    df_module4['module'] = 'module4_future'
    df_module4['french'] = future
    df_module4['structure'] = 'affirmative_future'
    df_module4['base_sentence_id'] = df_module1['sentence_id']
    df_module4.to_csv('module4_future.csv', index=False)
    
    print(f"\n✅ Module 4 saved: module4_future.csv")
    print(f"📝 Sample transformations:")
    for i in range(min(5, len(future))):
        if future[i] != 'ERROR':
            print(f"   {df_module1.iloc[i]['french']} → {future[i]}")
    
    # Module 5: Questions (split into yes/no and wh)
    print("\n" + "="*60)
    print("MODULE 5: QUESTIONS")
    print("="*60)
    
    # Split sentences for question types
    half = len(base_sentences) // 2
    
    print(f"Part A: Yes/No Questions ({half} sentences)")
    questions_yn = batch_transform(base_sentences[:half], "question_yn", batch_size=15)
    
    print(f"\nPart B: Wh-Questions ({len(base_sentences) - half} sentences)")
    questions_wh = batch_transform(base_sentences[half:], "question_wh", batch_size=15)
    
    # Combine
    all_questions = questions_yn + questions_wh
    
    df_module5 = df_module1.copy()
    df_module5['sentence_id'] = df_module5['sentence_id'].str.replace('M1_', 'M5_')
    df_module5['module'] = 'module5_questions'
    df_module5['french'] = all_questions
    df_module5['structure'] = ['question_yn'] * half + ['question_wh'] * (len(base_sentences) - half)
    df_module5['base_sentence_id'] = df_module1['sentence_id']
    df_module5.to_csv('module5_questions.csv', index=False)
    
    print(f"\n✅ Module 5 saved: module5_questions.csv")
    print(f"📝 Sample transformations:")
    for i in range(min(3, len(all_questions))):
        if all_questions[i] != 'ERROR':
            print(f"   {df_module1.iloc[i]['french']} → {all_questions[i]}")
    
    print("\n" + "="*60)
    print("✅ ALL COMPLEX MODULES GENERATED!")
    print("="*60)
    
    # Summary
    error_count = past.count('ERROR') + future.count('ERROR') + all_questions.count('ERROR')
    total_count = len(past) + len(future) + len(all_questions)
    
    print(f"\n📊 Summary:")
    print(f"   Total sentences generated: {total_count}")
    print(f"   Successful: {total_count - error_count}")
    print(f"   Errors: {error_count}")
    
    if error_count > 0:
        print(f"\n⚠️  Check failed_batches_*.txt files for manual review")














