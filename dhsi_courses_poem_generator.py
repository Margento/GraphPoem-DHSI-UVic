import os
import random

# ================= CONFIGURATION =================
SOURCE_FOLDER = './poems_#dhsi25'      
OUTPUT_FILE = 'dhsi_courses_compiled_poem.txt'
# =================================================

def get_random_chunk(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if not text:
        return None

    # Pick a random starting point in the text
    start_idx = random.randint(0, max(0, len(text) - 1))
    remaining_text = text[start_idx:]
    
    # Split into words
    words = remaining_text.split()
    if not words:
        return None

    lines = []
    word_idx = 0
    
    # Loop to create lines of 4 and 3 words alternatively
    while word_idx < len(words):
        # Determine if we need 4 or 3 words for this line
        target_len = 4 if len(lines) % 2 == 0 else 3
        
        # Grab the slice of words
        line_words = words[word_idx : word_idx + target_len]
        line_text = " ".join(line_words)
        
        lines.append(line_text)
        word_idx += target_len
        
        # BREAK CONDITION 1: Hit a period
        if line_text.endswith('.'):
            break
            
        # BREAK CONDITION 2: Hit 6 or 8 lines (randomly chosen for variety)
        if len(lines) == random.choice([6, 8]):
            break
            
    return "\n".join(lines)

def main():
    # Get all files starting with dhsi_
    files = [f for f in os.listdir(SOURCE_FOLDER) if f.startswith('dhsi_') and f.endswith('.txt')]
    files.sort() # Ensure consistent order

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for i, filename in enumerate(files):
            path = os.path.join(SOURCE_FOLDER, filename)
            chunk = get_random_chunk(path)
            
            if chunk:
                out_f.write(chunk + "\n")
            
            # STANZA BREAK LOGIC:
            # "Alternately every other file or every 4 files"
            # We use a simple toggle: if index is even, check for 2; if odd, check for 4.
            if i % 2 == 0:
                if (i + 1) % 2 == 0: # Every 2nd file
                    out_f.write("\n\n")
            else:
                if (i + 1) % 4 == 0: # Every 4th file
                    out_f.write("\n\n")

    print(f"Done! Processed {len(files)} files into {OUTPUT_FILE}")

if __name__ == "__main__":
    main()