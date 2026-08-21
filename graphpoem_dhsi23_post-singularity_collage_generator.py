import os
import random
import string

# ================= CONFIGURATION =================
SOURCE_FOLDER = './poems_#dhsi25'
OUTPUT_FILE = 'graphpoem_dhsi23_post-singularity_shuffled_excerpt_collage.txt' 
FILE_LIST = [
    'pierre_peuchmaurd_glimmers_from_french_trans_ec_belli_poem_1.txt', 
    'saksiri_meesomsueb_dogs_in_the_lead_from_thai_trans_noh_anothai_poem_3.txt', 
    'qiu_jin_reflections_from_chinese_trans_yilin_wang_poem_3.txt', 
    'phu_recalling_love_scenes_by_pleasant_river_from_thai_trans_noh_anothai_poem_2.txt', 
    'pierre_peuchmaurd_the_foam_of_lions_from_french_trans_ec_belli_poem_2.txt', 
    'saksiri_meesomsueb_sleight_from_thai_trans_noh_anothai_poem_1.txt', 
    'sappho_31_trans_julia_dubnoff_trans_chris_childers_trans_anne_carson_walt_whitman_woman_waits_for_me.txt', 
    'place_rachel_by_matthew_arnold_paris_provence_french_riviera.txt'
]
# =================================================

def get_poem_excerpt(lines, exclude_idx=None):
    if not lines:
        return None

    # 1. Find potential starting lines based on punctuation priority
    period_lines = [i for i, line in enumerate(lines) if line.endswith('.')]
    punct_lines = [i for i, line in enumerate(lines) if line[-1] in string.punctuation]
    
    # Create a list of candidates based on priority
    if period_lines:
        candidates = period_lines
    elif punct_lines:
        candidates = punct_lines
    else:
        candidates = list(range(len(lines)))

    # Ensure we don't pick the same starting line as the previous excerpt
    candidates = [c for c in candidates if c != exclude_idx]
    
    if not candidates: # Fallback if only one line exists in the file
        start_idx = random.randint(0, len(lines) - 1)
    else:
        start_idx = random.choice(candidates)

    # 2. Copy 6 or 8 lines starting from that line
    num_lines = random.choice([4, 6])
    excerpt_lines = lines[start_idx : start_idx + num_lines]
    
    return "\n".join(excerpt_lines), start_idx

def main():
    all_excerpts = []

    for filename in FILE_LIST:
        path = os.path.join(SOURCE_FOLDER, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if not lines:
                continue

            # --- EXTRACT TWO EXCERPTS PER FILE ---
            first_excerpt, first_idx = get_poem_excerpt(lines)
            all_excerpts.append(first_excerpt)
            
            # Pass the first_idx to ensure the second one starts elsewhere
            second_excerpt, _ = get_poem_excerpt(lines, exclude_idx=first_idx)
            all_excerpts.append(second_excerpt)
            # -------------------------------------
            
        else:
            print(f"Warning: File {filename} not found in folder.")

    # Shuffle all collected excerpts (now 2x the number of files)
    random.shuffle(all_excerpts)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for i, excerpt in enumerate(all_excerpts):
            out_f.write(excerpt + "\n")
            
            # Stanza break logic: Alternately every 2nd or every 4th
            if i % 2 == 0:
                if (i + 1) % 2 == 0:
                    out_f.write("\n\n")
            else:
                if (i + 1) % 4 == 0:
                    out_f.write("\n\n")

    print(f"Done! Processed {len(all_excerpts)} excerpts into {OUTPUT_FILE}")

if __name__ == "__main__":
    main()