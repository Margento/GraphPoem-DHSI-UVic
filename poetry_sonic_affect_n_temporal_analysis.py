import time, random
# from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np
import librosa
# from moviepy.editor import *
# from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

import pyo
from pyo import *
import re, unicodedata, math
import numpy as np
from collections import Counter
# Download Unicode Scripts.txt (nned to run only once; you can also cache this locally)
SCRIPTS_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Scripts.txt"


# ALL SCRIPTS 'UNDER THE SUN' [IN UNICODE, THAT IS]

import regex
import urllib.request


def get_all_scripts() -> set[str]:
    """
    Fetch the official Unicode script names from Scripts.txt.
    """
    with urllib.request.urlopen(SCRIPTS_URL) as f:
        lines = f.read().decode("utf-8").splitlines()

    scripts = set()
    for line in lines:
        if line.strip() and not line.startswith("#"):
            # Example line: "0041..005A; Latin # L&  [26] LATIN CAPITAL LETTER A..Z"
            parts = line.split(";")
            if len(parts) >= 2:
                script = parts[1].strip().split()[0]
                scripts.add(script)
    return scripts

UNICODE_SCRIPTS = sorted(get_all_scripts())

def char_script(ch):
    import regex
    
    if not ch or len(ch) != 1:
        return "INVALID"

    for script in UNICODE_SCRIPTS:
        try:
            # Use the script name exactly as Unicode defines it
            if regex.match(rf"\p{{Script={script}}}", ch):
                return script  # return it as-is
        except regex.error:
            continue  # skip invalid/unrecognized scripts

    return "UNKNOWN"


from collections import Counter

def word_script(word: str) -> str:
    """
    Return the dominant script of a word (based on majority of alphabetic chars).
    """
    scripts = Counter(char_script(ch) for ch in word if ch.isalpha())
    return scripts.most_common(1)[0][0] if scripts else "OTHER"


def get_unicode_name(ch):
    try:
        return unicodedata.name(ch)
    except ValueError:
        return None


import unicodedata

# Latin/Cyrillic/Greek/Devanagari vowels (extendable)
_vowel_re_latin = re.compile(r"[aeiouy\u00E0-\u00FF]+", re.IGNORECASE)
_vowel_re_cyrillic = re.compile(r"[аеёиоуыэюя]+", re.IGNORECASE)  # basic Russian vowels
_vowel_re_greek = re.compile(r"[αεηιουωάέήίόύώ]", re.IGNORECASE)   # modern Greek vowels
_vowel_re_devanagari = re.compile(r"[अआइईउऊएऐओऔऋॠॡॢॣ]", re.IGNORECASE)


def approx_syllables_word(word: str) -> int:
    if not word:
        return 0
    w = unicodedata.normalize("NFC", word)
    script = word_script(w)

    if script == "LATIN":
        groups = _vowel_re_latin.findall(w)
        count = len(groups)
        if w.lower().endswith("e") and count > 1:  # silent 'e'
            count -= 1
        return max(1, count)

    if script == "CYRILLIC":
        groups = _vowel_re_cyrillic.findall(w)
        return max(1, len(groups))

    if script == "GREEK":
        groups = _vowel_re_greek.findall(w)
        return max(1, len(groups))

    if script == "DEVANAGARI":
        groups = _vowel_re_devanagari.findall(w)
        return max(1, len(groups))

    if script in ("HIRAGANA", "KATAKANA"):
        kana_chars = [ch for ch in w if '\u3040' <= ch <= '\u30FF']
        return max(1, len(kana_chars))

    if script == "HANGUL":
        return len([ch for ch in w if '\uAC00' <= ch <= '\uD7A3'])

    if script == "CJK":
        chars = [ch for ch in w if '\u4E00' <= ch <= '\u9FFF']
        return max(1, len(chars))

    if script == "THAI":
        return max(1, len([ch for ch in w if ch.strip()]))

    # Fallback
    groups = _vowel_re_latin.findall(w)
    return max(1, len(groups) if groups else len(w))


def extract_phonological_clusters(word: str):
    clusters = set()
    w = unicodedata.normalize("NFC", word.lower())
    script = word_script(w)

    if script in ("LATIN", "GREEK", "CYRILLIC"):
        consonant_matches = re.findall(r'[^aeiouy]+', w)
        for c in consonant_matches:
            for i in range(len(c)):
                for j in range(i+1, len(c)+1):
                    clusters.add(c[i:j])
        vowel_matches = re.findall(r'[aeiouy]+', w)
        for v in vowel_matches:
            for i in range(len(v)):
                for j in range(i+1, len(v)+1):
                    clusters.add(v[i:j])
        for k in range(2, 5):
            if len(w) >= k:
                clusters.add(w[-k:])

    elif script in ("ARABIC", "HEBREW"):
        consonant_runs = re.findall(r'[^aeiou]+', w)
        for c in consonant_runs:
            for i in range(len(c)):
                for j in range(i+1, len(c)+1):
                    clusters.add(c[i:j])
        for k in range(2, 5):
            if len(w) >= k:
                clusters.add(w[-k:])

    elif script == "DEVANAGARI":
        groups = _vowel_re_devanagari.findall(w)
        for g in groups:
            clusters.add(g)
        for k in range(2, 5):
            if len(w) >= k:
                clusters.add(w[-k:])

    elif script in ("HIRAGANA", "KATAKANA", "HANGUL", "CJK"):
        chars = list(w)
        clusters.update(chars)
        for i in range(len(chars)-1):
            clusters.add(chars[i] + chars[i+1])

    else:
        for i in range(len(w)):
            for j in range(i+1, min(i+4, len(w))+1):
                clusters.add(w[i:j])

    return clusters


_word_re = re.compile(r"\w+", re.UNICODE)

def tokenize_text(text):
    tokens = []
    for m in _word_re.finditer(text):
        tok = m.group(0)
        tokens.append(tok)
    return tokens

_fricatives = set(list("fvsz") + ["sh","zh","th"])
_plosives = set(list("pbtdkg"))

def phonetic_density(tokens):
    latin_tokens = [t for t in tokens if char_script(t[0]) == "LATIN"]
    joined = " ".join(latin_tokens).lower()
    letters = re.sub(r'[^a-z]', '', joined)
    if not letters:
        return 0.0, 0.0, 0.0
    fric_count = sum(joined.count(f) for f in ["f","v","s","z","sh","zh","th"])
    plos_count = sum(joined.count(p) for p in ["p","b","t","d","k","g"])
    vowel_count = sum(1 for c in letters if c in "aeiouy")
    total = len(letters)
    return fric_count/total, plos_count/total, vowel_count/(total+1e-9)

import re
import nltk
from nltk.corpus import words, stopwords, wordnet as wn
from nltk.corpus.reader import WordListCorpusReader
import numpy as np

# Download necessary NLTK resources
nltk.download('words')
nltk.download('stopwords')
nltk.download('omw')
nltk.download('omw-1.4')

# Load English words and stopwords
english_words = set(words.words())
stop_words = set(stopwords.words('english'))

# Define common prefixes and suffixes
PL_PREFIXES = {"re", "un", "in", "dis", "pre", "sub"}
PL_SUFFIXES = {"ing", "ed", "er", "ly", "es", "ful"}

def is_plausible_fragment(fragment):
    """Check if fragment is a plausible English word, prefix/suffix, or foreign fragment."""
    fragment = fragment.lower()
    if not fragment:
        return False
    if fragment in english_words:
        return True
    if fragment in PL_PREFIXES or fragment in PL_SUFFIXES:
        return True
    # Check if fragment exists in WordNet for any language
    for lang in wn.langs():
        if wn.synsets(fragment, lang=lang):
            return True
    # Fallback: accept fragments that are at least 2 characters long
    if len(fragment) > 1:
        return True
    return False

def extract_audio_features_from_stanza(stanza, expected_feet_per_line=(5,6), foot_syllables=(2,3)):
    lines = [ln.strip() for ln in stanza.strip().split("\n") if ln.strip()]
    n_lines = max(1, len(lines))
    tokens = tokenize_text(stanza)
    syll_counts_tokens = [approx_syllables_word(t) for t in tokens]
    total_syllables = sum(syll_counts_tokens)
    n_words = len(tokens) if tokens else 1
    syllable_density = total_syllables / n_words if n_words else 0.0
    target_feet = np.mean(expected_feet_per_line)
    avg_syll_per_line = total_syllables / max(1, len(lines))
    avg_foot_syll = np.mean(foot_syllables)
    tempo = avg_syll_per_line / avg_foot_syll
    sylls_per_line = [sum(approx_syllables_word(t) for t in tokenize_text(ln)) for ln in lines]
    pacing_variance = float(np.var(sylls_per_line)) if sylls_per_line else 0.0
    fric_density, plos_density, vowel_ratio = phonetic_density(tokens)
    vocal_smoothness = float(vowel_ratio)

    # --- [word-splitting] enjambment detection (needed in this specific case; if you need to process enjambments in general see https://github.com/Margento/Computationally_Assembled_Belgian_Poetry_Anthology ---
    enjambments = 0
    enjambed_positions = set()

    for ln_idx, ln in enumerate(lines):
        # 1. End-of-line split (including ellipses)
        end_match = re.search(r'(\w+(?:\.\.\.)?)-/?(\w*)$', ln)
        if end_match:
            left, right = end_match.groups()
            if is_plausible_fragment(left) and (not right or is_plausible_fragment(right)):
                enjambments += 1
                enjambed_positions.add(end_match.start())

        # 2. Start-of-line split
        if ln_idx > 0:
            start_match = re.match(r'^(\w*)-/(\w+)', ln)
            if start_match:
                left, right = start_match.groups()
                if (not left or is_plausible_fragment(left)) and is_plausible_fragment(right):
                    enjambments += 1
                    enjambed_positions.add(start_match.start())

        # 3. Multi-word or foreign-word consideration (fallback)
        for match in re.finditer(r'(\S+)/(\S+)', ln):
            left, right = match.groups()
            if is_plausible_fragment(left) and is_plausible_fragment(right):
                enjambments += 1
                enjambed_positions.add(match.start())

    # Count pause marks excluding those part of valid enjambments
    pause_marks = 0
    for m in re.finditer(r'[,;:\-\—\(\)]', stanza):
        if m.start() not in enjambed_positions:
            pause_marks += 1

    silence_ratio = pause_marks / (total_syllables + 1e-9)
    caesura = sum(1 for ln in lines if "," in ln or ";" in ln or "—" in ln)

    enjambments_norm = enjambments / n_lines
    caesura_norm = caesura / n_lines

    audio = {
        "syllable_density": float(syllable_density),
        "tempo": float(tempo),
        "pacing_variance": float(pacing_variance),
        "fricative_density": float(fric_density),
        "plosive_density": float(plos_density),
        "vocal_smoothness": float(vocal_smoothness),
        "silence_ratio": float(silence_ratio),
        "total_syllables": int(total_syllables),
        "sylls_per_line": sylls_per_line,
        "enjambments": float(enjambments_norm),
        "caesura": float(caesura_norm),
        "n_words": n_words
    }
    return audio


import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon GPU via MPS")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available, falling back to CPU")


import math

from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification, pipeline
# import torch

# Model + tokenizer
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)  # force slow tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device
)

import math

def stanza_affect_vector(stanza):
    """
    Extract affective features (valence, arousal, energy) from a stanza of text,
    combining multilingual sentiment analysis with audio-like features.
    """

    # --- 1. Multilingual Sentiment Analysis (Hugging Face) ---
    try:
        sentiment_result = sentiment_pipeline(stanza[:512])[0]  # truncate to model limit
        label = sentiment_result["label"].lower()
        score = sentiment_result["score"]

        # Map labels to a polarity value in [-1, 1]
        if "negative" in label:
            polarity = -score
        elif "positive" in label:
            polarity = score
        else:  # neutral
            polarity = 0.0
    except Exception as e:
        print(f"Sentiment analysis failed: {e}")
        polarity = 0.0

    # Calculate valence from polarity
    valence = float(math.tanh(polarity * 5.0))

    # --- 2. Extract "audio" features ---
    audio_feats = extract_audio_features_from_stanza(stanza)

    # --- 3. Calculate arousal & energy ---
    arousal = (audio_feats["pacing_variance"] ** 0.5
               + audio_feats["fricative_density"] * 0.5
               + min(1.0, audio_feats["silence_ratio"] * 2.0))
    arousal = float(math.tanh(arousal))

    energy = float(math.tanh(
        (audio_feats["tempo"] * 0.6) +
        (audio_feats["syllable_density"] * 0.2)
    ))

    return {
        "valence": valence,
        "arousal": arousal,
        "energy": energy,
        # "audio_feats": audio_feats
    }
# TEMPORAL FEATURES
def stanza_temporal_structures(stanza):
    lines = [ln.strip() for ln in stanza.strip().split("\n") if ln.strip()]
    n_segments = len(lines)
    segment_annotations = []
    motifs_counter = Counter()
    
    # process clusters instead of whole words
    for i, ln in enumerate(lines):
        words = tokenize_text(ln)
        segment_annotations.append(
            f"line_{i+1}: {len(words)} words, {sum(approx_syllables_word(w) for w in words)} sylls"
        )
        for w in words:
            clusters = extract_phonological_clusters(w)
            for c in clusters:
                motifs_counter[c] += 1
    
    motifs = [cl for cl,cnt in motifs_counter.items() if cnt > 1]
    n_motifs = len(motifs)
    n_uniques = len(motifs_counter)
    
    sylls_per_line = [
        sum(approx_syllables_word(w) for w in tokenize_text(ln)) for ln in lines
    ] if lines else []
    
    ruptures = []
    if sylls_per_line:
        mean = np.mean(sylls_per_line); sd = np.std(sylls_per_line)
        for i, s in enumerate(sylls_per_line):
            if sd > 0 and abs(s-mean) > 1.5*sd:
                ruptures.append({
                    "line": i+1, 
                    "syllables": int(s), 
                    "deviation": float((s-mean)/sd)
                })
    
    score_linear = 0.0
    if len(sylls_per_line) > 1:
        x = np.arange(len(sylls_per_line))
        y = np.array(sylls_per_line)
        cov = np.cov(x, y)[0,1]
        if np.std(x) > 0 and np.std(y) > 0:
            score_linear = float(cov / (np.std(x) * np.std(y)))
    
    score_cyclical = 0.0
    if len(sylls_per_line) > 2:
        y = np.array(sylls_per_line) - np.mean(sylls_per_line)
        score_cyclical = float(np.correlate(y, np.roll(y,1))[0] / (np.sum(y*y)+1e-9))
    
    ngrams = Counter()
    for ln in lines:
        toks = [t.lower() for t in tokenize_text(ln)]
        for i in range(len(toks)-1):
            ngrams[" ".join(toks[i:i+2])] += 1
    repeated_ngrams = sum(1 for c in ngrams.values() if c>1)
    score_recursive = float(repeated_ngrams / (len(ngrams)+1e-9))
    score_hybrid = float((abs(score_linear) + abs(score_cyclical) + score_recursive)/3.0)
    
    recursive_events = motifs[:5]
    
    return {
        "segments": n_segments,
        "segment_annotations": segment_annotations,
        "number_of_motifs": n_motifs,
        "motifs": motifs,
        "uniques": n_uniques,
        "ruptures": ruptures,
        "score_linear": round(score_linear, 3),
        "score_cyclical": round(score_cyclical, 3),
        "score_recursive": round(score_recursive, 3),
        "score_hybrid": round(score_hybrid, 3),
        "recursive_events": recursive_events
    }

def extract_full_stanza_representation(stanza):
    audio_feats =  extract_audio_features_from_stanza(stanza)
    affect = stanza_affect_vector(stanza)
    temporal = stanza_temporal_structures(stanza)

    return {
        "audio_features": audio_feats,
        "affect_vector": affect,
        "temporal_features": temporal,
    }


import json
import os
from pathlib import Path

def process_stanza_file(input_file_path, output_file_path=None):
    """
    Read a stanza from a text file, process it, and save the output as JSON.

    Args:
        input_file_path (str): Path to the input text file containing the stanza
        output_file_path (str, optional): Path to save the JSON output.
                                        If None, uses input filename with .json extension
    """
    # Read the stanza from file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        stanza = f.read().strip()

    # Process the stanza
    result = extract_full_stanza_representation(stanza)

    # Determine output path if not provided
    if output_file_path is None:
        input_path = Path(input_file_path)
        output_file_path = input_path.with_suffix('.json')

    # Save the result as JSON
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Processed stanza saved to: {output_file_path}")
    return result


# process_stanza_file("draft_65_that_by_blau_duplessis.txt", "that_features.json")

# process_stanza_file("after_that_by_margento.txt", "after_that_features.json")

# process_stanza_file("place_flagey_by_margento.txt", "place_flagey_features.json")

# process_stanza_file("dhsi_courses_collage_poem.txt", "dhsi_collage_features.json")

# process_stanza_file("3086153_mermaid_beach_by_margento.txt", "mermaid_beach_features.json")

# process_stanza_file("graphpoem_dhsi23_post-singularity_shuffled_excerpt_collage.txt", "post-singularity_collage_features.json")

process_stanza_file("dhsi_hk_montreal.txt", "hk_montreal_features.json")