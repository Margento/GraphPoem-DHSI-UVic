import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def load_data(poem_files, audio_files):
    """Load poem and audio data from JSON files"""
    poems = [json.load(open(f)) for f in poem_files]
    audios = [json.load(open(f)) for f in audio_files]
    return poems, audios

def extract_features(data, data_type):
    """Extract relevant features from poem or audio data"""
    features = []

    for item in data:
        # Audio features
        audio_feat = item['audio_features']
        feat = {
            'syllable_density': audio_feat['syllable_density'],
            'tempo': audio_feat['tempo'],
            'pacing_variance': audio_feat['pacing_variance'],
            'fricative_density': audio_feat['fricative_density'],
            'plosive_density': audio_feat['plosive_density'],
            'vocal_smoothness': audio_feat['vocal_smoothness'],
            'silence_ratio': audio_feat['silence_ratio']
        }

        # Temporal features
        temporal_feat = item['temporal_features']
        feat.update({
            'segments': temporal_feat['segments'],
            'motifs': len(temporal_feat.get('motifs', [])) if isinstance(temporal_feat.get('motifs'), list) else temporal_feat.get('motifs', 0),
            'uniques': temporal_feat.get('uniques', 0),
            'ruptures': len(temporal_feat.get('ruptures', [])) if isinstance(temporal_feat.get('ruptures'), list) else temporal_feat.get('ruptures', 0),
            'score_linear': temporal_feat['score_linear'],
            'score_cyclical': temporal_feat['score_cyclical'],
            'score_recursive': temporal_feat['score_recursive'],
            'score_hybrid': temporal_feat['score_hybrid']
        })

        # Add data type identifier
        feat['type'] = data_type

        features.append(feat)

    return features

def normalize_features(features):
    """Normalize features for comparison"""
    # Extract feature values
    values = [list(f.values())[:-1] for f in features]  # Exclude 'type'

    # Standardize features
    scaler = StandardScaler()
    normalized = scaler.fit_transform(values)

    # Create new feature dicts with normalized values
    normalized_features = []
    for i, f in enumerate(features):
        new_f = {}
        for j, key in enumerate(f.keys()):
            if key != 'type':
                new_f[key] = normalized[i][j]
            else:
                new_f[key] = f[key]
        normalized_features.append(new_f)

    return normalized_features

def calculate_similarity(poem_features, audio_features):
    """Calculate similarity between poems and audio files"""
    # Combine and normalize features
    all_features = poem_features + audio_features
    normalized = normalize_features(all_features)

    # Split back into poems and audios
    n_poems = len(poem_features)
    normalized_poems = normalized[:n_poems]
    normalized_audios = normalized[n_poems:]

    # Calculate cosine similarity
    poem_vectors = np.array([list(f.values())[:-1] for f in normalized_poems])
    audio_vectors = np.array([list(f.values())[:-1] for f in normalized_audios])

    similarities = cosine_similarity(poem_vectors, audio_vectors)

    return similarities

def find_best_matches(similarities, poem_files, audio_files):
    """Find best audio match for each poem"""
    matches = []

    for i, poem_sim in enumerate(similarities):
        best_audio_idx = np.argmax(poem_sim)
        similarity_score = poem_sim[best_audio_idx]

        matches.append({
            'poem': poem_files[i],
            'best_audio': audio_files[best_audio_idx],
            'similarity_score': similarity_score,
            'all_scores': {audio_files[j]: poem_sim[j] for j in range(len(audio_files))}
        })

    return matches

def main(poem_files, audio_files):
    # Load data
    poems, audios = load_data(poem_files, audio_files)

    # Extract features
    poem_features = extract_features(poems, 'poem')
    audio_features = extract_features(audios, 'audio')

    # Calculate similarities
    similarities = calculate_similarity(poem_features, audio_features)

    # Find best matches
    matches = find_best_matches(similarities, poem_files, audio_files)

    # Print results
    print("Best audio matches for each poem:")
    print("=" * 50)
    for match in matches:
        print(f"\nPoem: {match['poem']}")
        print(f"Best matching audio: {match['best_audio']}")
        print(f"Similarity score: {match['similarity_score']:.4f}")
        print("\nAll audio similarity scores:")
        for audio, score in match['all_scores'].items():
            print(f"  {audio}: {score:.4f}")

    return matches

if __name__ == "__main__":
    # Example usage - replace with your actual file paths
    poem_files = ["mermaid_beach_features.json", "post-singularity_collage_features.json", "hk_montreal_features.json"]
    audio_files = ["shot_0441_time_4099s_audio_analysis.json", "shot_0668_time_5646s_audio_analysis.json", "shot_0073_time_489s_audio_analysis.json"]

    matches = main(poem_files, audio_files)


# #graphpoem @ dhsi output

# Best audio matches for each poem:
# ==================================================

# Poem: mermaid_beach_features.json
# Best matching audio: shot_0073_time_489s_audio_analysis.json
# Similarity score: -0.6171

# All audio similarity scores:
#   shot_0441_time_4099s_audio_analysis.json: -0.6409
#   shot_0668_time_5646s_audio_analysis.json: -0.7328
#   shot_0073_time_489s_audio_analysis.json: -0.6171

# Poem: post-singularity_collage_features.json
# Best matching audio: shot_0668_time_5646s_audio_analysis.json
# Similarity score: -0.6776

# All audio similarity scores:
#   shot_0441_time_4099s_audio_analysis.json: -0.7944
#   shot_0668_time_5646s_audio_analysis.json: -0.6776
#  shot_0073_time_489s_audio_analysis.json: -0.7795

# Poem: hk_montreal_features.json
# Best matching audio: shot_0668_time_5646s_audio_analysis.json
# Similarity score: -0.7651

# All audio similarity scores:
#   shot_0441_time_4099s_audio_analysis.json: -0.8010
#   shot_0668_time_5646s_audio_analysis.json: -0.7651
#   shot_0073_time_489s_audio_analysis.json: -0.8123
