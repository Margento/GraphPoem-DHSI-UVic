import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def load_json_data(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_relevant_entries(json_data, mp4_folder):
    """
    Extract entries from JSON that match MP4 files in the folder
    Returns a dict mapping base filenames to their affect vectors
    """
    # Get all MP4 filenames without extension
    mp4_files = set()
    for f in os.listdir(mp4_folder):
        if f.endswith('.mp4'):
            base_name = os.path.splitext(f)[0]
            mp4_files.add(base_name)

    # Extract matching entries
    relevant_entries = {}
    for entry in json_data:
        # Extract base name from image_path
        image_name = os.path.basename(entry['image_path'])
        base_name = os.path.splitext(image_name)[0]

        if base_name in mp4_files:
            # Ensure affect_vector is a flat list
            affect_vec = entry['affect_vector']
            if isinstance(affect_vec, list) and len(affect_vec) == 1:
                affect_vec = affect_vec[0]  # Unwrap if nested
            relevant_entries[base_name] = {
                'affect_vector': affect_vec,
                'full_entry': entry
            }

    return relevant_entries

def compute_cosine_similarities(vectors1, vectors2):
    """
    Compute cosine similarities between two sets of vectors.
    Handles vectors that may be stored as [[x, y, z]] (nested) or [x, y, z].
    Returns a similarity matrix and mappings for row/column indices.
    """
    # ---------- Helper to get a flat 1‑D list ----------
    def flatten(vec):
        """
        Accepts a vector that can be:
          * [x, y, z]               -> returns the same list
          * [[x, y, z]] (nested)    -> returns the inner list
        Anything else raises a clear error.
        """
        if isinstance(vec, list):
            # If it is a list of length 1 and that element is also a list,
            # we assume the inner list is the actual vector.
            if len(vec) == 1 and isinstance(vec[0], list):
                return vec[0]
            return vec
        raise TypeError(f"Unexpected vector type: {type(vec)} – value: {vec}")

    # ---------- Build the two 2‑D arrays ----------
    # vectors1 may be a dict (mp4‑shots) or a list (image entries)
    if isinstance(vectors1, dict):
        arr1 = np.array([flatten(v['affect_vector']) for v in vectors1.values()], dtype=float)
    else:  # list
        arr1 = np.array([flatten(v['affect_vector']) for v in vectors1], dtype=float)

    # vectors2 may also be a dict or a list
    if isinstance(vectors2, dict):
        arr2 = np.array([flatten(v['affect_vector']) for v in vectors2.values()], dtype=float)
    else:
        arr2 = np.array([flatten(v['affect_vector']) for v in vectors2], dtype=float)

    # At this point arr1.shape → (n1, d) and arr2.shape → (n2, d)
    # If any vector had a wrong dimensionality we raise a helpful error.
    if arr1.ndim != 2 or arr2.ndim != 2:
        raise ValueError(
            f"After flattening, expected 2‑D arrays but got shapes {arr1.shape} and {arr2.shape}. "
            "Check that every affect_vector is a flat list of numbers."
        )

    # ---------- Cosine similarity ----------
    similarity_matrix = cosine_similarity(arr1, arr2)

    # ---------- Index mappings ----------
    if isinstance(vectors1, dict):
        index_to_name1 = {i: name for i, name in enumerate(vectors1.keys())}
    else:
        index_to_name1 = {i: v.get('image_name', f'img_{i}') for i, v in enumerate(vectors1)}

    if isinstance(vectors2, dict):
        index_to_name2 = {i: name for i, name in enumerate(vectors2.keys())}
    else:
        index_to_name2 = {
            i: os.path.splitext(os.path.basename(v['image_path']))[0]
            for i, v in enumerate(vectors2)
        }

    return similarity_matrix, index_to_name1, index_to_name2

def create_llm_friendly_output(similarity_matrix, index_to_name1, index_to_name2, vectors1, vectors2):
    """
    Create output format suitable for LLM processing
    """
    # Create a structured output
    output = {
        'similarity_matrix': similarity_matrix.tolist(),
        'row_indices': index_to_name1,
        'column_indices': index_to_name2,
        'similarity_pairs': []
    }

    # Create detailed similarity pairs
    for i, row_name in index_to_name1.items():
        for j, col_name in index_to_name2.items():
            similarity = similarity_matrix[i][j]

            # Get the full entries
            if isinstance(vectors1, dict):
                mp4_entry = vectors1[row_name]['full_entry']
            else:
                mp4_entry = vectors1[i]

            if isinstance(vectors2, dict):
                image_entry = vectors2[col_name]
            else:
                image_entry = vectors2[j]

            output['similarity_pairs'].append({
                'mp4_shot': row_name,
                'image': col_name,
                'similarity': float(similarity),
                'mp4_entry': mp4_entry,
                'image_entry': image_entry
            })

    return output

def main(mp4_folder, input_json_path, image_affect_json_path, output_json_path):
    """
    Main processing function
    """
    # 1. Load all data
    print("Loading data...")
    input_data = load_json_data(input_json_path)
    image_affect_data = load_json_data(image_affect_json_path)

    # 2. Extract relevant entries from input JSON
    print("Extracting relevant entries...")
    relevant_entries = extract_relevant_entries(input_data, mp4_folder)

    # 3. Compute cosine similarities
    print("Computing similarities...")
    similarity_matrix, row_indices, col_indices = compute_cosine_similarities(
        relevant_entries, image_affect_data
    )

    # 4. Create LLM-friendly output
    print("Preparing output...")
    output_data = create_llm_friendly_output(
        similarity_matrix, row_indices, col_indices,
        relevant_entries, image_affect_data
    )

    # 5. Save results
    print(f"Saving results to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("Processing complete!")

    return output_data


if __name__ == "__main__":
    config = {
        # 'mp4_folder': 'dhsi_collage_graphpoem_dhsi_extracted_clips',
        # 'mp4_folder': 'after_that_graphpoem_dhsi_extracted_clips',
        'mp4_folder': 'place_flagey_graphpoem_dhsi_extracted_clips',
        # 'input_json_path': 'dhsi_collage_sorted_images_by_affect.json',  # JSON with MP4-related affect vectors
        # 'input_json_path': 'after_that_sorted_images_by_affect.json',
        'input_json_path': 'place_flagey_sorted_images_by_affect.json',
        # 'image_affect_json_path': 'costin_graphpoem_affect_vectors.json',  # JSON from previous image processing
        # 'image_affect_json_path':'comm_singularity_graphpoem_affect_vectors.json',
        'image_affect_json_path': 'graphpoem_dhsi_uvic_1_affect_vectors.json',
        'output_json_path': 'graphpoem_dhsi_uvic_place_flagey_similarity_results.json'  # Output file
    }

    # Run the processing
    results = main(**config)

    # Print summary
    print("\nSummary:")
    print(f"Processed {len(results['row_indices'])} MP4 shots")
    print(f"Compared with {len(results['column_indices'])} images")
    print(f"Generated {len(results['similarity_pairs'])} similarity comparisons")