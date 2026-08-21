import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# Cache directory setup
CACHE_DIR = Path("graphpoem_dhsi_image_affect_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_image_cache_path(image_path):
    """Generate a cache path based on image path hash"""
    image_hash = hashlib.md5(str(image_path).encode()).hexdigest()
    return CACHE_DIR / f"{image_hash}.json"

def load_cached_affect_vector(image_path):
    """Try to load cached affect vector for an image"""
    cache_path = get_image_cache_path(image_path)
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Verify the cache is for the correct image
                if data.get('image_path') == str(image_path):
                    print(f"✅ Using cached affect vector for {image_path.name}")
                    return np.array(data['affect_vector'])
        except (json.JSONDecodeError, KeyError):
            pass
    return None

def save_affect_vector_cache(image_path, affect_vector):
    """Save affect vector to cache"""
    cache_path = get_image_cache_path(image_path)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump({
            'image_path': str(image_path),
            'affect_vector': affect_vector.tolist()
        }, f, indent=2)
    print(f"💾 Cached affect vector for {image_path.name}")

def load_stanza_affect_vector(json_path):
    """Load the affect vector from a previously saved JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array([
        data['affect_vector']['valence'],
        data['affect_vector']['arousal'],
        data['affect_vector']['energy']
    ]).reshape(1, -1)

def setup_image_model():
    """Initialize the image feature extraction model"""
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Remove the final classification layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    # Device selection with MPS support
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Silicon GPU via MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (no GPU acceleration available)")

    model = model.to(device)

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return model, preprocess, device

def extract_image_features(image_path, model, preprocess, device):
    """Extract features from an image using ResNet50"""
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    with torch.no_grad():
        features = model(batch_t)

    # Convert to numpy array and flatten
    return features.squeeze().cpu().numpy()

def calculate_image_affect_vector(image_features):
    """Convert image features to an affect vector"""
    # Normalize features
    normalized = image_features / np.linalg.norm(image_features)

    # Take first 3 components as our affect vector
    affect_vector = normalized[:3].reshape(1, -1)

    # Ensure the values are in reasonable ranges
    affect_vector = np.tanh(affect_vector * 2)  # Scale to [-1, 1] range

    return affect_vector

def process_image_folder(folder_path, folder_id, model, preprocess, device):
    """Process all images in a folder and return their affect vectors with folder info"""
    image_paths = [p for p in Path(folder_path).glob('*.jpg')]
    results = []

    for img_path in image_paths:
        try:
            # Try to load from cache first
            cached_vector = load_cached_affect_vector(img_path)
            if cached_vector is not None:
                results.append({
                    'folder_id': folder_id,
                    'folder_path': str(folder_path),
                    'image_path': str(img_path),
                    'image_name': img_path.name,
                    'affect_vector': cached_vector.tolist()[0],
                    'source': 'cache'
                })
                continue

            # If not in cache, calculate
            print(f"🔍 Processing {img_path.name}...")
            features = extract_image_features(img_path, model, preprocess, device)
            affect_vec = calculate_image_affect_vector(features)

            # Save to cache
            save_affect_vector_cache(img_path, affect_vec)

            results.append({
                'folder_id': folder_id,
                'folder_path': str(folder_path),
                'image_path': str(img_path),
                'image_name': img_path.name,
                'affect_vector': affect_vec.tolist()[0],
                'source': 'calculated'
            })
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

    return results

def sort_images_by_similarity(stanza_affect, image_data):
    """Sort images by cosine similarity to the stanza's affect vector"""
    # Convert stanza affect to numpy array
    stanza_vec = np.array(stanza_affect)

    # Calculate similarities
    similarities = []
    for item in image_data:
        img_vec = np.array(item['affect_vector']).reshape(1, -1)
        similarity = cosine_similarity(stanza_vec, img_vec)[0][0]
        similarities.append({
            'folder_id': item['folder_id'],
            'folder_path': item['folder_path'],
            'image_path': item['image_path'],
            'image_name': item['image_name'],
            'similarity': similarity,
            'affect_vector': item['affect_vector'],
            'source': item.get('source', 'unknown')
        })

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)

    return similarities

def main():
    # 1. Load your stanza's affect vector
    # stanza_json_path = "after_that_features.json"
    # stanza_json_path = "place_flagey_features.json"
    stanza_json_path = "dhsi_collage_features.json"
    stanza_affect = load_stanza_affect_vector(stanza_json_path)

    # 2. Setup image processing
    model, preprocess, device = setup_image_model()

    # 3. Process all image folders with folder IDs
    image_folders = {
        1: "graphpoem_dhsi20_extracted_shots",
        2: "graphpoem_dhsi21_extracted_shots",
        3: "graphpoem_dhsi22_extracted_shots",
        4: "graphpoem_dhsi23_extracted_shots"
    }

    all_images = []
    for folder_id, folder_path in image_folders.items():
        print(f"\n📁 Processing folder {folder_id}: {folder_path}")
        folder_images = process_image_folder(folder_path, folder_id, model, preprocess, device)
        all_images.extend(folder_images)

    # 4. Sort images by similarity
    print("\n🔄 Sorting images by similarity...")
    sorted_images = sort_images_by_similarity(stanza_affect, all_images)

    # 5. Save results
    # output_path = "after_that_sorted_images_by_affect.json"
    # output_path = "place_flagey_sorted_images_by_affect.json"
    output_path = "dhsi_collage_sorted_images_by_affect.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_images, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {output_path}")
    print("\n🏆 Top 10 most similar images:")
    for i, img in enumerate(sorted_images[:10]):
        source_emoji = "💾" if img.get('source') == 'cache' else "🔍"
        print(f"{i+1}. {source_emoji} Folder {img['folder_id']}: {img['image_name']} "
              f"(similarity: {img['similarity']:.3f})")

if __name__ == "__main__":
    main()