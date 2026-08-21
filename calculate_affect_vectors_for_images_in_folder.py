import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms

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

    return affect_vector.tolist()  # Convert to list for JSON serialization

def process_images_folder(folder_path, output_json, batch_size=32):
    """
    Process all images in a folder and save their affect vectors to JSON

    Args:
        folder_path: Path to folder containing images
        output_json: Path to output JSON file
        batch_size: Number of images to process at once (for efficiency)
    """
    # Initialize model and preprocessing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]

    results = []

    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]

        print(f"Processing batch {i//batch_size + 1}/{len(image_paths)//batch_size + 1}...")

        for img_path in batch_paths:
            try:
                # Extract features and calculate affect vector
                features = extract_image_features(img_path, model, preprocess, device)
                affect_vector = calculate_image_affect_vector(features)

                # Add to results
                results.append({
                    "image_path": img_path,
                    "affect_vector": affect_vector
                })

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

    # Save results to JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nProcessing complete! Saved {len(results)} image affect vectors to {output_json}")

if __name__ == "__main__":
    # image_folder = "costin_graphpoem_dhsi_shots" 
    # image_folder = "comm_singularity_graphpoem_dhsi_shots"
    image_folder = "graphpoem_dhsi_uvic_1_shots"
    # output_file = "costin_graphpoem_affect_vectors.json"
    # output_file = "comm_singularity_graphpoem_affect_vectors.json"
    output_file = "graphpoem_dhsi_uvic_1_affect_vectors.json"
    process_images_folder(image_folder, output_file)