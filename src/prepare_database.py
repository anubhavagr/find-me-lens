import os
import faiss
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pickle

# Initialize the ResNet-34 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device)
model.fc = torch.nn.Identity()  # Remove the classification head to get feature embeddings
model.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    """Extract normalized features from an image using ResNet-34."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(image_tensor)
        
        # Normalize the feature vector
        return feature.cpu().numpy().squeeze() / np.linalg.norm(feature.cpu().numpy().squeeze())
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def prepare_database(folder_path):
    """Prepare the database features and save them to disk using FAISS."""
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    print(f"Processing images in folder and subfolders: {folder_path}")

    features = []
    image_paths = []

    # Walk through all subdirectories and files
    total_files = sum(len(files) for _, _, files in os.walk(folder_path))
    processed_files = 0

    for subdir, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(subdir, filename)
                print(f"Processing: {image_path} ({processed_files + 1}/{total_files})")
                feature = extract_features(image_path)
                if feature is not None:
                    features.append(feature)
                    image_paths.append(image_path)
                processed_files += 1

    # Convert features to a NumPy array
    features = np.array(features)

    if features.size == 0:
        print("No valid features extracted from the dataset. Exiting.")
        return

    # Create a FAISS index
    print("Creating FAISS index...")
    try:
        res = faiss.StandardGpuResources()  # Use GPU resources if available
        index = faiss.IndexFlatL2(features.shape[1])
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(features)
        final_index = faiss.index_gpu_to_cpu(gpu_index)  # Save the index back to CPU
    except AttributeError:
        print("FAISS GPU is not available. Falling back to CPU.")
        final_index = faiss.IndexFlatL2(features.shape[1])
        final_index.add(features)

    # Save the index and image paths
    print("Saving FAISS index and image paths...")
    faiss.write_index(final_index, "database_index.faiss")
    with open("image_paths.pkl", "wb") as f:
        pickle.dump(image_paths, f)

    print("Feature extraction and index creation complete. Files saved.")

if __name__ == "__main__":
    folder_path = "/home/anubhav/find-me-lens/data/ecommerce products" #input("Enter the path to the folder containing images: ").strip()
    prepare_database(folder_path)

