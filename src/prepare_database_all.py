import os
import faiss
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pickle

# Supported models
MODELS = {
    "resnet34": models.resnet34,
    "mobilenet_v2": models.mobilenet_v2,
    "efficientnet_b0": models.efficientnet_b0,
    "densenet121": models.densenet121,
    "inception_v3": models.inception_v3
}

# Initialize transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(model, image_path):
    """Extract normalized features from an image using the specified model."""
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
    """Prepare the database features for all models and save them."""
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    for model_name, model_fn in MODELS.items():
        print(f"Preparing database for model: {model_name}")

        # Load the model
        model = model_fn(pretrained=True).to(device)
        if hasattr(model, 'fc'):
            model.fc = torch.nn.Identity()  # Replace classification head
        elif hasattr(model, 'classifier'):
            model.classifier = torch.nn.Identity()
        model.eval()

        features = []
        image_paths = []

        # Walk through all subdirectories and files
        total_files = sum(len(files) for _, _, files in os.walk(folder_path))
        processed_files = 0

        for subdir, _, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(subdir, filename)
                    print(f"[{model_name}] Processing: {image_path} ({processed_files + 1}/{total_files})")
                    feature = extract_features(model, image_path)
                    if feature is not None:
                        features.append(feature)
                        image_paths.append(image_path)
                    processed_files += 1

        # Convert features to a NumPy array
        features = np.array(features)

        if features.size == 0:
            print(f"No valid features extracted for model '{model_name}'. Skipping.")
            continue

        # Create a FAISS index
        print(f"[{model_name}] Creating FAISS index...")
        try:
            res = faiss.StandardGpuResources()  # Use GPU resources if available
            index = faiss.IndexFlatL2(features.shape[1])
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(features)
            final_index = faiss.index_gpu_to_cpu(gpu_index)  # Save the index back to CPU
        except AttributeError:
            print(f"[{model_name}] FAISS GPU is not available. Falling back to CPU.")
            final_index = faiss.IndexFlatL2(features.shape[1])
            final_index.add(features)

        # Save the index and image paths
        print(f"[{model_name}] Saving FAISS index and image paths...")
        faiss.write_index(final_index, f"/home/anubhav/find-me-lens/database/database_index_{model_name}.faiss")
        with open(f"/home/anubhav/find-me-lens/database/image_paths_{model_name}.pkl", "wb") as f:
            pickle.dump(image_paths, f)

        print(f"[{model_name}] Feature extraction and index creation complete.")

if __name__ == "__main__":
    folder_path = "/home/anubhav/find-me-lens/data/ecommerce products"#input("Enter the path to the folder containing images: ").strip()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_database(folder_path)

