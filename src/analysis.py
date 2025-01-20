import os
import time
import numpy as np
import pickle
import faiss
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(model, image):
    """Extract normalized features from an image using the specified model."""
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image_tensor)
    return feature.cpu().numpy().squeeze() / np.linalg.norm(feature.cpu().numpy().squeeze())

def load_model_and_data(model_name):
    """Load the model, FAISS index, and image paths for the selected architecture."""
    if model_name not in MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Supported models are: {list(MODELS.keys())}")

    # Load the model
    model = MODELS[model_name](pretrained=True).to(device)
    if hasattr(model, 'fc'):
        model.fc = torch.nn.Identity()
    elif hasattr(model, 'classifier'):
        model.classifier = torch.nn.Identity()
    model.eval()

    # Load the FAISS index and image paths
    index = faiss.read_index(f"database_index_{model_name}.faiss")
    with open(f"image_paths_{model_name}.pkl", "rb") as f:
        image_paths = pickle.load(f)

    return model, index, image_paths

def prepare_query_and_ground_truth(dataset_folder):
    """Prepare query images and ground truths from the dataset folder structure."""
    query_images = []
    ground_truths = []

    # Iterate over each subfolder
    for subfolder in os.listdir(dataset_folder):
        subfolder_path = os.path.join(dataset_folder, subfolder)
        if os.path.isdir(subfolder_path):
            ground_truth = []
            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(subfolder_path, filename)
                    ground_truth.append(image_path)
            
            # Use one image as query; the rest are ground truth
            if ground_truth:
                query_image = Image.open(ground_truth[0]).convert("RGB")
                query_images.append(query_image)
                ground_truths.append(ground_truth)

    return query_images, ground_truths

def evaluate_model(model_name, query_images, ground_truths):
    """Evaluate the model with precision, recall, F1-score, accuracy, and inference time using top-1 prediction."""
    print(f"Evaluating model: {model_name}...")
    model, index, image_paths = load_model_and_data(model_name)
    gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

    total_time = 0
    y_true = []  # Ground truth (1 if the top-1 prediction is correct, else 0)
    y_pred = []  # Predicted labels (always 1 since we are taking top-1 prediction)

    for query_image, ground_truth in zip(query_images, ground_truths):
        # Extract features and measure inference time
        start_time = time.time()
        query_feature = extract_features(model, query_image)
        distances, indices = gpu_index.search(np.expand_dims(query_feature, axis=0), 1)  # Top-1 prediction
        total_time += time.time() - start_time

        # Get the best match
        best_match = image_paths[indices[0][0]]

        # Append true and predicted labels
        y_true.append(1 if best_match in ground_truth else 0)
        y_pred.append(1)  # Always 1 since we are making a prediction

    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = np.mean(y_true)
    avg_inference_time = total_time / len(query_images)

    print(f"Model: {model_name}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Avg Inference Time: {avg_inference_time:.4f} seconds\n")

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "inference_time": avg_inference_time
    }

def main():
    # Specify dataset folder containing subfolders
    dataset_folder = "/home/anubhav/find-me-lens/data/ecommerce products"  # Update with your dataset folder path
    query_images, ground_truths = prepare_query_and_ground_truth(dataset_folder)

    results = []

    # Evaluate each model
    for model_name in MODELS.keys():
        result = evaluate_model(model_name, query_images, ground_truths)
        results.append(result)

    # Display comparison results
    print("Model Comparison:")
    for res in results:
        print(f"Model: {res['model']}")
        print(f"  Precision: {res['precision']:.4f}")
        print(f"  Recall: {res['recall']:.4f}")
        print(f"  F1-Score: {res['f1_score']:.4f}")
        print(f"  Accuracy: {res['accuracy']:.4f}")
        print(f"  Inference Time: {res['inference_time']:.4f} seconds\n")

if __name__ == "__main__":
    main()

