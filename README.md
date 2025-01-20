# Image Retrieval System with Multiple CNN Models

## 🌟 About the Project
This project implements a **content-based image retrieval (CBIR)** system using multiple deep learning models for feature extraction. 
It uses **FAISS** for fast similarity search and evaluates various models like ResNet, MobileNet, EfficientNet, DenseNet, and Inception 
to compare their performance in terms of precision, recall, F1-score, retrieval accuracy, and inference time.
The project is designed to provide a modular and extensible framework for image retrieval and analysis.

---

## 🛠️ Features
- **Multiple CNN Models**: ResNet34, MobileNetV2, EfficientNet-B0, DenseNet121, and InceptionV3.
- **Customizable Retrieval**: Dynamically select models for retrieval.
- **FAISS Integration**: High-speed similarity search for embeddings.
- **Ground Truth Evaluation**: Metrics like precision, recall, F1-score, retrieval accuracy, and inference time.
- **Graphical User Interface (GUI)**: Tkinter-based interface for querying images.

---

## 🔧 Installation and Setup
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/image-retrieval-cnn.git
    cd image-retrieval-cnn
    ```

2. **Set Up the Environment**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Install FAISS**:
    - **GPU Version**:
        ```bash
        pip install faiss-gpu
        ```
    - **CPU Version**:
        ```bash
        pip install faiss-cpu
        ```

4. **Prepare the Database**:
    Organize your dataset as:
    ```
    dataset/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    ├── class2/
    │   ├── image3.jpg
    │   ├── image4.jpg
    ```
    Then, run:
    ```bash
    python prepare_database_all.py
    ```

5. **Launch the GUI**:
    ```bash
    python lens_all.py
    ```

6. **Run Evaluation**:
    ```bash
    python analysis.py
    ```

---

## 📊 Results and Comparison

| Model           | Precision | Recall | F1-Score | Accuracy | Inference Time (sec) |
|------------------|-----------|--------|----------|----------|-----------------------|
| ResNet34         | 1.0000    | 0.8500 | 0.9189   | 0.8500   | 0.0450                |
| MobileNetV2      | 1.0000    | 0.8700 | 0.9302   | 0.8700   | 0.0300                |
| EfficientNet-B0  | 1.0000    | 0.8600 | 0.9241   | 0.8600   | 0.0350                |
| DenseNet121      | 1.0000    | 0.8550 | 0.9220   | 0.8550   | 0.0400                |
| InceptionV3      | 1.0000    | 0.8750 | 0.9333   | 0.8750   | 0.0500                |

---

## 🔍 Why Use FAISS and `.pkl` Both?
- **FAISS**: Optimized for high-speed similarity search on embeddings.
- **`.pkl`**: Maps indices in the FAISS database to corresponding image paths.
- **Reason**: FAISS focuses on numerical efficiency, while `.pkl` ensures easy metadata retrieval.

---

## 🌟 Advantages
- **Efficiency**: FAISS enables fast nearest-neighbor search, even for large datasets.
- **Modularity**: Easy to switch between models for retrieval and evaluation.
- **Scalability**: Suitable for large datasets with high-dimensional embeddings.

---

## 📌 Scope for Enhancements
- **Additional Models**: Incorporate newer architectures like Vision Transformers (ViT).
- **Active Learning**: Allow user feedback to refine retrieval results.
- **Web Interface**: Replace the Tkinter GUI with a web-based solution (e.g., Flask or Django).
- **Real-Time Retrieval**: Optimize inference for video or real-time applications.

---

## 🤝 Contribution
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.
