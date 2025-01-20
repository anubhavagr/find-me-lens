<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width-width, initial-scale=1.0">
    <title>README</title>
</head>
<body>
    <h1>Image Retrieval System with Multiple CNN Models</h1>

    <h2>🌟 About the Project</h2>
    <p>
        This project implements a <strong>content-based image retrieval (CBIR)</strong> system using multiple deep learning models for feature extraction. 
        It uses <strong>FAISS</strong> for fast similarity search and evaluates various models like ResNet, MobileNet, EfficientNet, DenseNet, and Inception 
        to compare their performance in terms of precision, recall, F1-score, retrieval accuracy, and inference time.
        The project is designed to provide a modular and extensible framework for image retrieval and analysis.
    </p>

    <hr>

    <h2>🛠️ Features</h2>
    <ul>
        <li><strong>Multiple CNN Models:</strong> ResNet34, MobileNetV2, EfficientNet-B0, DenseNet121, and InceptionV3.</li>
        <li><strong>Customizable Retrieval:</strong> Dynamically select models for retrieval.</li>
        <li><strong>FAISS Integration:</strong> High-speed similarity search for embeddings.</li>
        <li><strong>Ground Truth Evaluation:</strong> Metrics like precision, recall, F1-score, retrieval accuracy, and inference time.</li>
        <li><strong>Graphical User Interface (GUI):</strong> Tkinter-based interface for querying images.</li>
    </ul>

    <hr>

    <h2>🔧 Installation and Setup</h2>
    <ol>
        <li><strong>Clone the Repository:</strong>
            <pre><code>git clone https://github.com/yourusername/image-retrieval-cnn.git
cd image-retrieval-cnn
            </code></pre>
        </li>
        <li><strong>Set Up the Environment:</strong>
            <pre><code>pip install -r requirements.txt</code></pre>
        </li>
        <li><strong>Install FAISS:</strong>
            <ul>
                <li><strong>GPU Version:</strong>
                    <pre><code>pip install faiss-gpu</code></pre>
                </li>
                <li><strong>CPU Version:</strong>
                    <pre><code>pip install faiss-cpu</code></pre>
                </li>
            </ul>
        </li>
        <li><strong>Prepare the Database:</strong>
            <p>Organize your dataset as follows:</p>
            <pre><code>
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
├── class2/
│   ├── image3.jpg
│   ├── image4.jpg
            </code></pre>
            <p>Then run:</p>
            <pre><code>python prepare_database_all.py</code></pre>
        </li>
        <li><strong>Launch the GUI:</strong>
            <pre><code>python lens_all.py</code></pre>
        </li>
        <li><strong>Run Evaluation:</strong>
            <pre><code>python analysis.py</code></pre>
        </li>
    </ol>

    <hr>

    <h2>📊 Results and Comparison</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Accuracy</th>
                <th>Inference Time (sec)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>ResNet34</td>
                <td>1.0000</td>
                <td>0.8500</td>
                <td>0.9189</td>
                <td>0.8500</td>
                <td>0.0450</td>
            </tr>
            <tr>
                <td>MobileNetV2</td>
                <td>1.0000</td>
                <td>0.8700</td>
                <td>0.9302</td>
                <td>0.8700</td>
                <td>0.0300</td>
            </tr>
            <tr>
                <td>EfficientNet-B0</td>
                <td>1.0000</td>
                <td>0.8600</td>
                <td>0.9241</td>
                <td>0.8600</td>
                <td>0.0350</td>
            </tr>
            <tr>
                <td>DenseNet121</td>
                <td>1.0000</td>
                <td>0.8550</td>
                <td>0.9220</td>
                <td>0.8550</td>
                <td>0.0400</td>
            </tr>
            <tr>
                <td>InceptionV3</td>
                <td>1.0000</td>
                <td>0.8750</td>
                <td>0.9333</td>
                <td>0.8750</td>
                <td>0.0500</td>
            </tr>
        </tbody>
    </table>

    <hr>

    <h2>🔍 Why Use FAISS and `.pkl` Both?</h2>
    <ul>
        <li><strong>FAISS:</strong> Optimized for high-speed similarity search on embeddings.</li>
        <li><strong>.pkl:</strong> Maps indices in the FAISS database to corresponding image paths.</li>
        <li><strong>Reason:</strong> FAISS focuses on numerical efficiency, while `.pkl` ensures easy metadata retrieval.</li>
    </ul>

    <hr>

    <h2>🌟 Advantages</h2>
    <ul>
        <li><strong>Efficiency:</strong> FAISS enables fast nearest-neighbor search, even for large datasets.</li>
        <li><strong>Modularity:</strong> Easy to switch between models for retrieval and evaluation.</li>
        <li><strong>Scalability:</strong> Suitable for large datasets with high-dimensional embeddings.</li>
    </ul>

    <hr>

    <h2>📌 Scope for Enhancements</h2>
    <ul>
        <li><strong>Additional Models:</strong> Incorporate newer architectures like Vision Transformers (ViT).</li>
        <li><strong>Active Learning:</strong> Allow user feedback to refine retrieval results.</li>
        <li><strong>Web Interface:</strong> Replace the Tkinter GUI with a web-based solution (e.g., Flask or Django).</li>
        <li><strong>Real-Time Retrieval:</strong> Optimize inference for video or real-time applications.</li>
    </ul>

    <hr>

    <h2>🤝 Contribution</h2>
    <p>
        Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.
    </p>

    <hr>

    <h2>📜 License</h2>
    <p>
        This project is licensed under the MIT License. See <code>LICENSE</code> for details.
    </p>
</body>
</html>

