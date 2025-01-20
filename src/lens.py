import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import faiss
import pickle
import torch
from torchvision import models, transforms

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

# Load FAISS index and image paths
index = faiss.read_index("database_index.faiss")
with open("image_paths.pkl", "rb") as f:
    database_image_paths = pickle.load(f)

# Move FAISS index to GPU
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

# Global variables for drawing on the canvas
drawing = False
ix, iy = -1, -1
rect = None
rect_id = None

def extract_features(image):
    """Extract normalized features from an image using ResNet-34."""
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image_tensor)
    return feature.cpu().numpy().squeeze() / np.linalg.norm(feature.cpu().numpy().squeeze())

def start_drawing(event):
    """Start drawing the rectangle."""
    global drawing, ix, iy, rect_id
    drawing = True
    ix, iy = event.x, event.y
    rect_id = canvas.create_rectangle(ix, iy, ix, iy, outline="green", width=2)

def draw(event):
    """Draw the rectangle live."""
    global rect_id
    if drawing:
        canvas.coords(rect_id, ix, iy, event.x, event.y)

def stop_drawing(event):
    """Stop drawing and finalize the rectangle."""
    global drawing, rect
    if drawing:
        drawing = False
        rect = (ix, iy, event.x, event.y)

def upload_image():
    """Allow the user to upload an image."""
    global img, img_display, canvas, rect, roi_image

    filepath = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[
            ("All Files", "*.*"),
            ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")
        ]
    )

    if filepath:
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_display = img.copy()

        display_image(img_display, canvas_width, canvas_height)
        rect = None
        roi_image = None

def display_image(image, canvas_width, canvas_height):
    """Display the image on the canvas resized to fit the screen."""
    global canvas, img_tk
    h, w, _ = image.shape
    scale = min(canvas_width / w, canvas_height / h)
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    img_tk = ImageTk.PhotoImage(Image.fromarray(resized_image))
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

def find_similar():
    """Find and display the top 5 best-matching images based on the selected ROI."""
    global rect, roi_image, img

    if rect is None:
        messagebox.showerror("Error", "No ROI defined. Draw a rectangle first.")
        return

    # Extract ROI
    x1, y1, x2, y2 = rect
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    roi_image = img[y1:y2, x1:x2]
    if roi_image.size == 0:
        messagebox.showerror("Error", "Invalid ROI. Draw a valid rectangle.")
        return

    # Resize the ROI to the expected input size
    roi_image_resized = cv2.resize(roi_image, (128, 128))  # Resize to 128x128
    roi_image_pil = Image.fromarray(roi_image_resized)

    # Extract the feature vector
    roi_feature = extract_features(roi_image_pil)

    # Use FAISS to find the nearest neighbors
    distances, indices = gpu_index.search(np.expand_dims(roi_feature, axis=0), k=5)

    # Display results
    display_results(roi_image, [database_image_paths[i] for i in indices[0]])

def display_results(query_image, image_paths):
    """Display the retrieved results vertically with labels."""
    results_window = tk.Toplevel(root)
    results_window.title("Search Results")
    results_window.geometry("600x900")

    ttk.Label(results_window, text="Query Image", font=("Helvetica", 16, "bold")).pack(pady=10)

    # Display the query image
    query_image_resized = cv2.resize(query_image, (256, 256))  # Resize to fit
    query_image_tk = ImageTk.PhotoImage(Image.fromarray(query_image_resized))
    query_label = ttk.Label(results_window, image=query_image_tk)
    query_label.image = query_image_tk
    query_label.pack(pady=10)

    ttk.Label(results_window, text="Top 5 Matches", font=("Helvetica", 16, "bold")).pack(pady=10)

    # Display the top 5 matches vertically
    for rank, path in enumerate(image_paths, start=1):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (256, 256))  # Resize for better viewing
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_resized))
        
        # Display the image and label
        frame = ttk.Frame(results_window)
        frame.pack(pady=5)
        lbl_rank = ttk.Label(frame, text=f"Rank {rank}", font=("Helvetica", 14))
        lbl_rank.grid(row=0, column=0, padx=10)
        lbl_image = ttk.Label(frame, image=img_tk)
        lbl_image.image = img_tk
        lbl_image.grid(row=0, column=1, padx=10)

# GUI Setup
root = tk.Tk()
root.title("Image Retrieval System")

# Dynamically determine screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set dynamic canvas dimensions based on screen size
canvas_width = int(screen_width * 0.8)
canvas_height = int(screen_height * 0.7)
root.geometry(f"{screen_width}x{screen_height}")
root.resizable(False, False)

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=6)
style.configure("TLabel", font=("Helvetica", 14))

# Canvas for displaying image
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="lightgray", relief=tk.SUNKEN)
canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

# Bind mouse events for drawing
canvas.bind("<ButtonPress-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_drawing)

# Buttons
button_frame = ttk.Frame(root)
button_frame.grid(row=1, column=0, columnspan=3, pady=10)

btn_upload = ttk.Button(button_frame, text="Upload Image", command=upload_image)
btn_upload.grid(row=0, column=0, padx=10)

btn_find = ttk.Button(button_frame, text="Find Similar", command=find_similar)
btn_find.grid(row=0, column=1, padx=10)

btn_close = ttk.Button(button_frame, text="Close", command=root.destroy)
btn_close.grid(row=0, column=2, padx=10)

root.mainloop()

