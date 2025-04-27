# app/main.py

import io
import base64
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2  # OpenCV needed for grad-cam utils

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Grad-CAM imports
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import your model definition and loading function
# Ensure model_definition.py contains the EXACT class from your notebook
from model_definition import AttentionTransformer, load_model

# --- Configuration ---
MODEL_PATH = "G:\AI_ML project workshop\liver fibrosis app\model_files\liver_fibrosis_model_efficientnet_attention.pth" # Path to your model weights
NUM_CLASSES = 5
CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4'] # Class names from your dataset folders
IMAGE_SIZE = 224 # Image size your model expects

# Define the exact preprocessing steps from your training notebook
# This must match how your training data was processed!
preprocess_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")), # Ensure image is RGB
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats used in notebook
])

# --- Load Model ---
try:
    # Ensure load_model uses the correct AttentionTransformer definition
    model, device = load_model(MODEL_PATH, NUM_CLASSES)
    print(f"Model loaded successfully on device: {device}")
    # Identify the target layer for Grad-CAM
    # This should be a convolutional layer, typically the last one in the feature extractor
    target_layer = model.efficientnet.features[-1]
    print(f"Grad-CAM target layer identified: {type(target_layer)}")
except Exception as e:
    print(f"FATAL: Error loading model from {MODEL_PATH}: {e}")
    # Depending on deployment, you might want the app to exit or raise a clearer startup error
    raise RuntimeError(f"Could not load the model: {e}")

# --- Initialize FastAPI App ---
app = FastAPI(title="Liver Fibrosis Classification API")

# Add CORS middleware to allow requests from your frontend during development
# IMPORTANT: Restrict origins in a production environment for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for testing
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# --- Define Prediction Endpoint ---
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image file, predicts the liver fibrosis stage,
    generates a Grad-CAM++ visualization highlighting important regions,
    and returns the predicted class, confidence score, and the Grad-CAM image.
    """
    contents = await file.read()

    # 1. Preprocess Image for Model and Grad-CAM Base
    try:
        # Load image using PIL, ensuring it's RGB
        image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
        # Resize for the model and keep this version for Grad-CAM overlay
        base_image_pil_resized = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE))

        # Apply full preprocessing pipeline for model input tensor
        input_tensor = preprocess_transform(base_image_pil_resized).unsqueeze(0) # Add batch dimension
        input_tensor = input_tensor.to(device) # Move tensor to the correct device (CPU or GPU)

        # Prepare the base image for Grad-CAM overlay (needs 0-1 float, numpy)
        rgb_img_numpy = np.array(base_image_pil_resized) / 255.0
        rgb_img_float = np.float32(rgb_img_numpy)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file or error during preprocessing: {e}")

    # 2. Make Prediction
    try:
        with torch.no_grad(): # Ensure gradients are not computed for inference
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1) # Convert logits to probabilities
            confidence, predicted_idx_tensor = torch.max(probabilities, 1) # Get highest probability and its index
            predicted_idx = predicted_idx_tensor.item() # Get index as an integer
            predicted_class = CLASS_NAMES[predicted_idx] # Map index to class name
            confidence_score = confidence.item() # Get confidence as a float
    except Exception as e:
         print(f"Error during model inference: {e}")
         raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # 3. Generate Grad-CAM++ Visualization
    gradcam_data_url = None # Initialize to None
    try:
        # Define the target for Grad-CAM based on the predicted class
        targets = [ClassifierOutputTarget(predicted_idx)]

        # Initialize Grad-CAM++ algorithm
        # Using 'with' ensures resources are handled correctly
        cam_algorithm = GradCAMPlusPlus
        with cam_algorithm(model=model, target_layers=[target_layer]) as cam:
            # Generate the CAM heatmap
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                # aug_smooth=True, # Optional: Test if smoothing helps
                                # eigen_smooth=True # Optional: Test if smoothing helps
                                )[0, :] # Get CAM for the first (and only) image in the batch

            # Overlay the heatmap onto the base image
            cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
            # cam_image is now a numpy array with RGB values in the range [0, 1]

        # Convert the Grad-CAM image (NumPy 0-1) to a Base64 encoded Data URL for the frontend
        cam_image_uint8 = np.uint8(255 * cam_image) # Convert to 0-255 integer range
        cam_pil = Image.fromarray(cam_image_uint8) # Convert back to PIL image
        buffered = io.BytesIO()
        cam_pil.save(buffered, format="PNG") # Save as PNG into memory buffer
        gradcam_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8') # Encode as base64 string
        gradcam_data_url = f"data:image/png;base64,{gradcam_base64}" # Format as data URL

    except Exception as e:
        # Log the error but don't stop the request; prediction result is still valuable
        print(f"Warning: Grad-CAM++ generation failed for {file.filename}: {e}")
        gradcam_data_url = None # Ensure it's None if generation fails

    # 4. Return Result Package
    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence_score * 100, 2), # Return confidence as percentage
        "gradcam_image_url": gradcam_data_url # Include the Grad-CAM data URL (or None if failed)
    }

# --- Root Endpoint (Optional but good for testing API availability) ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Liver Fibrosis Classification API with Grad-CAM!"}

# --- How to Run Locally (Comment for reference) ---
# Navigate to the directory containing the 'app' folder in your terminal
# Run: uvicorn app.main:app --reload
# Access the API docs at http://127.0.0.1:8000/docs