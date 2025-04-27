# 🩺 End-to-End Liver Fibrosis Staging from CT Images with Grad-CAM++ Visualization

## 📖 Project Overview

This project delivers a complete pipeline for staging liver fibrosis (F0–F4) directly from CT images. Using a deep learning model built with PyTorch, it predicts the fibrosis stage and highlights important regions in the scan through Grad-CAM++ visualization.

It also features a **user-friendly web application** where users can upload CT images and instantly view:
- The predicted fibrosis stage
- Model confidence score
- A heatmap showing where the model focused its attention

The backend runs on **FastAPI**, and the frontend is built with **HTML/CSS/JavaScript**.



## ✨ Key Features

- **CT Image Upload:** Supports `.png`, `.jpg`, `.jpeg` formats.
- **Fibrosis Stage Prediction:** Classifies images into F0, F1, F2, F3, or F4.
- **Confidence Score:** Displays how confident the model is about its prediction.
- **Explainable AI:** Grad-CAM++ heatmaps highlight important regions influencing the decision.
- **Web-Based Interface:** Clean, intuitive design using vanilla JS, HTML, and CSS.
- **FastAPI Backend:** Efficient and scalable API service.

---

## 📸 Demo / Screenshots

- **Upload Page:**
- ![UI](https://github.com/user-attachments/assets/1ce6d98e-6452-4935-935d-61ec434324e1)
  

- **Result Page (Prediction + Grad-CAM++ Visualization):**  
  _![result](https://github.com/user-attachments/assets/a2562d2e-869c-4d79-b093-748bbe52ad21)
_



## 🛠 Tech Stack

- **Machine Learning:**
  - Python 3.x
  - PyTorch & Torchvision
  - NumPy
  - OpenCV
  - Pillow
  - Matplotlib (for visualizations)
  - Scikit-learn (for model evaluation)
  - pytorch-grad-cam (Grad-CAM++ visualization)

- **Backend:**
  - FastAPI
  - Uvicorn
  - Python-Multipart (file uploads)

- **Frontend:**
  - HTML5
  - CSS3
  - Vanilla JavaScript (Fetch API)

- **(Optional) Deployment:**
  - Docker
  - Platforms like Heroku, Google Cloud Run, AWS Elastic Beanstalk

---

## 🗂 Dataset

- **Source:** Kaggle — Liver Fibrosis CT Database  
 

- **Description:**  
  CT scans labeled into five fibrosis stages: **F0** to **F4**.

- **Preprocessing:**
  - Converted to RGB
  - Resized to 224×224 pixels
  - Normalized with ImageNet standards
  - (Optionally mention any data augmentation if used)

---

## 🧠 Model Architecture

- **Backbone:** Pre-trained EfficientNet-B0 (ImageNet weights)
- **Attention Mechanism:** Multi-Head Attention applied to extracted features
- **Classifier:** Fully connected layer mapping features to five classes (F0–F4)

- **Explainability:**  
  Grad-CAM++ is applied to the last convolutional block of EfficientNet to visualize important areas.

- **Performance Highlights:**  
  _(Replace with your actual results)_
  - Accuracy: ~97%
  - Precision, Recall, F1-Score (Weighted Avg): ~0.97
  - _(Optional: Insert confusion matrix image)_

---

## 🏗 Installation & Setup

Follow these steps to get started locally:

### 1. Clone the Repository

```bash
git clone [Your Repository URL]
cd [Repository Name]
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
# Activate:
# Windows (Powershell)
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Download and Place Model Weights

- Save `liver_fibrosis_model_efficientnet_attention.pth` inside a `model_files/` folder.
- Ensure the path matches what's expected in `app/main.py`.

---

## 🚀 Usage

### 1. Start the Backend Server

```bash
python -m uvicorn app.main:app --reload
```

- API will be live at: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

### 2. Launch the Frontend

- Open `index.html` directly in your web browser.

### 3. Interact

- Upload a CT image
- View the predicted fibrosis stage, confidence, and Grad-CAM++ heatmap

---

## 📡 API Details

**Endpoint:** `/predict/`  
**Method:** `POST`  
**Payload:** `multipart/form-data` with an image file

**Example Successful Response:**

```json
{
  "filename": "example_ct.png",
  "predicted_class": "F2",
  "confidence": 88.45,
  "gradcam_image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA..."
}
```

**Error Responses:**
- `400 Bad Request`: Invalid image
- `500 Internal Server Error`: Server-side failure

---



## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Kaggle dataset contributors
- Open-source libraries: PyTorch, FastAPI, Uvicorn, pytorch-grad-cam, OpenCV, Pillow, NumPy
- Research inspirations from works on medical imaging and explainable AI




