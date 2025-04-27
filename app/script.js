const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const predictButton = document.getElementById('predictButton');
const loadingText = document.getElementById('loading');
const predictionResult = document.getElementById('predictionResult');
const confidenceResult = document.getElementById('confidenceResult');
const gradcamImage = document.getElementById('gradcamImage'); // Get Grad-CAM image element

// Backend API Endpoint URL (update if deployed)
const API_URL = 'http://127.0.0.1:8000/predict/';

let selectedFile = null;

// Hide preview and gradcam images initially
imagePreview.style.display = 'none';
gradcamImage.style.display = 'none';

imageUpload.addEventListener('change', (event) => {
    selectedFile = event.target.files[0];
    if (selectedFile) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        }
        reader.readAsDataURL(selectedFile);
        predictButton.disabled = false;
        predictionResult.textContent = '';
        confidenceResult.textContent = '';
        gradcamImage.style.display = 'none'; // Hide previous gradcam
        gradcamImage.src = '#'; // Reset src
    } else {
        imagePreview.style.display = 'none';
        gradcamImage.style.display = 'none';
        predictButton.disabled = true;
        selectedFile = null;
    }
});

predictButton.addEventListener('click', async () => {
    if (!selectedFile) {
        alert("Please select an image file first.");
        return;
    }

    loadingText.style.display = 'block';
    predictionResult.textContent = '';
    confidenceResult.textContent = '';
    gradcamImage.style.display = 'none'; // Hide during loading
    predictButton.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData,
        });

        loadingText.style.display = 'none'; // Hide loading text once response starts processing

        if (!response.ok) {
            let errorMsg = `HTTP error! status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorMsg += ` - ${errorData.detail || JSON.stringify(errorData)}`;
            } catch (e) { /* Ignore if response isn't JSON */ }
            throw new Error(errorMsg);
        }

        const result = await response.json();

        predictionResult.textContent = `Predicted Stage: ${result.predicted_class}`;
        confidenceResult.textContent = `Confidence: ${result.confidence}%`;

        // Display Grad-CAM image using the data URL from the response
        if (result.gradcam_image_url) {
             gradcamImage.src = result.gradcam_image_url;
             gradcamImage.style.display = 'block';
        } else {
             gradcamImage.alt = 'Grad-CAM generation failed or not available.';
             gradcamImage.style.display = 'block'; // Show alt text
        }


    } catch (error) {
        console.error("Prediction Error:", error);
        predictionResult.textContent = `Error: ${error.message}`;
        confidenceResult.textContent = '';
        gradcamImage.style.display = 'none';
         loadingText.style.display = 'none';
    } finally {
        predictButton.disabled = false;
    }
});