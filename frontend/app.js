// TEDR Frontend JavaScript

const API_URL = window.location.origin;

// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const newImageBtn = document.getElementById('newImageBtn');
const resultCanvas = document.getElementById('resultCanvas');
const objectCount = document.getElementById('objectCount');
const processingTime = document.getElementById('processingTime');
const imageSize = document.getElementById('imageSize');
const detectionsList = document.getElementById('detectionsList');

let currentImage = null;

// Event Listeners
browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
newImageBtn.addEventListener('click', resetUI);

// Drag and Drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('drag-over');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('drag-over');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

uploadBox.addEventListener('click', () => fileInput.click());

// File Handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (JPEG or PNG)');
        return;
    }

    // Read and process file
    const reader = new FileReader();
    reader.onload = (e) => {
        currentImage = e.target.result;
        uploadImage(file);
    };
    reader.readAsDataURL(file);
}

// API Communication
async function uploadImage(file) {
    // Show loading overlay
    loadingOverlay.style.display = 'flex';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/detect`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        alert(`Error processing image: ${error.message}`);
        loadingOverlay.style.display = 'none';
    }
}

// Display Results
function displayResults(data) {
    // Hide loading and upload section
    loadingOverlay.style.display = 'none';
    document.querySelector('.upload-section').style.display = 'none';
    resultsSection.style.display = 'block';

    // Update stats
    objectCount.textContent = data.num_detections || 0;
    processingTime.textContent = `${data.processing_time}s`;
    imageSize.textContent = `${data.image_size[0]}x${data.image_size[1]}`;

    // Draw image with bounding boxes
    drawDetections(currentImage, data.detections, data.image_size);

    // Display detection list
    displayDetectionsList(data.detections);
}

// Draw Detections on Canvas
function drawDetections(imageSrc, detections, imageSize) {
    const img = new Image();
    img.onload = () => {
        const ctx = resultCanvas.getContext('2d');
        
        // Set canvas size to match image
        resultCanvas.width = img.width;
        resultCanvas.height = img.height;
        
        // Draw image
        ctx.drawImage(img, 0, 0);
        
        // Draw bounding boxes
        detections.forEach((detection, index) => {
            const [x1, y1, x2, y2] = detection.bbox;
            const width = x2 - x1;
            const height = y2 - y1;
            
            // Scale coordinates to canvas size
            const scaleX = img.width / imageSize[0];
            const scaleY = img.height / imageSize[1];
            
            const scaledX = x1 * scaleX;
            const scaledY = y1 * scaleY;
            const scaledWidth = width * scaleX;
            const scaledHeight = height * scaleY;
            
            // Generate color based on index
            const colors = [
                '#3b82f6', '#ef4444', '#10b981', '#f59e0b', 
                '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'
            ];
            const color = colors[index % colors.length];
            
            // Draw bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
            
            // Draw label background
            const label = `${detection.label} ${(detection.confidence * 100).toFixed(1)}%`;
            ctx.font = 'bold 16px Arial';
            const textMetrics = ctx.measureText(label);
            const textHeight = 20;
            
            ctx.fillStyle = color;
            ctx.fillRect(scaledX, scaledY - textHeight - 4, textMetrics.width + 10, textHeight + 4);
            
            // Draw label text
            ctx.fillStyle = 'white';
            ctx.fillText(label, scaledX + 5, scaledY - 8);
        });
    };
    img.src = imageSrc;
}

// Display Detections List
function displayDetectionsList(detections) {
    detectionsList.innerHTML = '';
    
    if (detections.length === 0) {
        detectionsList.innerHTML = '<p style="text-align: center; color: #6b7280; padding: 20px;">No objects detected with confidence above threshold.</p>';
        return;
    }
    
    detections.forEach((detection, index) => {
        const item = document.createElement('div');
        item.className = 'detection-item';
        
        const [x1, y1, x2, y2] = detection.bbox;
        const bbox = `[${Math.round(x1)}, ${Math.round(y1)}, ${Math.round(x2)}, ${Math.round(y2)}]`;
        
        item.innerHTML = `
            <div class="detection-label">${index + 1}. ${detection.label}</div>
            <div class="detection-bbox">BBox: ${bbox}</div>
            <div class="detection-confidence">${(detection.confidence * 100).toFixed(1)}%</div>
        `;
        
        detectionsList.appendChild(item);
    });
}

// Reset UI
function resetUI() {
    resultsSection.style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block';
    fileInput.value = '';
    currentImage = null;
    detectionsList.innerHTML = '';
}

// Check API health on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API is not accessible:', error);
    }
}

// Initialize
checkHealth();
