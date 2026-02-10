// DOM Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const previewImage = document.getElementById('preview-image');
const removeBtn = document.getElementById('remove-btn');
const detectBtn = document.getElementById('detect-btn');
const uploadSection = document.getElementById('upload-section');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');
const resultImage = document.getElementById('result-image');
const statsContainer = document.getElementById('stats-container');
const detectionsDetails = document.getElementById('detections-details');
const downloadBtn = document.getElementById('download-btn');
const uploadAnotherBtn = document.getElementById('upload-another-btn');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toast-message');

let selectedFile = null;
let detectionResult = null;

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
removeBtn.addEventListener('click', resetUpload);
detectBtn.addEventListener('click', performDetection);
downloadBtn.addEventListener('click', downloadResult);
uploadAnotherBtn.addEventListener('click', resetAll);

// Drag and Drop Handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// File Handling
function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showToast('Invalid file type. Please upload JPG, PNG, or WebP image.', 'error');
        return;
    }
    
    // Validate file size (10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast('File too large. Maximum size is 10MB.', 'error');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        previewSection.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
    
    showToast('Image loaded successfully!', 'success');
}

// Reset Upload
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewSection.classList.add('hidden');
    previewImage.src = '';
}

// Reset All
function resetAll() {
    resetUpload();
    resultsSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    detectionResult = null;
}

// Perform Detection
async function performDetection() {
    if (!selectedFile) {
        showToast('Please select an image first.', 'error');
        return;
    }
    
    // Show loading, hide upload
    uploadSection.classList.add('hidden');
    loadingSection.classList.remove('hidden');
    
    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
        // Call API
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Detection failed');
        }
        
        // Store result
        detectionResult = data;
        
        // Display results
        displayResults(data);
        
        // Hide loading, show results
        loadingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        
        showToast('Detection completed successfully!', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        loadingSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        showToast(error.message || 'An error occurred during detection.', 'error');
    }
}

// Display Results
function displayResults(data) {
    // Display annotated image
    resultImage.src = data.annotated_image;
    
    // Display statistics
    displayStatistics(data.statistics);
    
    // Display detection details
    displayDetectionDetails(data.detections);
}

// Display Statistics
function displayStatistics(stats) {
    statsContainer.innerHTML = '';
    
    // Create stat cards for each category
    const categories = [
        { key: 'vehicle', label: 'Vehicles', icon: 'fa-car' },
        { key: 'pedestrian', label: 'Pedestrians', icon: 'fa-walking' },
        { key: 'animal', label: 'Animals', icon: 'fa-paw' },
        { key: 'traffic', label: 'Traffic Signs', icon: 'fa-traffic-light' },
        { key: 'other', label: 'Others', icon: 'fa-cube' }
    ];
    
    categories.forEach(category => {
        const count = stats.by_category[category.key] || 0;
        
        const card = document.createElement('div');
        card.className = `stat-card ${category.key}`;
        card.innerHTML = `
            <div class="stat-number">${count}</div>
            <div class="stat-label">
                <i class="fas ${category.icon}"></i> ${category.label}
            </div>
        `;
        
        statsContainer.appendChild(card);
    });
}

// Display Detection Details
function displayDetectionDetails(detections) {
    detectionsDetails.innerHTML = '<h3 style="margin-bottom: 15px; color: var(--text-light);">Detected Objects</h3>';
    
    if (detections.length === 0) {
        detectionsDetails.innerHTML += '<p style="color: var(--text-muted);">No objects detected.</p>';
        return;
    }
    
    detections.forEach((det, index) => {
        const item = document.createElement('div');
        item.className = 'detection-item';
        
        const confidence = (det.score * 100).toFixed(1);
        
        item.innerHTML = `
            <div>
                <span class="detection-label">${index + 1}. ${det.label}</span>
                <span style="color: var(--text-muted); font-size: 0.9rem; margin-left: 10px;">
                    (${det.category})
                </span>
            </div>
            <div class="detection-confidence">${confidence}%</div>
        `;
        
        detectionsDetails.appendChild(item);
    });
}

// Download Result
function downloadResult() {
    if (!detectionResult || !detectionResult.annotated_image) {
        showToast('No result to download.', 'error');
        return;
    }
    
    // Create a download link
    const link = document.createElement('a');
    link.href = detectionResult.annotated_image;
    link.download = `detection_result_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showToast('Image downloaded successfully!', 'success');
}

// Toast Notification
function showToast(message, type = 'info') {
    toastMessage.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.remove('hidden');
    
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

// Initialize
console.log('TEDR Object Detection System - Ready!');
