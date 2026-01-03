/**
 * Action Recognition Frontend JavaScript
 * Handles video upload and prediction API calls
 */

// Configuration
const API_BASE_URL = 'http://127.0.0.1:5000';
const API_PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;
const API_CLASSES_ENDPOINT = `${API_BASE_URL}/classes`;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const videoInput = document.getElementById('videoInput');
const browseBtn = document.getElementById('browseBtn');
const filePreview = document.getElementById('filePreview');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFile = document.getElementById('removeFile');
const videoPreview = document.getElementById('videoPreview');
const predictBtn = document.getElementById('predictBtn');
const resultsSection = document.getElementById('resultsSection');
const predictedAction = document.getElementById('predictedAction');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const predictionsList = document.getElementById('predictionsList');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const tryAgainBtn = document.getElementById('tryAgainBtn');

// State
let selectedFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    fetchClasses();
});

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Browse button click
    browseBtn.addEventListener('click', () => videoInput.click());

    // File input change
    videoInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);

    // Remove file
    removeFile.addEventListener('click', clearSelection);

    // Predict button
    predictBtn.addEventListener('click', predictAction);

    // Try again button
    tryAgainBtn.addEventListener('click', resetUI);
}

/**
 * Handle file selection from input
 */
function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

/**
 * Handle drag over
 */
function handleDragOver(event) {
    event.preventDefault();
    dropZone.classList.add('drag-over');
}

/**
 * Handle drag leave
 */
function handleDragLeave(event) {
    event.preventDefault();
    dropZone.classList.remove('drag-over');
}

/**
 * Handle file drop
 */
function handleDrop(event) {
    event.preventDefault();
    dropZone.classList.remove('drag-over');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

/**
 * Process selected file
 */
function processFile(file) {
    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mpeg', 
                        'video/mpg', 'video/quicktime', 'video/x-msvideo',
                        'video/webm', 'video/x-matroska'];
    
    if (!file.type.startsWith('video/') && !validTypes.includes(file.type)) {
        showError('Please select a valid video file (MP4, AVI, MOV, MPG, etc.)');
        return;
    }

    // Check file size (max 100MB)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size exceeds 100MB limit. Please select a smaller video.');
        return;
    }

    selectedFile = file;

    // Update UI
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    // Create video preview URL
    const videoURL = URL.createObjectURL(file);
    videoPreview.src = videoURL;

    // Show preview and enable predict button
    dropZone.style.display = 'none';
    filePreview.style.display = 'block';
    predictBtn.disabled = false;
    
    // Hide results and errors
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
}

/**
 * Clear file selection
 */
function clearSelection() {
    selectedFile = null;
    videoInput.value = '';
    videoPreview.src = '';
    
    dropZone.style.display = 'block';
    filePreview.style.display = 'none';
    predictBtn.disabled = true;
}

/**
 * Make prediction API call
 */
async function predictAction() {
    if (!selectedFile) {
        showError('Please select a video file first.');
        return;
    }

    // Show loading state
    setLoading(true);
    hideResults();
    hideError();

    try {
        const formData = new FormData();
        formData.append('video', selectedFile);

        const response = await fetch(API_PREDICT_ENDPOINT, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            showResults(data);
        } else {
            showError(data.message || 'Prediction failed. Please try again.');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showError('Failed to connect to the API. Make sure the backend server is running on http://127.0.0.1:5000');
    } finally {
        setLoading(false);
    }
}

/**
 * Display prediction results
 */
function showResults(data) {
    const prediction = data.prediction;
    const allPredictions = data.all_predictions;

    // Main prediction
    predictedAction.textContent = formatClassName(prediction.action);
    confidenceValue.textContent = `${prediction.confidence}%`;
    confidenceFill.style.width = `${prediction.confidence}%`;
    
    // Set confidence bar color based on value
    if (prediction.confidence >= 80) {
        confidenceFill.className = 'confidence-fill high';
    } else if (prediction.confidence >= 50) {
        confidenceFill.className = 'confidence-fill medium';
    } else {
        confidenceFill.className = 'confidence-fill low';
    }

    // All predictions list
    predictionsList.innerHTML = '';
    for (const [className, probability] of Object.entries(allPredictions)) {
        const percentage = (probability * 100).toFixed(2);
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <span class="class-name">${formatClassName(className)}</span>
            <div class="prediction-bar-container">
                <div class="prediction-bar" style="width: ${percentage}%"></div>
            </div>
            <span class="class-prob">${percentage}%</span>
        `;
        predictionsList.appendChild(item);
    }

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Show error message
 */
function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    resultsSection.style.display = 'none';
}

/**
 * Hide error section
 */
function hideError() {
    errorSection.style.display = 'none';
}

/**
 * Hide results section
 */
function hideResults() {
    resultsSection.style.display = 'none';
}

/**
 * Set loading state
 */
function setLoading(loading) {
    predictBtn.disabled = loading;
    
    const btnText = predictBtn.querySelector('.btn-text');
    const btnLoading = predictBtn.querySelector('.btn-loading');
    
    if (loading) {
        btnText.style.display = 'none';
        btnLoading.style.display = 'inline-flex';
    } else {
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }
}

/**
 * Reset UI to initial state
 */
function resetUI() {
    clearSelection();
    hideResults();
    hideError();
}

/**
 * Format file size to human readable
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format class name for display
 */
function formatClassName(name) {
    return name
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

/**
 * Fetch available classes from API
 */
async function fetchClasses() {
    try {
        const response = await fetch(API_CLASSES_ENDPOINT);
        const data = await response.json();
        
        if (data.classes) {
            updateClassesList(data.classes);
        }
    } catch (error) {
        console.log('Could not fetch classes from API. Using default list.');
    }
}

/**
 * Update classes list in UI
 */
function updateClassesList(classes) {
    const classesList = document.getElementById('classesList');
    classesList.innerHTML = '';
    
    classes.forEach(className => {
        const li = document.createElement('li');
        li.textContent = formatClassName(className);
        classesList.appendChild(li);
    });
}
