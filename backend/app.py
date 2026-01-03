"""
Flask REST API for Action Recognition using CNN + LSTM Model
University Assignment - UCF11 Dataset
"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from model_loader import load_model, get_class_labels
from video_utils import extract_frames, preprocess_video

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mpg', 'mpeg', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and class labels at startup
print("Loading CNN + LSTM model...")
model = load_model()
class_labels = get_class_labels()
print(f"Model loaded successfully! Classes: {class_labels}")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Action Recognition API is active',
        'model': 'CNN + LSTM (UCF11)',
        'classes': class_labels,
        'endpoints': {
            '/predict': 'POST - Upload video for action prediction',
            '/classes': 'GET - Get list of action classes'
        }
    })


@app.route('/classes', methods=['GET'])
def get_classes():
    """Return available action classes"""
    return jsonify({
        'classes': class_labels,
        'count': len(class_labels)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict action from uploaded video
    
    Expects:
        - POST request with video file in 'video' field
    
    Returns:
        - JSON with predicted action and confidence scores
    """
    try:
        # Check if video file is present in request
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided',
                'message': 'Please upload a video file with key "video"'
            }), 400
        
        video_file = request.files['video']
        
        # Check if file is selected
        if video_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'message': 'Please select a video file to upload'
            }), 400
        
        # Check file extension
        if not allowed_file(video_file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type',
                'message': f'Allowed formats: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)
        
        try:
            # Extract and preprocess frames
            frames = extract_frames(filepath)
            
            if frames is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to process video',
                    'message': 'Could not extract frames from video. Please try another file.'
                }), 400
            
            # Preprocess for model input
            processed_video = preprocess_video(frames)
            
            # Make prediction
            predictions = model.predict(processed_video, verbose=0)
            
            # Get predicted class
            predicted_class_idx = int(predictions[0].argmax())
            confidence = float(predictions[0][predicted_class_idx])
            predicted_action = class_labels[predicted_class_idx]
            
            # Get all class probabilities
            all_predictions = {
                class_labels[i]: float(predictions[0][i]) 
                for i in range(len(class_labels))
            }
            
            # Sort by confidence
            sorted_predictions = dict(
                sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
            )
            
            return jsonify({
                'success': True,
                'prediction': {
                    'action': predicted_action,
                    'confidence': round(confidence * 100, 2),
                    'class_index': predicted_class_idx
                },
                'all_predictions': sorted_predictions,
                'message': f'Predicted action: {predicted_action} ({confidence*100:.2f}% confidence)'
            })
            
        finally:
            # Clean up - remove uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large',
        'message': 'Maximum file size is 100MB'
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Action Recognition API - CNN + LSTM")
    print("="*50)
    print(f"Model: UCF11 CNN+LSTM")
    print(f"Classes: {len(class_labels)}")
    print(f"Endpoint: http://127.0.0.1:5000/predict")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
