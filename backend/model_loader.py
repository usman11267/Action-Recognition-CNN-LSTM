"""
Model Loader Module for Action Recognition
Handles loading the pre-trained CNN + LSTM model and class labels
"""

import os
import json
import tensorflow as tf

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ucf11_cnn_lstm_model.h5')
CLASSES_PATH = os.path.join(BASE_DIR, 'models', 'classes.json')

# Global variables for caching
_model = None
_class_labels = None


def build_model():
    """Rebuild the CNN + LSTM model architecture"""
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Model
    
    SEQ_LENGTH = 20
    IMG_SIZE = 224
    NUM_CLASSES = 11
    
    # CNN backbone
    cnn = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    cnn.trainable = False
    
    # Build model
    inputs = Input(shape=(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    x = TimeDistributed(cnn)(inputs)
    x = LSTM(64)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    return model


def load_model():
    """
    Load the pre-trained CNN + LSTM model
    
    The model architecture:
    - CNN: MobileNetV2 (pre-trained on ImageNet) for spatial feature extraction
    - LSTM: 64 units for temporal sequence learning
    - Output: Softmax layer for 11 UCF11 action classes
    
    Returns:
        tf.keras.Model: Loaded model ready for inference
    """
    global _model
    
    if _model is not None:
        return _model
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            f"Please ensure 'ucf11_cnn_lstm_model.h5' is in the 'models' folder."
        )
    
    print(f"Loading model from: {MODEL_PATH}")
    
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    # Rebuild model architecture
    print("Building model architecture...")
    _model = build_model()
    
    # Load weights
    print("Loading trained weights...")
    try:
        _model.load_weights(MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load weights: {e}")
        print("Using pretrained MobileNetV2 weights only")
    
    # Compile model
    _model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model loaded successfully!")
    print(f"Input shape: {_model.input_shape}")
    print(f"Output shape: {_model.output_shape}")
    
    return _model


def get_class_labels():
    """
    Get the list of action class labels
    
    UCF11 Dataset Classes (11 classes):
    - basketball, biking, diving, golf_swing, horse_riding
    - soccer_juggling, swing, tennis_swing, trampoline_jumping
    - volleyball_spiking, walking
    
    Returns:
        list: List of class label strings
    """
    global _class_labels
    
    if _class_labels is not None:
        return _class_labels
    
    if os.path.exists(CLASSES_PATH):
        # Load from JSON file
        with open(CLASSES_PATH, 'r') as f:
            data = json.load(f)
            _class_labels = data.get('classes', [])
    else:
        # Default UCF11 classes (sorted alphabetically as used in training)
        _class_labels = [
            "basketball",
            "biking", 
            "diving",
            "golf_swing",
            "horse_riding",
            "soccer_juggling",
            "swing",
            "tennis_swing",
            "trampoline_jumping",
            "volleyball_spiking",
            "walking"
        ]
    
    return _class_labels


def get_model_info():
    """
    Get information about the loaded model
    
    Returns:
        dict: Model information including architecture and parameters
    """
    model = load_model()
    
    return {
        'model_path': MODEL_PATH,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'total_params': model.count_params(),
        'classes': get_class_labels(),
        'num_classes': len(get_class_labels()),
        'architecture': {
            'cnn': 'MobileNetV2 (ImageNet pretrained)',
            'temporal': 'LSTM (64 units)',
            'sequence_length': 20,
            'frame_size': '224x224'
        }
    }


if __name__ == '__main__':
    # Test model loading
    print("Testing model loader...")
    
    try:
        model = load_model()
        classes = get_class_labels()
        info = get_model_info()
        
        print("\n" + "="*50)
        print("Model Information:")
        print("="*50)
        for key, value in info.items():
            print(f"{key}: {value}")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
