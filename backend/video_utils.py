"""
Video Processing Utilities for Action Recognition
Handles frame extraction and preprocessing for CNN + LSTM model
"""

import numpy as np
import cv2
import imageio.v2 as imageio

# Model configuration constants
SEQ_LENGTH = 20      # Number of frames to extract
IMG_SIZE = 224       # Frame dimensions (224x224)
CHANNELS = 3         # RGB channels


def extract_frames(video_path, seq_length=SEQ_LENGTH, img_size=IMG_SIZE):
    """
    Extract frames from a video file
    
    This function:
    1. Opens the video using imageio (supports multiple formats)
    2. Extracts up to seq_length frames
    3. Resizes each frame to img_size x img_size
    4. Normalizes pixel values to [0, 1]
    5. Pads with zeros if video has fewer frames
    
    Args:
        video_path (str): Path to the video file
        seq_length (int): Number of frames to extract (default: 20)
        img_size (int): Size to resize frames (default: 224)
    
    Returns:
        numpy.ndarray: Array of shape (seq_length, img_size, img_size, 3)
                      or None if extraction fails
    """
    frames = []
    
    try:
        # Try to open video with imageio (ffmpeg backend)
        reader = imageio.get_reader(video_path, format='ffmpeg')
        
        # Get video metadata
        try:
            meta = reader.get_meta_data()
            total_frames = meta.get('nframes', float('inf'))
            fps = meta.get('fps', 30)
            duration = meta.get('duration', 0)
            print(f"Video info: {total_frames} frames, {fps} fps, {duration:.2f}s duration")
        except:
            pass
        
        # Extract frames
        for i, frame in enumerate(reader):
            if i >= seq_length:
                break
            
            # Resize frame
            frame_resized = cv2.resize(frame, (img_size, img_size))
            
            # Normalize to [0, 1]
            frame_normalized = frame_resized / 255.0
            
            frames.append(frame_normalized)
        
        reader.close()
        
    except Exception as e:
        print(f"Error extracting frames with imageio: {e}")
        
        # Fallback to OpenCV
        try:
            frames = extract_frames_opencv(video_path, seq_length, img_size)
            if frames is None or len(frames) == 0:
                return None
        except Exception as e2:
            print(f"Error with OpenCV fallback: {e2}")
            return None
    
    if len(frames) == 0:
        print("No frames extracted from video")
        return None
    
    # Pad with zeros if we have fewer frames than required
    while len(frames) < seq_length:
        frames.append(np.zeros((img_size, img_size, CHANNELS)))
    
    print(f"Extracted {len(frames)} frames successfully")
    return np.array(frames, dtype=np.float32)


def extract_frames_opencv(video_path, seq_length=SEQ_LENGTH, img_size=IMG_SIZE):
    """
    Fallback frame extraction using OpenCV
    
    Args:
        video_path (str): Path to the video file
        seq_length (int): Number of frames to extract
        img_size (int): Size to resize frames
    
    Returns:
        list: List of normalized frame arrays
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None
    
    frame_count = 0
    
    while frame_count < seq_length:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
        
        # Normalize
        frame_normalized = frame_resized / 255.0
        
        frames.append(frame_normalized)
        frame_count += 1
    
    cap.release()
    return frames


def preprocess_video(frames):
    """
    Preprocess extracted frames for model input
    
    The model expects input shape: (batch_size, seq_length, img_size, img_size, 3)
    This function adds the batch dimension.
    
    Args:
        frames (numpy.ndarray): Array of shape (seq_length, img_size, img_size, 3)
    
    Returns:
        numpy.ndarray: Array of shape (1, seq_length, img_size, img_size, 3)
    """
    if frames is None:
        raise ValueError("Frames cannot be None")
    
    # Ensure correct shape
    if len(frames.shape) == 4:
        # Add batch dimension
        processed = np.expand_dims(frames, axis=0)
    elif len(frames.shape) == 5:
        # Already has batch dimension
        processed = frames
    else:
        raise ValueError(f"Invalid frames shape: {frames.shape}")
    
    # Ensure float32 type
    processed = processed.astype(np.float32)
    
    print(f"Preprocessed video shape: {processed.shape}")
    return processed


def validate_video(video_path):
    """
    Validate that a video file can be processed
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        dict: Validation result with status and info
    """
    import os
    
    result = {
        'valid': False,
        'path': video_path,
        'exists': False,
        'readable': False,
        'has_frames': False,
        'error': None
    }
    
    # Check if file exists
    if not os.path.exists(video_path):
        result['error'] = 'File does not exist'
        return result
    result['exists'] = True
    
    # Try to read video
    try:
        reader = imageio.get_reader(video_path, format='ffmpeg')
        result['readable'] = True
        
        # Check for frames
        frame_count = 0
        for frame in reader:
            frame_count += 1
            if frame_count >= 1:
                break
        reader.close()
        
        if frame_count > 0:
            result['has_frames'] = True
            result['valid'] = True
        else:
            result['error'] = 'No frames found in video'
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


def get_video_info(video_path):
    """
    Get detailed information about a video file
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        dict: Video information
    """
    info = {
        'path': video_path,
        'fps': None,
        'duration': None,
        'total_frames': None,
        'resolution': None
    }
    
    try:
        reader = imageio.get_reader(video_path, format='ffmpeg')
        meta = reader.get_meta_data()
        
        info['fps'] = meta.get('fps')
        info['duration'] = meta.get('duration')
        info['total_frames'] = meta.get('nframes')
        info['resolution'] = meta.get('size')
        
        reader.close()
    except Exception as e:
        info['error'] = str(e)
    
    return info


if __name__ == '__main__':
    # Test video processing
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"Testing video processing for: {video_path}")
        
        # Validate
        validation = validate_video(video_path)
        print(f"Validation: {validation}")
        
        if validation['valid']:
            # Extract frames
            frames = extract_frames(video_path)
            if frames is not None:
                print(f"Frames shape: {frames.shape}")
                
                # Preprocess
                processed = preprocess_video(frames)
                print(f"Processed shape: {processed.shape}")
    else:
        print("Usage: python video_utils.py <video_path>")
        print("\nConfiguration:")
        print(f"  Sequence length: {SEQ_LENGTH} frames")
        print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
        print(f"  Channels: {CHANNELS} (RGB)")
