# Action Recognition using CNN + LSTM

Deep learning model for human action recognition fine-tuned on the UCF11 dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-REST%20API-green.svg)

## Overview

This project implements a CNN + LSTM architecture for video-based action recognition. The model uses MobileNetV2 for spatial feature extraction and LSTM for temporal sequence learning, fine-tuned on UCF11 dataset.

### Supported Actions (UCF11)

| Basketball | Biking | Diving | Golf Swing | Horse Riding | Soccer Juggling |
|------------|--------|--------|------------|--------------|-----------------|
| Swing | Tennis Swing | Trampoline Jumping | Volleyball Spiking | Walking | |

## Model Architecture

```
Input (20 frames × 224 × 224 × 3)
         ↓
TimeDistributed(MobileNetV2) - Spatial Features
         ↓
LSTM (64 units) - Temporal Learning
         ↓
Dropout (0.5)
         ↓
Dense (11, softmax) - Classification
```

## Results

### Training Curves
![Accuracy and Loss](plots/accuracy_loss.png)

### Confusion Matrix
![Confusion Matrix](plots/confusion_matrix.png)

## Installation

```bash
git clone https://github.com/usman11267/Action-Recognition-CNN-LSTM.git
cd Action-Recognition-CNN-LSTM

cd backend
pip install -r requirements.txt
```

## Usage

### Start Server

```bash
cd backend
python app.py
```

Server runs at `http://127.0.0.1:5000`

### Open Frontend

Open `frontend/index.html` in browser

### API

```bash
POST /predict -F "video=@sample.mp4"
```

```json
{
  "success": true,
  "prediction": {
    "action": "basketball",
    "confidence": 92.45
  }
}
```

## Project Structure

```
├── backend/
│   ├── app.py
│   ├── model_loader.py
│   ├── video_utils.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── models/
│   ├── ucf11_cnn_lstm_model.h5
│   └── classes.json
├── samples/              # Test videos
└── plots/
```

## Sample Test Videos

Download sample videos from UCF11 dataset to test:
- [UCF11 Dataset](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php)

Or use any short video (MP4/AVI) showing these actions:
- Basketball, Biking, Diving, Golf, Horse Riding
- Soccer Juggling, Swing, Tennis, Trampoline, Volleyball, Walking

## Tech Stack

- TensorFlow/Keras
- MobileNetV2 (ImageNet pretrained)
- Flask REST API
- ImageIO, OpenCV

## Dataset

[UCF11 - YouTube Action Dataset](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php)

## License

MIT License
