# Facial Emotion Detection System

A real-time facial emotion detection system using Convolutional Neural Networks (CNNs) and OpenCV.

## Overview

This project implements a system that detects emotions from facial expressions in real-time using a webcam video stream. The system can classify seven distinct emotions:

- **Happy** 😊
- **Sad** 😢
- **Angry** 😠
- **Surprise** 😲
- **Fear** 😨
- **Disgust** 🤢
- **Neutral** 😐

## Project Structure

```
Emotions-detector/
├── data/
│   ├── train.csv          # Training data (FER-2013)
│   ├── test.csv           # Test data
│   └── ...
├── results/
│   ├── model/
│   │   ├── final_emotion_model.keras      # Trained model
│   │   ├── final_emotion_model_arch.txt   # Model architecture
│   │   ├── learning_curves.png            # Training curves
│   │   └── tensorboard.png                # TensorBoard screenshot
│   └── preprocessing_test/
│       ├── input_video.mp4
│       └── image*.png
├── scripts/
│   ├── train.py             # Model training
│   ├── predict.py           # Test set evaluation
│   ├── predict_live_stream.py  # Real-time prediction
│   ├── preprocess.py        # Data preprocessing utilities
│   └── validation_loss_accuracy.py  # Validation metrics
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- pip
- Webcam (for live stream prediction)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/steodhiambo/Emotions-detector.git
cd Emotions-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
python scripts/download_data.py
```

## Usage

### Training the Model

```bash
python scripts/train.py
```

This will:
- Load and preprocess the FER-2013 dataset
- Train a CNN with early stopping and TensorBoard monitoring
- Save the model and architecture files
- Generate learning curve plots

### Evaluate on Test Set

```bash
python scripts/predict.py
```

Expected output:
```
Accuracy on test set: 62%
```

### Real-time Emotion Detection

```bash
python scripts/predict_live_stream.py
```

This will:
- Access your webcam
- Detect faces in real-time
- Classify emotions every second
- Display results in the terminal

### Using a Pre-recorded Video

```bash
python scripts/predict_live_stream.py --video path/to/video.mp4
```

## Model Architecture

The CNN architecture consists of:
- Multiple convolutional blocks with batch normalization
- Max pooling layers for downsampling
- Dropout for regularization
- Fully connected layers for classification

See `results/model/final_emotion_model_arch.txt` for detailed architecture.

## Dataset

This project uses the **FER-2013** dataset from the ICML 2013 Challenges on Representation Learning. The dataset contains:
- 28,709 training samples
- 3,589 test samples
- 48x48 grayscale face images
- 7 emotion labels

## Technical Details

### Face Detection
- Uses OpenCV's Haar Cascade classifier
- Pre-trained model for frontal face detection

### Preprocessing Pipeline
1. Capture video frame
2. Detect face using Haar Cascade
3. Crop and resize to 48x48
4. Convert to grayscale
5. Normalize pixel values

### Training Configuration
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch Size: 64
- Early Stopping: Monitor validation loss
- TensorBoard: Real-time training monitoring

## Performance

- **Target Accuracy**: >60% on test set
- **Achieved Accuracy**: ~62%
- **Inference Time**: ~100ms per frame

## TensorBoard

To view training metrics:
```bash
tensorboard --logdir=logs/
```

Then open http://localhost:6006 in your browser.

## License

This project is for educational purposes.

## Acknowledgments

- FER-2013 dataset: Goodfellow et al., ICML 2013
- OpenCV for computer vision utilities
- TensorFlow/Keras for deep learning
