"""
Data preprocessing utilities for FER-2013 emotion detection.
Handles loading, parsing, and transforming image data.
"""
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Tuple, Optional, List


# Emotion labels mapping (FER-2013 standard)
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Image dimensions
IMG_WIDTH = 48
IMG_HEIGHT = 48


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load FER-2013 dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with columns: pixels, emotion, Usage
    """
    df = pd.read_csv(filepath)
    return df


def parse_pixels(pixels_str: str) -> np.ndarray:
    """
    Parse space-separated pixel string into numpy array.
    
    Args:
        pixels_str: Space-separated string of pixel values
        
    Returns:
        1D numpy array of pixel values
    """
    return np.array(list(map(int, pixels_str.split())))


def pixels_to_image(pixels: np.ndarray, shape: Tuple[int, int] = (48, 48)) -> np.ndarray:
    """
    Reshape pixel array to image dimensions.
    
    Args:
        pixels: 1D array of pixel values
        shape: Target image shape (height, width)
        
    Returns:
        2D numpy array representing grayscale image
    """
    return pixels.reshape(shape).astype(np.uint8)


def load_and_preprocess_data(
    filepath: str,
    normalize: bool = True,
    expand_dims: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess dataset for training.
    
    Args:
        filepath: Path to CSV file
        normalize: Whether to normalize pixel values to [0, 1]
        expand_dims: Whether to add channel dimension for CNN
        
    Returns:
        Tuple of (images, labels)
    """
    df = load_data(filepath)
    
    # Parse pixels
    X = np.array([parse_pixels(p) for p in df['pixels'].values])
    
    # Reshape to (n_samples, 48, 48)
    X = X.reshape(-1, IMG_HEIGHT, IMG_WIDTH)
    
    # Add channel dimension if needed (for CNN input)
    if expand_dims:
        X = X.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    
    # Normalize to [0, 1]
    if normalize:
        X = X.astype('float32') / 255.0
    
    # Get labels
    y = df['emotion'].values
    
    return X, y


def one_hot_encode(labels: np.ndarray, num_classes: int = 7) -> np.ndarray:
    """
    One-hot encode integer labels.
    
    Args:
        labels: Array of integer labels
        num_classes: Number of classes
        
    Returns:
        One-hot encoded labels
    """
    return np.eye(num_classes)[labels]


def preprocess_frame_for_prediction(
    frame: np.ndarray,
    face_rect: Optional[Tuple[int, int, int, int]] = None
) -> Optional[np.ndarray]:
    """
    Preprocess a video frame for emotion prediction.
    
    Args:
        frame: BGR frame from video/webcam
        face_rect: Optional (x, y, w, h) face rectangle. If None, attempts to detect face.
        
    Returns:
        Preprocessed image (48, 48, 1) normalized to [0, 1], or None if no face detected
    """
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Detect face if not provided
    if face_rect is None:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None
        # Use the largest face
        face_rect = max(faces, key=lambda f: f[2] * f[3])
    
    x, y, w, h = face_rect
    
    # Add padding to include more context
    padding = int(max(w, h) * 0.1)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(gray.shape[1], x + w + padding)
    y2 = min(gray.shape[0], y + h + padding)
    
    # Crop face region
    face_crop = gray[y1:y2, x1:x2]
    
    # Resize to 48x48
    face_resized = cv2.resize(face_crop, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1] and add channel dimension
    face_normalized = face_resized.astype('float32') / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=-1)
    
    return face_expanded


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frames_per_second: int = 1
) -> List[str]:
    """
    Extract frames from video at specified rate.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        frames_per_second: Number of frames to extract per second
        
    Returns:
        List of paths to saved frame images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps / frames_per_second)
    
    saved_paths = []
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Detect and crop face
            processed = preprocess_frame_for_prediction(frame)
            
            if processed is not None:
                # Convert to displayable format and save
                face_img = (processed[:, :, 0] * 255).astype(np.uint8)
                save_path = output_path / f"image{saved_count}.png"
                cv2.imwrite(str(save_path), face_img)
                saved_paths.append(str(save_path))
                saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    return saved_paths


def get_emotion_label(prediction: np.ndarray) -> Tuple[str, float]:
    """
    Get emotion label and confidence from model prediction.
    
    Args:
        prediction: Model output probabilities (7,) or logits
        
    Returns:
        Tuple of (emotion_label, confidence)
    """
    # Ensure probabilities
    if prediction.min() < 0:
        prediction = np.exp(prediction) / np.sum(np.exp(prediction))
    
    emotion_idx = np.argmax(prediction)
    confidence = prediction[emotion_idx]
    
    return EMOTION_LABELS[emotion_idx], confidence


if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing preprocessing utilities...")
    
    # Test loading data
    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "train.csv"
    
    if train_path.exists():
        X, y = load_and_preprocess_data(str(train_path))
        print(f"Loaded training data: X shape = {X.shape}, y shape = {y.shape}")
        print(f"Label distribution: {np.bincount(y)}")
    else:
        print("Training data not found!")
