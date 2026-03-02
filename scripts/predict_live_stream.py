"""
Real-time emotion detection from webcam or video stream.
Uses OpenCV for face detection and the trained CNN for emotion classification.
"""
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import preprocess_frame_for_prediction, get_emotion_label, EMOTION_LABELS

# Paths
MODEL_DIR = Path(__file__).parent.parent / "results" / "model"
MODEL_PATH = MODEL_DIR / "final_emotion_model.keras"
PREPROCESSING_TEST_DIR = Path(__file__).parent.parent / "results" / "preprocessing_test"


def load_model():
    """Load the trained emotion detection model."""
    from tensorflow import keras
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Please run 'python scripts/train.py' first to train the model."
        )
    
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(str(MODEL_PATH))
    return model


def load_face_cascade():
    """Load OpenCV's pre-trained face detection cascade."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade


def detect_faces(frame, face_cascade, min_size=(30, 30)):
    """
    Detect faces in a frame.
    
    Args:
        frame: BGR frame from video/webcam
        face_cascade: Haar cascade classifier
        min_size: Minimum face size
        
    Returns:
        List of face rectangles (x, y, w, h)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces


def predict_emotion(model, frame, face_rect):
    """
    Predict emotion from a face region.
    
    Args:
        model: Trained emotion detection model
        frame: Video frame
        face_rect: Face rectangle (x, y, w, h)
        
    Returns:
        Tuple of (emotion_label, confidence) or (None, None) if no face
    """
    x, y, w, h = face_rect
    
    # Preprocess the face region
    face_crop = frame[y:y+h, x:x+w]
    processed = preprocess_frame_for_prediction(face_crop, face_rect=None)
    
    if processed is None:
        return None, None
    
    # Add batch dimension
    processed = np.expand_dims(processed, axis=0)
    
    # Predict
    prediction = model.predict(processed, verbose=0)[0]
    
    # Get emotion label and confidence
    emotion, confidence = get_emotion_label(prediction)
    
    return emotion, confidence


def draw_results(frame, faces, predictions, fps=None):
    """
    Draw face rectangles and emotion labels on frame.
    
    Args:
        frame: Video frame
        faces: List of face rectangles
        predictions: List of (emotion, confidence) tuples
        fps: Optional FPS value to display
    """
    for i, (face_rect, (emotion, confidence)) in enumerate(zip(faces, predictions)):
        if emotion is None:
            continue
            
        x, y, w, h = face_rect
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Create label background
        label = f"{emotion}: {confidence*100:.1f}%"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x, y - label_h - baseline - 5),
            (x + label_w, y),
            (0, 255, 0),
            cv2.FILLED
        )
        
        # Draw label text
        cv2.putText(
            frame, label, (x, y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
    
    # Draw FPS
    if fps is not None:
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    
    return frame


def process_video_stream(
    model,
    face_cascade,
    video_source=0,
    output_dir=None,
    predictions_per_second=1
):
    """
    Process video stream and predict emotions.
    
    Args:
        model: Trained emotion detection model
        face_cascade: Face detection cascade
        video_source: Camera index or video path
        output_dir: Directory to save preprocessed frames (for testing)
        predictions_per_second: Number of predictions to make per second
    """
    # Open video source
    if isinstance(video_source, int):
        print(f"Opening webcam (device {video_source})...")
        cap = cv2.VideoCapture(video_source)
    else:
        print(f"Reading video from {video_source}...")
        cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {video_source}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default for webcams
    
    frame_interval = int(fps / predictions_per_second)
    
    print("\nReading video stream...")
    print(f"Video FPS: {fps:.1f}")
    print(f"Predictions per second: {predictions_per_second}")
    print(f"Frame interval: {frame_interval}")
    print("\nPress 'q' to quit\n")
    
    # Initialize variables
    frame_count = 0
    saved_frames = 0
    last_prediction_time = None
    
    # Create output directory for preprocessing test
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Clear existing files
        for f in output_path.glob("*.png"):
            f.unlink()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            frame_count += 1
            
            # Detect faces
            faces = detect_faces(frame, face_cascade)
            predictions = []
            
            # Process each face
            for i, face_rect in enumerate(faces):
                x, y, w, h = face_rect
                
                # Make prediction at specified interval
                if frame_count % frame_interval == 0:
                    emotion, confidence = predict_emotion(model, frame, face_rect)
                    
                    if emotion is not None:
                        predictions.append((emotion, confidence))
                        
                        # Print prediction to console
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"Preprocessing ...")
                        print(f"{timestamp}s : {emotion} , {confidence*100:.0f}%")
                        
                        # Save preprocessed frame for testing
                        if output_dir:
                            face_crop = frame[y:y+h, x:x+w]
                            processed = preprocess_frame_for_prediction(face_crop)
                            if processed is not None:
                                save_path = output_path / f"image{saved_frames}.png"
                                save_img = (processed[:, :, 0] * 255).astype(np.uint8)
                                cv2.imwrite(str(save_path), save_img)
                                saved_frames += 1
                    else:
                        predictions.append((None, None))
                else:
                    predictions.append((None, None))
            
            # Draw results on frame
            frame = draw_results(frame, faces, predictions)
            
            # Show frame
            cv2.imshow('Emotion Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"\nProcessed {frame_count} frames")
    if output_dir:
        print(f"Saved {saved_frames} preprocessed frames to {output_dir}")
    
    return saved_frames


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Real-time facial emotion detection from webcam or video.'
    )
    parser.add_argument(
        '--video', '-v',
        type=str,
        default=None,
        help='Path to video file (default: use webcam)'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device index (default: 0)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=str(PREPROCESSING_TEST_DIR),
        help='Output directory for preprocessed frames'
    )
    parser.add_argument(
        '--predictions-per-second', '-pps',
        type=int,
        default=1,
        help='Number of predictions per second (default: 1)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FACIAL EMOTION DETECTION - LIVE STREAM")
    print("=" * 60)
    
    try:
        # Load model
        model = load_model()
        
        # Load face detector
        print("Loading face detector...")
        face_cascade = load_face_cascade()
        
        # Determine video source
        video_source = args.camera if args.video is None else args.video
        
        # Process stream
        process_video_stream(
            model=model,
            face_cascade=face_cascade,
            video_source=video_source,
            output_dir=args.output,
            predictions_per_second=args.predictions_per_second
        )
        
        print("\n✓ Emotion detection completed!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
