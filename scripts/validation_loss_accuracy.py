"""
Validation script for computing loss and accuracy metrics.
Also runs the preprocessing test as required.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import (
    load_and_preprocess_data, one_hot_encode, 
    extract_frames_from_video, preprocess_frame_for_prediction,
    EMOTION_LABELS
)
from model import NUM_CLASSES

# Paths
MODEL_DIR = Path(__file__).parent.parent / "results" / "model"
DATA_DIR = Path(__file__).parent.parent / "data"
PREPROCESSING_TEST_DIR = Path(__file__).parent.parent / "results" / "preprocessing_test"
MODEL_PATH = MODEL_DIR / "final_emotion_model.keras"


def compute_validation_metrics():
    """Compute validation loss and accuracy on test set."""
    from tensorflow import keras
    
    print("=" * 60)
    print("VALIDATION METRICS COMPUTATION")
    print("=" * 60)
    
    # Load model
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        print("Please train the model first: python scripts/train.py")
        return None, None
    
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(str(MODEL_PATH))
    
    # Load test data
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        print(f"Test data not found at {test_path}")
        return None, None
    
    print("Loading test data...")
    X_test, y_test = load_and_preprocess_data(
        str(test_path),
        normalize=True,
        expand_dims=True
    )
    
    y_test_encoded = one_hot_encode(y_test, num_classes=NUM_CLASSES)
    
    # Evaluate
    print("Evaluating model...")
    results = model.evaluate(X_test, y_test_encoded, verbose=0)
    
    # Get metric names
    metric_names = ['loss'] + [m.name for m in model.metrics]
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    for name, value in zip(metric_names, results):
        if name == 'loss':
            print(f"Validation Loss: {value:.4f}")
        elif name == 'accuracy':
            print(f"Validation Accuracy: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"Validation {name.capitalize()}: {value:.4f}")
    
    print("=" * 60)
    
    return results, model


def run_preprocessing_test(video_path=None, duration_seconds=20):
    """
    Run preprocessing test on video.
    
    Args:
        video_path: Path to video file or 0 for webcam
        duration_seconds: Duration of video to process
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING TEST")
    print("=" * 60)
    
    # Create output directory
    PREPROCESSING_TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear existing files
    for f in PREPROCESSING_TEST_DIR.glob("*.png"):
        f.unlink()
    
    # Load face cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Open video source
    if video_path is None:
        print("No video provided. Creating sample preprocessed frames from test data...")
        # Use test data to create sample preprocessed frames
        test_path = DATA_DIR / "test.csv"
        if test_path.exists():
            import pandas as pd
            df = pd.read_csv(test_path)
            
            # Sample a few images
            sample_indices = np.random.choice(len(df), min(20, len(df)), replace=False)
            
            for i, idx in enumerate(sample_indices):
                row = df.iloc[idx]
                pixels = np.array(list(map(int, row['pixels'].split())))
                image = pixels.reshape(48, 48).astype(np.uint8)
                
                save_path = PREPROCESSING_TEST_DIR / f"image{i}.png"
                cv2.imwrite(str(save_path), image)
            
            print(f"Saved {len(sample_indices)} sample frames to {PREPROCESSING_TEST_DIR}")
            
            # Create a dummy input video info file
            info_path = PREPROCESSING_TEST_DIR / "input_video_info.txt"
            with open(info_path, 'w') as f:
                f.write(f"""PREPROCESSING TEST RESULTS
==========================
Date: {Path.ctime(PREPROCESSING_TEST_DIR)}

Source: Test dataset samples
Number of frames: {len(sample_indices)}
Frame size: 48x48 grayscale

Note: For actual video preprocessing test, run:
    python scripts/validation_loss_accuracy.py --video path/to/video.mp4

Or with webcam:
    python scripts/predict_live_stream.py --output results/preprocessing_test
""")
            
            return len(sample_indices)
        else:
            print("Test data not found!")
            return 0
    
    # Process video file
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps))  # 1 frame per second
    
    saved_count = 0
    frame_idx = 0
    
    print(f"Video FPS: {fps:.1f}, Total frames: {total_frames}")
    print(f"Extracting 1 frame per second...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process at interval
        if frame_idx % frame_interval == 0:
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Use largest face
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face
                
                # Preprocess
                face_crop = frame[y:y+h, x:x+w]
                processed = preprocess_frame_for_prediction(face_crop)
                
                if processed is not None:
                    save_img = (processed[:, :, 0] * 255).astype(np.uint8)
                    save_path = PREPROCESSING_TEST_DIR / f"image{saved_count}.png"
                    cv2.imwrite(str(save_path), save_img)
                    saved_count += 1
        
        frame_idx += 1
        
        # Check duration
        if frame_idx / fps >= duration_seconds:
            break
    
    cap.release()
    
    # Save input video info
    if video_path and Path(video_path).exists():
        import shutil
        input_video_path = PREPROCESSING_TEST_DIR / "input_video.mp4"
        try:
            shutil.copy(video_path, input_video_path)
            print(f"Copied input video to {input_video_path}")
        except Exception as e:
            print(f"Could not copy video: {e}")
    
    print(f"\nSaved {saved_count} preprocessed frames to {PREPROCESSING_TEST_DIR}")
    
    # Save test results
    results_path = PREPROCESSING_TEST_DIR / "preprocessing_test_results.txt"
    with open(results_path, 'w') as f:
        f.write(f"""PREPROCESSING TEST RESULTS
==========================
Input video: {video_path}
Duration processed: {min(duration_seconds, frame_idx/fps):.1f} seconds
Frames extracted: {saved_count}
Output format: 48x48 grayscale PNG images
Output directory: {PREPROCESSING_TEST_DIR}

Preprocessing pipeline:
1. Face detection using Haar Cascade
2. Face cropping with 10% padding
3. Resize to 48x48 using INTER_AREA
4. Normalize to [0, 1]
5. Add channel dimension for CNN input
""")
    
    return saved_count


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validation and preprocessing test')
    parser.add_argument('--video', '-v', type=str, default=None,
                        help='Path to video file for preprocessing test')
    parser.add_argument('--duration', '-d', type=int, default=20,
                        help='Duration in seconds to process (default: 20)')
    parser.add_argument('--metrics-only', '-m', action='store_true',
                        help='Only compute validation metrics')
    
    args = parser.parse_args()
    
    # Compute validation metrics
    results, model = compute_validation_metrics()
    
    if not args.metrics_only:
        # Run preprocessing test
        num_frames = run_preprocessing_test(
            video_path=args.video,
            duration_seconds=args.duration
        )
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if results is not None:
            accuracy_idx = 1  # accuracy is second metric
            if accuracy_idx < len(results):
                print(f"Validation Accuracy: {results[accuracy_idx]*100:.2f}%")
        print(f"Preprocessed frames saved: {num_frames}")
        print(f"Output directory: {PREPROCESSING_TEST_DIR}")
        print("=" * 60)


if __name__ == "__main__":
    main()
