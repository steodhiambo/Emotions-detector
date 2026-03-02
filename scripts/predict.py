"""
Prediction script for evaluating the emotion detection model on test set.
"""
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import load_and_preprocess_data, one_hot_encode, EMOTION_LABELS
from model import NUM_CLASSES, INPUT_SHAPE

# Paths
MODEL_DIR = Path(__file__).parent.parent / "results" / "model"
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = MODEL_DIR / "final_emotion_model.keras"


def load_model():
    """Load the trained model."""
    from tensorflow import keras
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Please run 'python scripts/train.py' first to train the model."
        )
    
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(str(MODEL_PATH))
    return model


def evaluate_on_test_set(model):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained Keras model
        
    Returns:
        Accuracy on test set
    """
    test_path = DATA_DIR / "test.csv"
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")
    
    print("Loading test data...")
    X_test, y_test = load_and_preprocess_data(
        str(test_path),
        normalize=True,
        expand_dims=True
    )
    
    # One-hot encode labels
    y_test_encoded = one_hot_encode(y_test, num_classes=NUM_CLASSES)
    
    print(f"Test data shape: X={X_test.shape}, y={y_test_encoded.shape}")
    
    # Evaluate
    print("\nEvaluating model on test set...")
    results = model.evaluate(X_test, y_test_encoded, verbose=1)
    
    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test_encoded, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes) * 100
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"\nModel Metrics from evaluate():")
    metric_names = ['Loss'] + [m.name for m in model.metrics]
    for name, value in zip(metric_names, results):
        if name != 'loss':
            print(f"  {name}: {value:.4f}")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 40)
    for class_idx in range(NUM_CLASSES):
        class_mask = true_classes == class_idx
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predicted_classes[class_mask] == true_classes[class_mask]) * 100
            print(f"  {EMOTION_LABELS[class_idx]:10s}: {class_acc:.2f}% ({np.sum(class_mask)} samples)")
    
    # Confusion matrix summary
    print("\nPrediction Summary:")
    print("-" * 40)
    for class_idx in range(NUM_CLASSES):
        predicted_count = np.sum(predicted_classes == class_idx)
        true_count = np.sum(true_classes == class_idx)
        print(f"  {EMOTION_LABELS[class_idx]:10s}: True={true_count:4d}, Predicted={predicted_count:4d}")
    
    print("=" * 60)
    
    return accuracy


def predict_single_image(model, pixels_str):
    """
    Predict emotion from a pixel string.
    
    Args:
        model: Trained Keras model
        pixels_str: Space-separated pixel values
        
    Returns:
        Tuple of (emotion_label, confidence, all_probabilities)
    """
    from preprocess import parse_pixels, pixels_to_image
    
    # Parse and preprocess
    pixels = parse_pixels(pixels_str)
    image = pixels_to_image(pixels)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(np.expand_dims(image, -1), 0)  # Add batch and channel dims
    
    # Predict
    prediction = model.predict(image, verbose=0)[0]
    
    # Get top prediction
    emotion_idx = np.argmax(prediction)
    confidence = prediction[emotion_idx]
    
    return EMOTION_LABELS[emotion_idx], confidence, prediction


def main():
    """Main function."""
    print("=" * 60)
    print("FACIAL EMOTION DETECTION - TEST SET EVALUATION")
    print("=" * 60)
    
    try:
        # Load model
        model = load_model()
        
        # Evaluate on test set
        accuracy = evaluate_on_test_set(model)
        
        # Print final result
        print("\n" + "=" * 60)
        print(f"Accuracy on test set: {accuracy:.0f}%")
        print("=" * 60)
        
        return accuracy
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
