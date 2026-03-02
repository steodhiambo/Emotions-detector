"""
Training script for facial emotion detection CNN.
Includes TensorBoard integration, early stopping, and model saving.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import create_emotion_cnn, compile_model, create_callbacks, print_model_summary, NUM_CLASSES
from preprocess import load_and_preprocess_data, one_hot_encode

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODEL_DIR = RESULTS_DIR / "model"
LOGS_DIR = Path(__file__).parent.parent / "logs"

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2


def load_training_data():
    """Load and prepare training data."""
    train_path = DATA_DIR / "train.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")
    
    print("Loading training data...")
    X_train, y_train = load_and_preprocess_data(
        str(train_path),
        normalize=True,
        expand_dims=True
    )
    
    # One-hot encode labels
    y_train_encoded = one_hot_encode(y_train, num_classes=NUM_CLASSES)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train_encoded.shape}")
    print(f"Label distribution: {np.bincount(y_train)}")
    
    return X_train, y_train_encoded


def plot_learning_curves(history, save_path: str):
    """
    Plot and save learning curves.
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Mark the best epoch (early stopping point)
    val_loss = history.history['val_loss']
    best_epoch = np.argmin(val_loss)
    axes[1].axvline(x=best_epoch, color='r', linestyle='--', 
                    label=f'Best Epoch: {best_epoch + 1}', linewidth=2)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved to {save_path}")


def train_model():
    """Main training function."""
    print("=" * 60)
    print("FACIAL EMOTION DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, y_train = load_training_data()
    
    # Create model
    print("\nCreating CNN model...")
    model = create_emotion_cnn(
        input_shape=(48, 48, 1),
        num_classes=NUM_CLASSES,
        dropout_rate=0.4,
        l2_reg=0.001
    )
    
    # Compile model
    compile_model(model, learning_rate=LEARNING_RATE)
    
    # Print model summary
    print("\nModel Architecture:")
    print_model_summary(model, str(MODEL_DIR / "final_emotion_model_arch.txt"))
    
    # Create callbacks
    model_save_path = str(MODEL_DIR / "final_emotion_model.keras")
    callbacks = create_callbacks(
        log_dir=str(LOGS_DIR),
        model_save_path=model_save_path,
        patience=15,
        min_delta=0.001
    )
    
    # Train model
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Epochs: {EPOCHS}")
    print(f"Validation Split: {VALIDATION_SPLIT}")
    print(f"Model will be saved to: {model_save_path}")
    print("=" * 60 + "\n")
    
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # Save final model (not just best)
    final_model_path = str(MODEL_DIR / "final_emotion_model.keras")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Plot and save learning curves
    learning_curves_path = str(MODEL_DIR / "learning_curves.png")
    plot_learning_curves(history, learning_curves_path)
    
    # Print training summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Get best metrics
    best_val_acc = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    best_epoch = np.argmin(history.history['val_loss']) + 1
    
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Total Epochs Trained: {len(history.history['loss'])}")
    print("=" * 60)
    
    # Save TensorBoard info
    tensorboard_info = f"""
TensorBoard Training Log
========================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Best Epoch: {best_epoch}
Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)
Best Validation Loss: {best_val_loss:.4f}

To view TensorBoard:
    tensorboard --logdir={LOGS_DIR}

Then open http://localhost:6006 in your browser.
"""
    with open(MODEL_DIR / "tensorboard_info.txt", 'w') as f:
        f.write(tensorboard_info)
    
    print(tensorboard_info)
    
    return model, history


def capture_tensorboard_screenshot():
    """
    Note: This function provides instructions for capturing TensorBoard screenshot.
    Actual screenshot capture requires browser automation.
    """
    info_path = MODEL_DIR / "tensorboard_screenshot_instructions.txt"
    with open(info_path, 'w') as f:
        f.write("""
TENSORBOARD SCREENSHOT INSTRUCTIONS
====================================

1. Start TensorBoard:
   tensorboard --logdir=logs/

2. Open browser to: http://localhost:6006

3. Navigate to the SCALARS tab

4. Wait for all data to load

5. Take a screenshot showing:
   - Training and Validation Loss curves
   - Training and Validation Accuracy curves

6. Save the screenshot as: tensorboard.png in results/model/

Alternative: Use the learning_curves.png which shows the same information.
""")
    print(f"Screenshot instructions saved to {info_path}")


if __name__ == "__main__":
    try:
        model, history = train_model()
        capture_tensorboard_screenshot()
        print("\n✓ Training completed successfully!")
        print("\nNext steps:")
        print("1. Run: python scripts/predict.py (to evaluate on test set)")
        print("2. Run: python scripts/predict_live_stream.py (for real-time prediction)")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
