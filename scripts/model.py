"""
CNN Model Architecture for Facial Emotion Detection.
Implements a convolutional neural network for FER-2013 dataset.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from typing import Tuple, Optional


# Model configuration
INPUT_SHAPE = (48, 48, 1)
NUM_CLASSES = 7
L2_REG = 0.001


def create_emotion_cnn(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.4,
    l2_reg: float = L2_REG
) -> keras.Model:
    """
    Create a CNN model for emotion classification.
    
    Architecture inspired by VGG-style networks with:
    - Multiple convolutional blocks with increasing filters
    - Batch normalization for stable training
    - Dropout for regularization
    - Global average pooling before dense layers
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization coefficient
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Block 1: 32 filters
    x = layers.Conv2D(
        32, (3, 3), 
        padding='same', 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        32, (3, 3), 
        padding='same', 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Block 2: 64 filters
    x = layers.Conv2D(
        64, (3, 3), 
        padding='same', 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        64, (3, 3), 
        padding='same', 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Block 3: 128 filters
    x = layers.Conv2D(
        128, (3, 3), 
        padding='same', 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        128, (3, 3), 
        padding='same', 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Block 4: 256 filters
    x = layers.Conv2D(
        256, (3, 3), 
        padding='same', 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        256, (3, 3), 
        padding='same', 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Flatten and Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(
        512, 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate * 1.5)(x)
    
    x = layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name='EmotionCNN')
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    loss: str = 'categorical_crossentropy'
) -> None:
    """
    Compile the model with optimizer and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        loss: Loss function
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
    )


def create_callbacks(
    log_dir: str = 'logs',
    model_save_path: str = 'best_model.keras',
    patience: int = 10,
    min_delta: float = 0.001
) -> list:
    """
    Create training callbacks.
    
    Args:
        log_dir: Directory for TensorBoard logs
        model_save_path: Path to save best model
        patience: Early stopping patience
        min_delta: Minimum change for early stopping
        
    Returns:
        List of callbacks
    """
    import datetime
    
    # TensorBoard callback
    tensorboard_log_dir = f"{log_dir}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    
    # Model checkpoint - save best model
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    
    # Early stopping
    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        min_delta=min_delta,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_delta=min_delta,
        mode='min',
        min_lr=1e-7,
        verbose=1
    )
    
    return [tensorboard_cb, checkpoint_cb, early_stop_cb, reduce_lr_cb]


def print_model_summary(model: keras.Model, filepath: Optional[str] = None) -> str:
    """
    Print and optionally save model summary.
    
    Args:
        model: Keras model
        filepath: Optional path to save summary
        
    Returns:
        Model summary as string
    """
    # Get summary as string
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_str = stream.getvalue()
    
    print(summary_str)
    
    if filepath:
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EMOTION CNN MODEL ARCHITECTURE\n")
            f.write("=" * 70 + "\n\n")
            f.write(summary_str)
            f.write("\n" + "=" * 70 + "\n")
            f.write("ARCHITECTURE EXPLANATION\n")
            f.write("=" * 70 + "\n\n")
            f.write("""
This CNN architecture is designed for facial emotion classification:

1. CONVOLUTIONAL BLOCKS (4 blocks):
   - Each block contains 2 Conv2D layers with 3x3 kernels
   - Batch Normalization after each convolution for stable training
   - MaxPooling (2x2) for spatial downsampling
   - Dropout for regularization to prevent overfitting
   - Filter sizes: 32 -> 64 -> 128 -> 256 (increasing complexity)

2. DENSE LAYERS:
   - Flatten layer to convert 2D features to 1D
   - Two fully connected layers (512 and 256 units)
   - ReLU activation for non-linearity
   - Batch Normalization and Dropout for regularization

3. OUTPUT LAYER:
   - Dense layer with 7 units (one per emotion)
   - Softmax activation for probability distribution

4. REGULARIZATION TECHNIQUES:
   - L2 regularization on kernel weights
   - Dropout (40-60%) to prevent co-adaptation
   - Batch Normalization for internal covariate shift
   - Early Stopping during training

5. TRAINING CONFIGURATION:
   - Optimizer: Adam with learning rate scheduling
   - Loss: Categorical Crossentropy
   - Metrics: Accuracy, Precision, Recall, F1-Score

The architecture balances model capacity with regularization to achieve
good generalization on the FER-2013 dataset while avoiding overfitting.
""")
    
    return summary_str


if __name__ == "__main__":
    # Test model creation
    print("Creating emotion CNN model...")
    model = create_emotion_cnn()
    compile_model(model)
    print_model_summary(model)
    print(f"\nModel created successfully!")
    print(f"Total parameters: {model.count_params():,}")
