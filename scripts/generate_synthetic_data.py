"""
Generate improved synthetic FER-2013-like dataset for testing purposes.
Creates more distinguishable patterns for each emotion.
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Emotion labels for FER-2013
EMOTIONS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

def generate_synthetic_image(emotion_label, size=48, seed=None):
    """Generate a synthetic face-like image with emotion-specific patterns."""
    if seed is not None:
        np.random.seed(seed)
    
    # Base face pattern (simplified oval shape)
    y, x = np.ogrid[:size, :size]
    center = size // 2
    
    # Create face oval mask
    face_mask = ((x - center) ** 2 / (size/2.2) ** 2 + 
                 (y - center) ** 2 / (size/2.8) ** 2 <= 1).astype(float)
    
    # Base grayscale values - skin tone
    image = np.ones((size, size), dtype=np.float32) * 140
    image = image * (1 - face_mask) + (180 + np.random.normal(0, 5, (size, size))) * face_mask
    
    # Add hair region at top
    hair_mask = (y < size * 0.25) & (np.abs(x - center) < size * 0.4)
    image[hair_mask] = 50 + np.random.normal(0, 10, np.sum(hair_mask))
    
    # Eye positions
    eye_y = int(size * 0.4)
    eye_offset = int(size * 0.15)
    
    # Emotion-specific features
    if emotion_label == 3:  # happy - smile, raised cheeks
        # Upturned mouth (smile)
        for i in range(-10, 11):
            mouth_y = int(size * 0.65) + int(abs(i) * 0.3)
            if 0 <= mouth_y < size:
                image[mouth_y, center + i] = 100 + np.random.normal(0, 5)
        # Raised cheeks
        cheek_mask = ((y > size * 0.5) & (y < size * 0.65) & 
                      (np.abs(x - center) > size * 0.1) & (np.abs(x - center) < size * 0.35))
        image[cheek_mask] = np.clip(image[cheek_mask] + 15, 0, 255)
        # Eyes slightly closed (happy)
        image[eye_y-2:eye_y+3, center-eye_offset-6:center-eye_offset+6] = 80
        image[eye_y-2:eye_y+3, center+eye_offset-6:center+eye_offset+6] = 80
        
    elif emotion_label == 4:  # sad - downturned mouth, droopy eyes
        # Downturned mouth (frown)
        for i in range(-10, 11):
            mouth_y = int(size * 0.68) + int(abs(i) * 0.4)
            if 0 <= mouth_y < size:
                image[mouth_y, center + i] = 90 + np.random.normal(0, 5)
        # Droopy eyebrows
        image[eye_y-6:eye_y-2, center-eye_offset-8:center-eye_offset+8] = 70
        image[eye_y-6:eye_y-2, center+eye_offset-8:center+eye_offset+8] = 70
        # Eyes
        image[eye_y-2:eye_y+4, center-eye_offset-6:center-eye_offset+6] = 75
        image[eye_y-2:eye_y+4, center+eye_offset-6:center+eye_offset+6] = 75
        
    elif emotion_label == 0:  # angry - furrowed brows, tight lips
        # Furrowed brows (V-shape)
        for i in range(-8, 9):
            brow_y = eye_y - 8 + int(abs(i) * 0.5)
            if 0 <= brow_y < size:
                image[brow_y:brow_y+3, center + i] = 60
        # Tight straight mouth
        mouth_y = int(size * 0.68)
        image[mouth_y-2:mouth_y+3, center-12:center+12] = 95
        # Eyes narrowed
        image[eye_y-1:eye_y+4, center-eye_offset-7:center-eye_offset+7] = 70
        image[eye_y-1:eye_y+4, center+eye_offset-7:center+eye_offset+7] = 70
        
    elif emotion_label == 5:  # surprise - wide eyes, open mouth
        # Wide open eyes (larger)
        image[eye_y-5:eye_y+6, center-eye_offset-8:center-eye_offset+8] = 65
        image[eye_y-5:eye_y+6, center+eye_offset-8:center+eye_offset+8] = 65
        # Raised eyebrows
        image[eye_y-12:eye_y-7, center-eye_offset-10:center-eye_offset+10] = 75
        image[eye_y-12:eye_y-7, center+eye_offset-10:center+eye_offset+10] = 75
        # Open mouth (O-shape)
        mouth_y = int(size * 0.68)
        mouth_mask = ((y > mouth_y - 8) & (y < mouth_y + 8) & 
                      (np.abs(x - center) < 8))
        image[mouth_mask] = 60
        
    elif emotion_label == 2:  # fear - similar to surprise but different
        # Very wide eyes
        image[eye_y-6:eye_y+7, center-eye_offset-9:center-eye_offset+9] = 60
        image[eye_y-6:eye_y+7, center+eye_offset-9:center+eye_offset+9] = 60
        # Raised eyebrows (higher than surprise)
        image[eye_y-14:eye_y-9, center-eye_offset-10:center-eye_offset+10] = 70
        image[eye_y-14:eye_y-9, center+eye_offset-10:center+eye_offset+10] = 70
        # Stretched mouth (horizontal)
        mouth_y = int(size * 0.68)
        image[mouth_y-3:mouth_y+3, center-14:center+14] = 85
        
    elif emotion_label == 1:  # disgust - scrunched nose, raised upper lip
        # Scrunched nose
        nose_y = int(size * 0.52)
        image[nose_y-4:nose_y+4, center-6:center+6] = 90
        # Raised upper lip
        mouth_y = int(size * 0.62)
        image[mouth_y-4:mouth_y+2, center-10:center+10] = 85
        # Wrinkled nose area
        wrinkle_mask = ((y > nose_y - 6) & (y < nose_y + 2) & 
                        (np.abs(x - center) < 10))
        image[wrinkle_mask] = np.clip(image[wrinkle_mask] + 20, 0, 255)
        # Eyes squinted
        image[eye_y:eye_y+5, center-eye_offset-6:center-eye_offset+6] = 75
        image[eye_y:eye_y+5, center+eye_offset-6:center+eye_offset+6] = 75
        
    else:  # neutral - no special features
        # Normal eyes
        image[eye_y-3:eye_y+4, center-eye_offset-6:center-eye_offset+6] = 75
        image[eye_y-3:eye_y+4, center+eye_offset-6:center+eye_offset+6] = 75
        # Normal eyebrows
        image[eye_y-8:eye_y-4, center-eye_offset-8:center-eye_offset+8] = 70
        image[eye_y-8:eye_y-4, center+eye_offset-8:center+eye_offset+8] = 70
        # Neutral mouth (straight line)
        mouth_y = int(size * 0.68)
        image[mouth_y-2:mouth_y+2, center-10:center+10] = 100
    
    # Add nose
    nose_y = int(size * 0.52)
    nose_mask = ((y > nose_y - 6) & (y < nose_y + 6) & 
                 (np.abs(x - center) < 4))
    image[nose_mask] = 160
    
    # Add some noise for realism
    noise = np.random.normal(0, 8, (size, size))
    image = image + noise
    
    # Clip to valid range and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Flatten to string format (space-separated pixels)
    return ' '.join(map(str, image.flatten()))


def generate_dataset(n_samples_per_emotion=500, split='train', base_seed=42):
    """Generate a synthetic dataset."""
    data = []
    
    for emotion, label in EMOTIONS.items():
        for i in range(n_samples_per_emotion):
            seed = base_seed + emotion * 10000 + i
            pixels = generate_synthetic_image(emotion, seed=seed)
            data.append({
                'pixels': pixels,
                'emotion': emotion,
                'Usage': split
            })
    
    df = pd.DataFrame(data)
    return df


def main():
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("Generating improved synthetic FER-2013-like dataset...")
    print("(For production use, download actual FER-2013 from Kaggle)")
    
    # Generate training set
    print("Generating training samples...")
    train_df = generate_dataset(n_samples_per_emotion=500, split='Training', base_seed=42)
    train_df.to_csv(data_dir / "train.csv", index=False)
    print(f"  Created train.csv with {len(train_df)} samples")
    
    # Generate test set
    print("Generating test samples...")
    test_df = generate_dataset(n_samples_per_emotion=100, split='PublicTest', base_seed=99999)
    test_df.to_csv(data_dir / "test.csv", index=False)
    print(f"  Created test.csv with {len(test_df)} samples")
    
    # Create test_with_emotions
    test_df.to_csv(data_dir / "test_with_emotions.csv", index=False)
    print(f"  Created test_with_emotions.csv")
    
    print("\n✓ Improved synthetic dataset generation complete!")
    print("\nNote: This is a synthetic dataset for testing.")
    print("For actual FER-2013 dataset, download from:")
    print("https://www.kaggle.com/datasets/msambare/fer2013")


if __name__ == "__main__":
    main()
