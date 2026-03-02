"""
Download and prepare the FER-2013 dataset.
"""
import os
import pandas as pd
import urllib.request
import zipfile
from pathlib import Path

# Dataset URLs (using Kaggle or alternative sources)
DATASET_URLS = [
    "https://raw.githubusercontent.com/shamangary/FER-2013/master/fer2013.csv",
    "https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013",
]

DATA_DIR = Path(__file__).parent.parent / "data"


def download_from_kaggle():
    """Download dataset from Kaggle (requires authentication)."""
    print("Attempting to download from Kaggle...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("msambare/fer2013", path=str(DATA_DIR), unzip=True)
        return True
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return False


def download_direct():
    """Download dataset directly from alternative sources."""
    print("Attempting direct download...")
    
    # Try GitHub source first
    github_url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/fer2013.csv"
    target_path = DATA_DIR / "fer2013.csv"
    
    try:
        urllib.request.urlretrieve(github_url, target_path)
        print(f"Downloaded from GitHub to {target_path}")
        return True
    except Exception as e:
        print(f"GitHub download failed: {e}")
    
    # Try alternative source
    alt_url = "https://storage.googleapis.com/kagglesdsdata/datasets/1536/2946/fer2013.csv"
    try:
        urllib.request.urlretrieve(alt_url, target_path)
        print(f"Downloaded from alternative source to {target_path}")
        return True
    except Exception as e:
        print(f"Alternative download failed: {e}")
    
    return False


def prepare_dataset():
    """Prepare the dataset by splitting into train/test files."""
    fer_file = DATA_DIR / "fer2013.csv"
    
    if not fer_file.exists():
        print("FER-2013 dataset not found. Please download manually from:")
        print("https://www.kaggle.com/datasets/msambare/fer2013")
        print(f"And place it in: {DATA_DIR}")
        return False
    
    print("Loading FER-2013 dataset...")
    df = pd.read_csv(fer_file)
    
    # Check columns
    print(f"Dataset columns: {df.columns.tolist()}")
    print(f"Dataset shape: {df.shape}")
    
    # The FER-2013 dataset has 'Usage' column indicating train/test
    if 'Usage' in df.columns:
        # Split based on Usage column
        train_df = df[df['Usage'] == 'Training'].copy()
        test_df = df[df['Usage'] == 'PublicTest'].copy()
        
        # Also include PrivateTest in test if available
        private_test = df[df['Usage'] == 'PrivateTest'].copy()
        if len(private_test) > 0:
            test_df = pd.concat([test_df, private_test], ignore_index=True)
        
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Save split datasets
        train_df.to_csv(DATA_DIR / "train.csv", index=False)
        test_df.to_csv(DATA_DIR / "test.csv", index=False)
        
        print(f"Saved train.csv with {len(train_df)} samples")
        print(f"Saved test.csv with {len(test_df)} samples")
        
    else:
        # If no Usage column, do a random split
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        train_df.to_csv(DATA_DIR / "train.csv", index=False)
        test_df.to_csv(DATA_DIR / "test.csv", index=False)
        
        print(f"Saved train.csv with {len(train_df)} samples")
        print(f"Saved test.csv with {len(test_df)} samples")
    
    return True


def create_test_with_emotions():
    """Create test file with emotion labels for validation."""
    test_file = DATA_DIR / "test.csv"
    
    if not test_file.exists():
        print("test.csv not found!")
        return
    
    df = pd.read_csv(test_file)
    
    # Ensure we have the emotion column
    if 'emotion' in df.columns or 'Emotion' in df.columns:
        emotion_col = 'emotion' if 'emotion' in df.columns else 'Emotion'
        # Create test_with_emotions.csv
        df.to_csv(DATA_DIR / "test_with_emotions.csv", index=False)
        print("Created test_with_emotions.csv")
    else:
        print("No emotion column found in test.csv")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    
    print("=" * 50)
    print("FER-2013 Dataset Downloader")
    print("=" * 50)
    
    # Try different download methods
    success = False
    
    # Method 1: Direct download
    if download_direct():
        success = True
    
    # Method 2: Kaggle (if available)
    if not success:
        success = download_from_kaggle()
    
    if success:
        # Prepare the dataset
        if prepare_dataset():
            create_test_with_emotions()
            print("\n✓ Dataset preparation complete!")
        else:
            print("\n✗ Dataset preparation failed!")
    else:
        print("\n" + "=" * 50)
        print("MANUAL DOWNLOAD REQUIRED")
        print("=" * 50)
        print("\nPlease download the FER-2013 dataset from:")
        print("https://www.kaggle.com/datasets/msambare/fer2013")
        print("\nThen:")
        print("1. Extract the zip file")
        print("2. Copy fer2013.csv to the data/ folder")
        print("3. Run this script again")
