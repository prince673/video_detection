import cv2
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from utils.video_processing import extract_frames
from utils.config import DATA_CONFIG

def load_dataset():
    """Load and preprocess FaceForensics dataset"""
    X, y = [], []
    
    # Load original videos
    original_path = os.path.join(DATA_CONFIG['raw_path'], "original_sequences", "crop")
    for video in tqdm(os.listdir(original_path)[:DATA_CONFIG['num_samples']]):
        frames = extract_frames(os.path.join(original_path, video), 
                              DATA_CONFIG['frames_per_video'])
        X.extend(frames)
        y.extend([0] * len(frames))
    
    # Load manipulated videos
    for method in DATA_CONFIG['methods']:
        method_path = os.path.join(DATA_CONFIG['raw_path'], "manipulated_sequences", 
                                 method.lower(), "crop")
        for video in tqdm(os.listdir(method_path)[:DATA_CONFIG['num_samples']]):
            frames = extract_frames(os.path.join(method_path, video),
                                 DATA_CONFIG['frames_per_video'])
            X.extend(frames)
            y.extend([1] * len(frames))
    
    return np.array(X), np.array(y)

def save_dataset(X, y):
    """Save processed dataset"""
    os.makedirs(DATA_CONFIG['processed_path'], exist_ok=True)
    with open(os.path.join(DATA_CONFIG['processed_path'], 'dataset.pkl'), 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)

if __name__ == "__main__":
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    save_dataset({'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test)})
    print("Dataset prepared and saved successfully!")