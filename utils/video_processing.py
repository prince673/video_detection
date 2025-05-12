import cv2
import numpy as np
from tqdm import tqdm
from utils.config import DATA_CONFIG

def extract_frames(video_path, num_frames=10):
    """Extract frames from video file"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        return frames
    
    frame_indices = np.linspace(0, total_frames-1, num=min(num_frames, total_frames), dtype=np.int32)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, DATA_CONFIG['target_size'])
            frames.append(frame)
    
    cap.release()
    return frames

def preprocess_frames(frames):
    """Normalize frames for model input"""
    frames = np.array(frames).astype('float32') / 255.0
    return frames