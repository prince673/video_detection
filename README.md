# video_detection

# Deepfake Video Detection

A deep learning system for detecting manipulated videos using the FaceForensics dataset.

## Features

- EfficientNet-based deepfake detection
- Frame-level and video-level prediction
- Comprehensive evaluation metrics
- Easy-to-use prediction API

## Installation

```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
Usage
Download the dataset:

bash
python data/download_dataset.py
Preprocess the data:

bash
python data/prepare_dataset.py
Train the model:

bash
python models/train.py
Evaluate the model:

bash
python models/evaluate.py
Make predictions:

bash
python models/predict.py --input path/to/video.mp4
