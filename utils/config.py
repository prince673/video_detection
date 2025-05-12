import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'data': {
        'raw_path': os.path.join(BASE_DIR, 'data', 'raw'),
        'processed_path': os.path.join(BASE_DIR, 'data', 'processed'),
        'methods': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
        'num_samples': 200,
        'frames_per_video': 5,
        'target_size': (224, 224)
    },
    'model': {
        'input_shape': (224, 224, 3),
        'save_path': os.path.join(BASE_DIR, 'models', 'saved_models', 'best_model.h5'),
        'final_path': os.path.join(BASE_DIR, 'models', 'saved_models', 'final_model.h5')
    },
    'train': {
        'batch_size': 32,
        'epochs': 30,
        'patience': 5,
        'lr_patience': 3,
        'initial_lr': 1e-3,
        'fine_tune_lr': 1e-5
    }
}

DATA_CONFIG = CONFIG['data']
MODEL_CONFIG = CONFIG['model']
TRAIN_CONFIG = CONFIG['train']