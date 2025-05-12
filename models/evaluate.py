import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model_utils import load_model
from utils.visualization import plot_confusion_matrix
from utils.config import DATA_CONFIG

def evaluate():
    # Load model and data
    model = load_model()
    with open(os.path.join(DATA_CONFIG['processed_path'], 'dataset.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    X_test, y_test = data['test']
    
    # Evaluate
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test AUC: {results[2]:.4f}")
    
    # Classification report
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    evaluate()