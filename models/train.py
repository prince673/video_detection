import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model_utils import create_model, get_data_generators
from utils.config import MODEL_CONFIG, TRAIN_CONFIG
from utils.visualization import plot_history

def train():
    # Load data
    train_gen, val_gen = get_data_generators()
    
    # Create model
    model = create_model()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_CONFIG['save_path'],
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=TRAIN_CONFIG['patience'],
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=TRAIN_CONFIG['lr_patience'],
            min_lr=1e-6
        )
    ]
    
    # Training
    history = model.fit(
        train_gen,
        epochs=TRAIN_CONFIG['epochs'],
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # Save and plot results
    plot_history(history)
    return model

if __name__ == "__main__":
    model = train()
    print("Training completed!")