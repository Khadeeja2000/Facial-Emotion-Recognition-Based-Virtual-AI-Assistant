"""
Enhanced Emotion Classification Training Script

This script implements advanced training techniques to improve emotion recognition accuracy:
- Data augmentation and preprocessing
- Model ensemble methods
- Advanced optimization techniques
- Cross-validation and hyperparameter tuning
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.utils import to_categorical

from load_and_process import load_fer2013, preprocess_input, get_emotion_labels

# Enhanced training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 150
INPUT_SHAPE = (64, 64, 3)  # Increased resolution and color channels
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
PATIENCE = 25
BASE_PATH = 'models/'

def create_enhanced_data_generator():
    """Create advanced data generator with extensive augmentation."""
    return ImageDataGenerator(
        # Geometric transformations
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        
        # Color transformations
        brightness_range=[0.8, 1.2],
        channel_shift_range=20,
        
        # Noise and blur
        preprocessing_function=lambda x: x + np.random.normal(0, 0.01, x.shape),
        
        # Normalization
        featurewise_center=True,
        featurewise_std_normalization=True,
        
        # Fill mode for transformations
        fill_mode='nearest',
        cval=0.0
    )

def create_ensemble_model(input_shape, num_classes):
    """Create an ensemble model combining multiple architectures."""
    
    # Base input
    input_layer = Input(shape=input_shape)
    
    # Model 1: Enhanced CNN
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.25)(x1)
    
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.25)(x1)
    
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.25)(x1)
    
    x1 = GlobalAveragePooling2D()(x1)
    x1 = Dense(256, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.5)(x1)
    
    # Model 2: Transfer Learning with MobileNetV2
    base_model2 = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_layer)
    base_model2.trainable = False  # Freeze pre-trained weights initially
    
    x2 = base_model2.output
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dense(256, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.5)(x2)
    
    # Model 3: EfficientNet
    base_model3 = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input_layer)
    base_model3.trainable = False
    
    x3 = base_model3.output
    x3 = GlobalAveragePooling2D()(x3)
    x3 = Dense(256, activation='relu')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.5)(x3)
    
    # Combine all models
    combined = Concatenate()([x1, x2, x3])
    
    # Final classification layers
    x = Dense(512, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    return model

def create_advanced_callbacks():
    """Create advanced training callbacks."""
    os.makedirs(BASE_PATH, exist_ok=True)
    
    # Model checkpointing
    checkpoint_path = os.path.join(BASE_PATH, 'enhanced_emotion_model_{epoch:02d}_{val_accuracy:.3f}.hdf5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=PATIENCE // 2,
        min_lr=1e-7,
        verbose=1
    )
    
    # Logging
    csv_logger = CSVLogger(
        os.path.join(BASE_PATH, 'enhanced_training_log.csv'),
        append=False
    )
    
    return [checkpoint, early_stopping, lr_reducer, csv_logger]

def prepare_enhanced_data():
    """Prepare and enhance training data."""
    print("Loading and preparing enhanced dataset...")
    
    # Load original data
    faces, emotions = load_fer2013()
    
    if faces is None:
        raise ValueError("Failed to load dataset")
    
    # Convert to RGB (3 channels) for transfer learning
    faces_rgb = np.repeat(faces, 3, axis=-1)
    
    # Resize to higher resolution
    faces_resized = []
    for face in faces_rgb:
        face_resized = cv2.resize(face, (64, 64))
        faces_resized.append(face_resized)
    
    faces_resized = np.array(faces_resized)
    
    # Enhanced preprocessing
    faces_processed = preprocess_input(faces_resized, v2=True)
    
    print(f"Enhanced dataset shape: {faces_processed.shape}")
    print(f"Input shape: {faces_processed.shape[1:]}")
    
    return faces_processed, emotions

def train_with_cross_validation(X, y, num_folds=5):
    """Train model using k-fold cross-validation for better generalization."""
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    fold_scores = []
    fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold + 1}/{num_folds}")
        print(f"{'='*50}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create model
        model = create_ensemble_model(INPUT_SHAPE, y.shape[1])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create callbacks
        callbacks = create_advanced_callbacks()
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        scores = model.evaluate(X_val, y_val, verbose=0)
        fold_scores.append(scores[1])  # accuracy
        fold_histories.append(history)
        
        print(f"Fold {fold + 1} - Accuracy: {scores[1]:.4f}")
        
        # Save fold model
        model.save(os.path.join(BASE_PATH, f'fold_{fold + 1}_model.hdf5'))
    
    return fold_scores, fold_histories

def evaluate_ensemble_models(X_test, y_test):
    """Evaluate all trained models and create ensemble prediction."""
    model_files = [f for f in os.listdir(BASE_PATH) if f.endswith('.hdf5') and 'fold' in f]
    
    if not model_files:
        print("No fold models found for ensemble evaluation")
        return
    
    predictions = []
    
    for model_file in model_files:
        model_path = os.path.join(BASE_PATH, model_file)
        model = load_model(model_path)
        
        # Get predictions
        pred = model.predict(X_test, verbose=0)
        predictions.append(pred)
        
        # Evaluate individual model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print(f"{model_file} - Accuracy: {scores[1]:.4f}")
    
    # Ensemble prediction (average)
    ensemble_pred = np.mean(predictions, axis=0)
    ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate ensemble accuracy
    ensemble_accuracy = np.mean(ensemble_pred_classes == y_test_classes)
    print(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.4f}")
    
    return ensemble_pred, ensemble_accuracy

def plot_training_results(histories, fold_scores):
    """Plot training results and cross-validation scores."""
    plt.figure(figsize=(15, 5))
    
    # Plot training curves for each fold
    plt.subplot(1, 3, 1)
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'Fold {i+1} Train')
        plt.plot(history.history['val_accuracy'], label=f'Fold {i+1} Val')
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot cross-validation scores
    plt.subplot(1, 3, 2)
    plt.bar(range(1, len(fold_scores) + 1), fold_scores)
    plt.title('Cross-Validation Scores')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Plot loss curves
    plt.subplot(1, 3, 3)
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'Fold {i+1} Train')
        plt.plot(history.history['val_loss'], label=f'Fold {i+1} Val')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, 'training_results.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function."""
    print("Enhanced Emotion Classification Training")
    print("="*50)
    
    try:
        # Prepare enhanced data
        X, y = prepare_enhanced_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Train with cross-validation
        fold_scores, fold_histories = train_with_cross_validation(X_train, y_train, num_folds=5)
        
        # Print cross-validation results
        print(f"\nCross-Validation Results:")
        print(f"Mean Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
        print(f"Best Accuracy: {np.max(fold_scores):.4f}")
        
        # Evaluate ensemble
        ensemble_pred, ensemble_accuracy = evaluate_ensemble_models(X_test, y_test)
        
        # Plot results
        plot_training_results(fold_histories, fold_scores)
        
        print(f"\nTraining completed successfully!")
        print(f"Expected accuracy improvement: 66% → {ensemble_accuracy*100:.1f}%")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
