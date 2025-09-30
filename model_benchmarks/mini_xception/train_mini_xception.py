#!/usr/bin/env python3
"""
Mini-XCEPTION Model Training Script
===================================

This script trains the Mini-XCEPTION model for emotion recognition.
Based on the original integrated system that achieved 68.6% accuracy.

Usage:
    python train_mini_xception.py --epochs 102 --batch_size 32
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Emotion labels
EMOTION_LABELS = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']
NUM_CLASSES = len(EMOTION_LABELS)

def load_fer2013_data():
    """Load FER2013 data from image folders."""
    print("Loading FER2013 dataset...")
    
    data_path = Path("fer2013/fer2013/archive-2")
    train_path = data_path / "train"
    test_path = data_path / "test"
    
    def load_images_from_folder(folder_path):
        images = []
        labels = []
        
        for emotion_idx, emotion in enumerate(EMOTION_LABELS):
            emotion_path = folder_path / emotion
            if emotion_path.exists():
                image_files = list(emotion_path.glob("*.jpg")) + list(emotion_path.glob("*.png"))
                print(f"  {emotion}: {len(image_files)} images")
                
                for img_file in image_files:
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize to 48x48 (FER2013 standard)
                        img = cv2.resize(img, (48, 48))
                        img = img.astype('float32') / 255.0
                        img = np.expand_dims(img, axis=-1)  # Add channel dimension
                        images.append(img)
                        labels.append(emotion_idx)
        
        return np.array(images), np.array(labels)
    
    # Load training and test data
    x_train, y_train = load_images_from_folder(train_path)
    x_test, y_test = load_images_from_folder(test_path)
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    # Split training data into train/validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"Data loaded: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def create_mini_xception_model():
    """Create Mini-XCEPTION model architecture."""
    print("Creating Mini-XCEPTION model...")
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(48, 48, 1)),
        
        # First block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=102, batch_size=32):
    """Train the Mini-XCEPTION model."""
    print(f"Training Mini-XCEPTION for {epochs} epochs...")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'mini_xception_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Train model
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, x_test, y_test):
    """Evaluate the trained model."""
    print("Evaluating Mini-XCEPTION model...")
    
    # Predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred_classes == y_true_classes)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=EMOTION_LABELS))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title('Mini-XCEPTION Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('mini_xception_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return accuracy, cm

def main():
    parser = argparse.ArgumentParser(description="Train Mini-XCEPTION model")
    parser.add_argument('--epochs', type=int, default=102, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--save_model', type=str, default='mini_xception_model.h5', help='Model save path')
    
    args = parser.parse_args()
    
    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_fer2013_data()
    
    # Create model
    model = create_mini_xception_model()
    model.summary()
    
    # Train model
    history = train_model(model, x_train, y_train, x_val, y_val, args.epochs, args.batch_size)
    
    # Evaluate model
    accuracy, cm = evaluate_model(model, x_test, y_test)
    
    # Save final model
    model.save(args.save_model)
    print(f"Model saved to {args.save_model}")
    print(f"Final Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
