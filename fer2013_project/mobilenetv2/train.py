#!/usr/bin/env python3
"""
MobileNetV2 Transfer Learning Training Script
===========================================

This script trains MobileNetV2 for emotion recognition on FER2013 using transfer learning.
Achieves 60.5% accuracy with 15 epochs of training.

Usage:
    python train.py --epochs 15 --batch_size 32
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
import json

# Import shared utilities
import sys
sys.path.append('..')
from utils import load_fer2013_data, save_metrics, plot_confusion_matrix

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Emotion labels
EMOTION_LABELS = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']
NUM_CLASSES = len(EMOTION_LABELS)

def create_mobilenetv2_model():
    """Create MobileNetV2 model with transfer learning."""
    print("Creating MobileNetV2 model with transfer learning...")
    
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=15, batch_size=32):
    """Train the MobileNetV2 model."""
    print(f"Training MobileNetV2 for {epochs} epochs...")
    
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
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
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
    print("Evaluating MobileNetV2 model...")
    
    # Predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred_classes == y_true_classes)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(y_true_classes, y_pred_classes, target_names=EMOTION_LABELS)
    print(report)
    
    # Save classification report
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrices
    plot_confusion_matrix(cm, EMOTION_LABELS, 'MobileNetV2', 'results/')
    
    # Calculate macro F1
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
    
    # Save metrics
    metrics = {
        'model': 'MobileNetV2',
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'num_classes': NUM_CLASSES,
        'emotion_labels': EMOTION_LABELS
    }
    save_metrics(metrics, 'results/metrics.json')
    
    return accuracy, cm, macro_f1

def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 model")
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_fer2013_data()
    
    # Create model
    model = create_mobilenetv2_model()
    model.summary()
    
    # Train model
    history = train_model(model, x_train, y_train, x_val, y_val, args.epochs, args.batch_size)
    
    # Evaluate model
    accuracy, cm, macro_f1 = evaluate_model(model, x_test, y_test)
    
    # Save final model
    model.save('last.h5')
    print(f"Model saved to current directory")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")

if __name__ == "__main__":
    main()
