#!/usr/bin/env python3
"""
MobileNetV2 Transfer Learning Training Script
============================================

This script trains MobileNetV2 with transfer learning for emotion recognition.
Achieved 60.5% accuracy with two-phase training strategy.

Usage:
    python train_mobilenetv2.py --epochs_frozen 5 --epochs_finetune 10 --batch_size 64
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
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

# Configuration
EMOTION_LABELS = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']
NUM_CLASSES = len(EMOTION_LABELS)
IMG_SIZE_RAW = 48
IMG_RESIZED = 160

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
                        img = cv2.resize(img, (IMG_SIZE_RAW, IMG_SIZE_RAW))
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

def preprocess_for_mobilenetv2(images):
    """Preprocess images for MobileNetV2."""
    processed_images = []
    
    for img in images:
        # Convert grayscale to RGB
        img_rgb = np.repeat(img, 3, axis=-1)
        
        # Resize to MobileNetV2 input size
        img_resized = cv2.resize(img_rgb, (IMG_RESIZED, IMG_RESIZED))
        
        # Preprocess for MobileNetV2
        img_processed = preprocess_input(img_resized)
        processed_images.append(img_processed)
    
    return np.array(processed_images)

def create_mobilenetv2_model(trainable_backbone=False):
    """Create MobileNetV2 model with transfer learning."""
    print("Creating MobileNetV2 model...")
    
    # Create base model
    base_model = MobileNetV2(
        input_shape=(IMG_RESIZED, IMG_RESIZED, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_backbone
    
    # Add custom head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model, base_model

def create_data_generators(x_train, y_train, x_val, y_val, batch_size=64):
    """Create data generators with augmentation."""
    
    # Training data generator with augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.12,
        zoom_range=0.12,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode='nearest'
    )
    
    # Validation data generator (no augmentation)
    val_datagen = keras.preprocessing.image.ImageDataGenerator()
    
    return train_datagen, val_datagen

def train_phase1(model, x_train, y_train, x_val, y_val, epochs=5, batch_size=64):
    """Phase 1: Train with frozen backbone."""
    print(f"Phase 1: Training with frozen backbone ({epochs} epochs)...")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create data generators
    train_datagen, val_datagen = create_data_generators(x_train, y_train, x_val, y_val, batch_size)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    # Train
    history = model.fit(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
        validation_steps=len(x_val) // batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def train_phase2(model, base_model, x_train, y_train, x_val, y_val, epochs=10, batch_size=64):
    """Phase 2: Fine-tune with unfrozen upper layers."""
    print(f"Phase 2: Fine-tuning upper layers ({epochs} epochs)...")
    
    # Unfreeze upper layers
    unfreeze_from = "block_13_expand"
    unfreeze = False
    for layer in base_model.layers:
        if layer.name == unfreeze_from:
            unfreeze = True
        layer.trainable = unfreeze
    
    print(f"Unfrozen layers: {sum(l.trainable for l in base_model.layers)}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create data generators
    train_datagen, val_datagen = create_data_generators(x_train, y_train, x_val, y_val, batch_size)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint('mobilenetv2_best.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    # Train
    history = model.fit(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
        validation_steps=len(x_val) // batch_size,
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
    print(classification_report(y_true_classes, y_pred_classes, target_names=EMOTION_LABELS))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title('MobileNetV2 Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('mobilenetv2_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return accuracy, cm

def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 model")
    parser.add_argument('--epochs_frozen', type=int, default=5, help='Epochs for frozen backbone')
    parser.add_argument('--epochs_finetune', type=int, default=10, help='Epochs for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--save_model', type=str, default='mobilenetv2_model.h5', help='Model save path')
    
    args = parser.parse_args()
    
    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_fer2013_data()
    
    # Preprocess for MobileNetV2
    print("Preprocessing images for MobileNetV2...")
    x_train = preprocess_for_mobilenetv2(x_train)
    x_val = preprocess_for_mobilenetv2(x_val)
    x_test = preprocess_for_mobilenetv2(x_test)
    
    # Create model
    model, base_model = create_mobilenetv2_model(trainable_backbone=False)
    model.summary()
    
    # Phase 1: Train with frozen backbone
    if args.epochs_frozen > 0:
        history1 = train_phase1(model, x_train, y_train, x_val, y_val, args.epochs_frozen, args.batch_size)
    
    # Phase 2: Fine-tune upper layers
    if args.epochs_finetune > 0:
        history2 = train_phase2(model, base_model, x_train, y_train, x_val, y_val, args.epochs_finetune, args.batch_size)
    
    # Evaluate model
    accuracy, cm = evaluate_model(model, x_test, y_test)
    
    # Save final model
    model.save(args.save_model)
    print(f"Model saved to {args.save_model}")
    print(f"Final Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
