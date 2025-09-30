#!/usr/bin/env python3
"""
Evaluate Mini-XCEPTION Model Performance
=======================================

This script evaluates the actual Mini-XCEPTION model to get its real accuracy.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Emotion labels
EMOTION_LABELS = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']
NUM_CLASSES = len(EMOTION_LABELS)

def load_fer2013_test_data():
    """Load FER2013 test data from image folders."""
    print("Loading FER2013 test dataset...")
    
    data_path = Path("fer2013/fer2013/archive-2")
    test_path = data_path / "test"
    
    images = []
    labels = []
    
    for emotion_idx, emotion in enumerate(EMOTION_LABELS):
        emotion_path = test_path / emotion
        if emotion_path.exists():
            image_files = list(emotion_path.glob("*.jpg")) + list(emotion_path.glob("*.png"))
            print(f"  {emotion}: {len(image_files)} images")
            
            for img_file in image_files:
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to 64x64 (Mini-XCEPTION standard)
                    img = cv2.resize(img, (64, 64))
                    img = img.astype('float32') / 255.0
                    img = np.expand_dims(img, axis=-1)  # Add channel dimension
                    images.append(img)
                    labels.append(emotion_idx)
    
    images = np.array(images)
    labels = np.array(labels)
    labels_categorical = to_categorical(labels, NUM_CLASSES)
    
    print(f"Test data loaded: {len(images)} images")
    return images, labels, labels_categorical

def evaluate_model():
    """Evaluate the Mini-XCEPTION model."""
    print("=" * 60)
    print("EVALUATING MINI-XCEPTION MODEL")
    print("=" * 60)
    
    # Load model
    model_path = "models/_mini_XCEPTION.102-0.66.hdf5"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded successfully")
    
    # Load test data
    x_test, y_test, y_test_categorical = load_fer2013_test_data()
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(x_test, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"\n🎯 TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Classification report
    print("\n📊 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred_classes, target_names=EMOTION_LABELS))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title(f'Mini-XCEPTION Confusion Matrix\nAccuracy: {accuracy:.4f} ({accuracy*100:.1f}%)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('mini_xception_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📈 Confusion matrix saved as 'mini_xception_evaluation.png'")
    
    # Per-class accuracy
    print("\n📋 PER-CLASS ACCURACY:")
    for i, emotion in enumerate(EMOTION_LABELS):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.sum((y_pred_classes[class_mask] == y_test[class_mask])) / np.sum(class_mask)
            print(f"  {emotion}: {class_accuracy:.4f} ({class_accuracy*100:.1f}%)")
    
    # Model summary
    print(f"\n📊 MODEL SUMMARY:")
    print(f"  Model: Mini-XCEPTION")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Test Samples: {len(x_test)}")
    print(f"  Classes: {NUM_CLASSES}")
    
    return accuracy

if __name__ == "__main__":
    accuracy = evaluate_model()
    print(f"\n🎉 EVALUATION COMPLETE!")
    print(f"Mini-XCEPTION Model Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
