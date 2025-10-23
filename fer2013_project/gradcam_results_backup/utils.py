#!/usr/bin/env python3
"""
Shared Utilities for FER2013 Project
===================================

Common functions for data loading, metrics saving, plotting, and personalized content.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import backend as K

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Emotion labels
EMOTION_LABELS = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']
NUM_CLASSES = len(EMOTION_LABELS)

def load_fer2013_data():
    """Load FER2013 dataset from CSV file."""
    print("Loading FER2013 dataset...")
    
    # Check if data file exists
    data_path = 'data/fer2013.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        print("Please download FER2013 dataset and place it in the data/ folder")
        return None, None, None
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Extract pixels and emotions
    pixels = df['pixels'].values
    emotions = df['emotion'].values
    
    # Convert pixels to images
    images = []
    for pixel_string in pixels:
        pixel_array = np.array(pixel_string.split(), dtype='uint8')
        image = pixel_array.reshape(48, 48)
        images.append(image)
    
    images = np.array(images)
    print(f"Image shape: {images.shape}")
    
    # Encode emotions
    le = LabelEncoder()
    emotions_encoded = le.fit_transform(emotions)
    emotions_categorical = to_categorical(emotions_encoded, num_classes=NUM_CLASSES)
    
    # Split data
    x_train, x_temp, y_train, y_temp = train_test_split(
        images, emotions_categorical, test_size=0.3, random_state=RANDOM_SEED, stratify=emotions_encoded
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=np.argmax(y_temp, axis=1)
    )
    
    # Normalize images
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    print(f"Training set: {x_train.shape}, {y_train.shape}")
    print(f"Validation set: {x_val.shape}, {y_val.shape}")
    print(f"Test set: {x_test.shape}, {y_test.shape}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def save_metrics(metrics, filepath):
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filepath}")

def plot_confusion_matrix(cm, labels, model_name, output_dir):
    """Plot and save confusion matrices."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Counts confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'{model_name} - Confusion Matrix (Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cm_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'{model_name} - Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cm_normalized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrices saved to {output_dir}")

def get_personalized_content():
    """Get personalized content recommendations based on emotions."""
    return {
        'Angry': {
            'title': 'Let\'s help you feel calmer and more relaxed',
            'content': {
                'videos': [
                    'https://www.youtube.com/watch?v=inpok4MKVLM',  # Calming nature sounds
                    'https://www.youtube.com/watch?v=1ZYbU82GVz4',  # Meditation for anger
                    'https://www.youtube.com/watch?v=ZToicYcHIOU'   # Breathing exercises
                ],
                'music': [
                    'Weightless by Marconi Union',
                    'Clair de Lune by Claude Debussy',
                    'River Flows in You by Yiruma',
                    'Spiegel im Spiegel by Arvo Pärt'
                ],
                'activities': [
                    'Take 10 deep breaths slowly',
                    'Go for a 5-minute walk outside',
                    'Listen to calming music',
                    'Practice progressive muscle relaxation',
                    'Write down what\'s bothering you',
                    'Try a short meditation session'
                ]
            }
        },
        'Disgust': {
            'title': 'Let\'s help you feel more positive and refreshed',
            'content': {
                'videos': [
                    'https://www.youtube.com/watch?v=dQw4w9WgXcQ',  # Fun and uplifting
                    'https://www.youtube.com/watch?v=9bZkp7q19f0',  # Comedy clips
                    'https://www.youtube.com/watch?v=YQHsXMglC9A'   # Inspiring stories
                ],
                'music': [
                    'Happy by Pharrell Williams',
                    'Don\'t Stop Me Now by Queen',
                    'Good Vibrations by The Beach Boys',
                    'Walking on Sunshine by Katrina and the Waves'
                ],
                'activities': [
                    'Watch a funny video or movie',
                    'Listen to upbeat music',
                    'Call a friend for a chat',
                    'Do something creative (draw, write)',
                    'Take a refreshing shower',
                    'Go outside and get some fresh air'
                ]
            }
        },
        'Fear': {
            'title': 'Let\'s help you feel safe and secure',
            'content': {
                'videos': [
                    'https://www.youtube.com/watch?v=inpok4MKVLM',  # Calming sounds
                    'https://www.youtube.com/watch?v=1ZYbU82GVz4',  # Grounding techniques
                    'https://www.youtube.com/watch?v=ZToicYcHIOU'   # Breathing exercises
                ],
                'music': [
                    'Weightless by Marconi Union',
                    'Clair de Lune by Claude Debussy',
                    'River Flows in You by Yiruma',
                    'Spiegel im Spiegel by Arvo Pärt'
                ],
                'activities': [
                    'Practice grounding techniques (5-4-3-2-1)',
                    'Take slow, deep breaths',
                    'Call someone you trust',
                    'Write down your fears and challenge them',
                    'Listen to calming music',
                    'Try progressive muscle relaxation'
                ]
            }
        },
        'Happy': {
            'title': 'Great! Let\'s keep this positive energy going',
            'content': {
                'videos': [
                    'https://www.youtube.com/watch?v=dQw4w9WgXcQ',  # Fun content
                    'https://www.youtube.com/watch?v=9bZkp7q19f0',  # Comedy
                    'https://www.youtube.com/watch?v=YQHsXMglC9A'   # Inspiring content
                ],
                'music': [
                    'Happy by Pharrell Williams',
                    'Don\'t Stop Me Now by Queen',
                    'Good Vibrations by The Beach Boys',
                    'Walking on Sunshine by Katrina and the Waves'
                ],
                'activities': [
                    'Share your happiness with others',
                    'Dance to your favorite song',
                    'Plan something fun for later',
                    'Write down what made you happy',
                    'Call someone you love',
                    'Do something creative'
                ]
            }
        },
        'Sad': {
            'title': 'Let\'s help you feel better and more hopeful',
            'content': {
                'videos': [
                    'https://www.youtube.com/watch?v=inpok4MKVLM',  # Calming content
                    'https://www.youtube.com/watch?v=1ZYbU82GVz4',  # Inspiring stories
                    'https://www.youtube.com/watch?v=ZToicYcHIOU'   # Positive affirmations
                ],
                'music': [
                    'Weightless by Marconi Union',
                    'Clair de Lune by Claude Debussy',
                    'River Flows in You by Yiruma',
                    'Spiegel im Spiegel by Arvo Pärt'
                ],
                'activities': [
                    'Talk to someone you trust',
                    'Write down your feelings',
                    'Listen to comforting music',
                    'Take a warm bath or shower',
                    'Go for a gentle walk',
                    'Practice self-compassion'
                ]
            }
        },
        'Surprise': {
            'title': 'Let\'s help you process this surprise and feel balanced',
            'content': {
                'videos': [
                    'https://www.youtube.com/watch?v=inpok4MKVLM',  # Calming content
                    'https://www.youtube.com/watch?v=1ZYbU82GVz4',  # Mindfulness
                    'https://www.youtube.com/watch?v=ZToicYcHIOU'   # Breathing exercises
                ],
                'music': [
                    'Weightless by Marconi Union',
                    'Clair de Lune by Claude Debussy',
                    'River Flows in You by Yiruma',
                    'Spiegel im Spiegel by Arvo Pärt'
                ],
                'activities': [
                    'Take a few deep breaths',
                    'Process what just happened',
                    'Talk to someone about it',
                    'Write down your thoughts',
                    'Listen to calming music',
                    'Take a moment to reflect'
                ]
            }
        },
        'Neutral': {
            'title': 'Let\'s help you feel more engaged and energized',
            'content': {
                'videos': [
                    'https://www.youtube.com/watch?v=dQw4w9WgXcQ',  # Interesting content
                    'https://www.youtube.com/watch?v=9bZkp7q19f0',  # Educational content
                    'https://www.youtube.com/watch?v=YQHsXMglC9A'   # Inspiring content
                ],
                'music': [
                    'Happy by Pharrell Williams',
                    'Don\'t Stop Me Now by Queen',
                    'Good Vibrations by The Beach Boys',
                    'Walking on Sunshine by Katrina and the Waves'
                ],
                'activities': [
                    'Try something new and interesting',
                    'Listen to upbeat music',
                    'Call a friend for a chat',
                    'Do something creative',
                    'Go for a walk outside',
                    'Read something inspiring'
                ]
            }
        }
    }

def create_comparison_plots():
    """Create comparison plots for all models."""
    print("Creating comparison plots...")
    
    # This would be implemented to compare all models
    # For now, just create placeholder
    pass

def load_model_metrics(model_dir):
    """Load metrics for a specific model."""
    metrics_file = f"{model_dir}/results/metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def aggregate_model_metrics():
    """Aggregate metrics from all models."""
    models = ['mini_xception', 'mobilenetv2', 'efficientnetb0']
    aggregated_data = []
    
    for model in models:
        metrics = load_model_metrics(model)
        if metrics:
            aggregated_data.append({
                'model': metrics['model'],
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1']
            })
    
    # Save aggregated metrics
    df = pd.DataFrame(aggregated_data)
    df.to_csv('comparisons/aggregate_metrics.csv', index=False)
    print("Aggregated metrics saved to comparisons/aggregate_metrics.csv")
    
    return df

if __name__ == "__main__":
    # Test functions
    print("Testing utilities...")
    
    # Test data loading
    try:
        data = load_fer2013_data()
        if data:
            print("Data loading test passed")
        else:
            print("Data loading test failed")
    except Exception as e:
        print(f"Data loading test failed: {e}")
    
    # Test personalized content
    content = get_personalized_content()
    print(f"Personalized content loaded for {len(content)} emotions")
    
    print("Utilities test complete")

# =============================================================================
# Grad-CAM Implementation
# =============================================================================

def grad_cam(model, img_tensor, last_conv_layer_name, class_index=None):
    """
    Generate Grad-CAM heatmap for a given model and image.
    
    Args:
        model: Keras model
        img_tensor: Preprocessed image tensor (batch_size=1)
        last_conv_layer_name: Name of the last convolutional layer
        class_index: Index of the class to generate heatmap for (None for predicted class)
    
    Returns:
        heatmap: Grad-CAM heatmap (H, W)
    """
    # Get the last convolutional layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    
    # Create a model that maps the input to the last conv layer and final predictions
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        
        class_output = predictions[:, class_index]
    
    # Get gradients of the class output with respect to the conv outputs
    grads = tape.gradient(class_output, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Get the conv outputs (remove batch dimension)
    conv_outputs = conv_outputs[0]
    
    # Multiply each channel in the feature map by its corresponding gradient
    # Reshape pooled_grads to match conv_outputs
    pooled_grads = tf.reshape(pooled_grads, (1, 1, -1))
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def overlay_heatmap(heatmap, bgr_image, alpha=0.35):
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        heatmap: Grad-CAM heatmap (H, W)
        bgr_image: Original BGR image
        alpha: Transparency of the heatmap overlay
    
    Returns:
        overlay: Image with heatmap overlay
    """
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (bgr_image.shape[1], bgr_image.shape[0]))
    
    # Convert heatmap to 3-channel
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(bgr_image, 1-alpha, heatmap, alpha, 0)
    
    return overlay

def get_model_last_conv_layer(model_name):
    """
    Get the default last convolutional layer name for each model.
    
    Args:
        model_name: Name of the model ('mini_xception', 'mobilenetv2', 'efficientnetb0')
    
    Returns:
        layer_name: Name of the last convolutional layer
    """
    layer_mapping = {
        'mini_xception': 'conv2d_2',  # Last conv layer in Mini-Xception
        'mobilenetv2': 'block_16_project_BN',  # Last conv layer in MobileNetV2
        'efficientnetb0': 'block6a_expand_conv'  # Last conv layer in EfficientNetB0
    }
    
    return layer_mapping.get(model_name.lower(), 'conv2d_2')

def preprocess_image_for_model(image, model_name):
    """
    Preprocess image according to model requirements.
    
    Args:
        image: Input image (BGR format)
        model_name: Name of the model
    
    Returns:
        processed_image: Preprocessed image tensor
    """
    if model_name.lower() == 'mini_xception':
        # Mini-Xception: grayscale, 64x64, [0,1] scaled
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        normalized = resized.astype('float32') / 255.0
        processed = np.expand_dims(normalized, axis=-1)
        processed = np.expand_dims(processed, axis=0)
        
    elif model_name.lower() == 'mobilenetv2':
        # MobileNetV2: RGB, 160x160, mobilenet_v2.preprocess_input
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (160, 160))
        from tensorflow.keras.applications import mobilenet_v2
        processed = mobilenet_v2.preprocess_input(resized.astype('float32'))
        processed = np.expand_dims(processed, axis=0)
        
    elif model_name.lower() == 'efficientnetb0':
        # EfficientNetB0: RGB, 224x224, efficientnet.preprocess_input
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        from tensorflow.keras.applications import efficientnet
        processed = efficientnet.preprocess_input(resized.astype('float32'))
        processed = np.expand_dims(processed, axis=0)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return processed

def print_model_layers(model):
    """
    Print all layer names in the model for debugging.
    
    Args:
        model: Keras model
    """
    print("Model layers:")
    for i, layer in enumerate(model.layers):
        print(f"{i:3d}: {layer.name:30s} - {type(layer).__name__}")
