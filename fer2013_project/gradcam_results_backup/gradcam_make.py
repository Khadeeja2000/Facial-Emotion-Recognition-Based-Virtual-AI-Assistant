#!/usr/bin/env python3
"""
Grad-CAM Visualization Script for FER2013 Models
===============================================

This script generates Grad-CAM heatmaps for all three models:
- Mini-Xception
- MobileNetV2  
- EfficientNetB0

Usage:
    python3 gradcam_make.py --model mini_xception --image path/to/image.jpg
    python3 gradcam_make.py --model mobilenetv2 --image path/to/image.jpg --last_conv block_16_project_BN
    python3 gradcam_make.py --model efficientnetb0 --image path/to/image.jpg
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    grad_cam, 
    overlay_heatmap, 
    get_model_last_conv_layer, 
    preprocess_image_for_model,
    print_model_layers
)

# Emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_model(model_path, model_name):
    """
    Load a trained model from the specified path.
    
    Args:
        model_path: Path to the model file
        model_name: Name of the model for preprocessing
    
    Returns:
        model: Loaded Keras model
    """
    print(f"Loading {model_name} model from {model_path}...")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_gradcam_for_model(model_name, image_path, last_conv_layer=None, output_dir=None):
    """
    Generate Grad-CAM visualization for a specific model.
    
    Args:
        model_name: Name of the model ('mini_xception', 'mobilenetv2', 'efficientnetb0')
        image_path: Path to the input image
        last_conv_layer: Override for last conv layer name
        output_dir: Output directory for results
    """
    # Set up paths
    model_dir = f"{model_name}"
    model_path = f"{model_dir}/best.h5"
    
    # Check for nested directory structure
    if not os.path.exists(model_path):
        nested_path = f"{model_dir}/{model_name}/best.h5"
        if os.path.exists(nested_path):
            model_path = nested_path
            model_dir = f"{model_dir}/{model_name}"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    # Load model
    model = load_model(model_path, model_name)
    if model is None:
        return False
    
    # Print model layers for debugging
    print(f"\n{model_name.upper()} Model Layers:")
    print_model_layers(model)
    
    # Load and preprocess image
    print(f"\nLoading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False
    
    print(f"Original image shape: {image.shape}")
    
    # Preprocess image according to model requirements
    try:
        processed_image = preprocess_image_for_model(image, model_name)
        print(f"Processed image shape: {processed_image.shape}")
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return False
    
    # Get prediction
    predictions = model.predict(processed_image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    print(f"Predicted emotion: {EMOTION_LABELS[predicted_class]} (confidence: {confidence:.3f})")
    
    # Get last conv layer name
    if last_conv_layer is None:
        last_conv_layer = get_model_last_conv_layer(model_name)
    
    print(f"Using last conv layer: {last_conv_layer}")
    
    # Generate Grad-CAM heatmap
    print("Generating Grad-CAM heatmap...")
    try:
        heatmap = grad_cam(model, processed_image, last_conv_layer, predicted_class)
        print(f"Heatmap shape: {heatmap.shape}")
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        print("Available layers:")
        for i, layer in enumerate(model.layers):
            if 'conv' in layer.name.lower() or 'block' in layer.name.lower():
                print(f"  {layer.name}")
        return False
    
    # Create overlay
    print("Creating heatmap overlay...")
    overlay = overlay_heatmap(heatmap, image, alpha=0.4)
    
    # Save results
    if output_dir is None:
        output_dir = f"{model_dir}/results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overlay image
    output_path = f"{output_dir}/gradcam_example.png"
    cv2.imwrite(output_path, overlay)
    print(f"Grad-CAM overlay saved to: {output_path}")
    
    # Save heatmap only
    heatmap_path = f"{output_dir}/gradcam_heatmap.png"
    heatmap_vis = np.uint8(255 * heatmap)
    heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_path, heatmap_vis)
    print(f"Grad-CAM heatmap saved to: {heatmap_path}")
    
    # Save original image for comparison
    original_path = f"{output_dir}/gradcam_original.png"
    cv2.imwrite(original_path, image)
    print(f"Original image saved to: {original_path}")
    
    return True

def main():
    """Main function to run Grad-CAM generation."""
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations for FER2013 models')
    parser.add_argument('--model', required=True, 
                       choices=['mini_xception', 'mobilenetv2', 'efficientnetb0'],
                       help='Model to use for Grad-CAM generation')
    parser.add_argument('--image', required=True,
                       help='Path to input image')
    parser.add_argument('--last_conv', 
                       help='Override default last conv layer name')
    parser.add_argument('--output_dir',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FER2013 Grad-CAM Visualization")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    if args.last_conv:
        print(f"Last conv layer: {args.last_conv}")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Generate Grad-CAM
    success = generate_gradcam_for_model(
        model_name=args.model,
        image_path=args.image,
        last_conv_layer=args.last_conv,
        output_dir=args.output_dir
    )
    
    if success:
        print("\n" + "=" * 60)
        print("Grad-CAM generation completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Grad-CAM generation failed!")
        print("=" * 60)

if __name__ == "__main__":
    main()
