"""
FER2013 Dataset Loading and Processing

This module provides functions to load and preprocess the FER2013 emotion dataset.
It handles data loading, image preprocessing, and normalization.
"""

import pandas as pd
import cv2
import numpy as np

# Dataset configuration
DATASET_PATH = 'fer2013/fer2013/fer2013.csv'
IMAGE_SIZE = (48, 48)

def load_fer2013():
    """
    Load the FER2013 dataset from CSV file.
    
    Returns:
        tuple: (faces, emotions) where faces is numpy array of shape (N, 48, 48, 1)
               and emotions is one-hot encoded labels of shape (N, 7)
    """
    try:
        print(f"Loading dataset from: {DATASET_PATH}")
        data = pd.read_csv(DATASET_PATH)
        
        # Extract pixel data
        pixels = data['pixels'].tolist()
        width, height = IMAGE_SIZE
        
        faces = []
        for pixel_sequence in pixels:
            # Convert pixel string to integer array
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            
            # Resize to target size
            face = cv2.resize(face.astype('uint8'), IMAGE_SIZE)
            faces.append(face.astype('float32'))
        
        # Convert to numpy array and add channel dimension
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        
        # Convert emotion labels to one-hot encoding
        emotions = pd.get_dummies(data['emotion']).values
        
        print(f"Loaded {len(faces)} face images")
        print(f"Face shape: {faces.shape}")
        print(f"Emotion labels shape: {emotions.shape}")
        
        return faces, emotions
        
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        print("Please download the FER2013 dataset and place it in the correct location")
        return None, None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None

def preprocess_input(x, v2=True):
    """
    Preprocess input images for model training.
    
    Args:
        x (numpy.ndarray): Input images array
        v2 (bool): Whether to use v2 preprocessing (normalize to [-1, 1])
    
    Returns:
        numpy.ndarray: Preprocessed images
    """
    # Convert to float32
    x = x.astype('float32')
    
    # Normalize to [0, 1]
    x = x / 255.0
    
    if v2:
        # Normalize to [-1, 1]
        x = x - 0.5
        x = x * 2.0
    
    return x

def get_emotion_labels():
    """
    Get the emotion label names.
    
    Returns:
        list: List of emotion label names
    """
    return ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def get_dataset_info():
    """
    Get basic information about the dataset.
    
    Returns:
        dict: Dictionary containing dataset information
    """
    try:
        data = pd.read_csv(DATASET_PATH)
        return {
            'total_samples': len(data),
            'emotion_distribution': data['emotion'].value_counts().to_dict(),
            'image_size': IMAGE_SIZE,
            'num_classes': 7
        }
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    faces, emotions = load_fer2013()
    
    if faces is not None:
        print("Dataset loaded successfully!")
        print(f"Faces shape: {faces.shape}")
        print(f"Emotions shape: {emotions.shape}")
        
        # Test preprocessing
        processed_faces = preprocess_input(faces)
        print(f"Preprocessed faces shape: {processed_faces.shape}")
        print(f"Preprocessed faces range: [{processed_faces.min():.3f}, {processed_faces.max():.3f}]")
    else:
        print("Failed to load dataset")