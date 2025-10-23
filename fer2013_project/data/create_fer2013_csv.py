#!/usr/bin/env python3
"""
Create FER2013 CSV file from image folders
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import glob

def create_fer2013_csv():
    """Create FER2013 CSV file from image folders."""
    
    # Emotion mapping
    emotion_map = {
        'angry': 0,
        'disgust': 1, 
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }
    
    # Dataset paths
    train_path = '../../fer2013/fer2013/archive-2/train'
    test_path = '../../fer2013/fer2013/archive-2/test'
    
    data = []
    
    # Process train data
    for emotion_name, emotion_id in emotion_map.items():
        emotion_path = os.path.join(train_path, emotion_name)
        if os.path.exists(emotion_path):
            images = glob.glob(os.path.join(emotion_path, '*.jpg'))
            for img_path in images:
                try:
                    # Load image and convert to grayscale
                    img = Image.open(img_path).convert('L')
                    img_array = np.array(img)
                    
                    # Flatten to pixel string
                    pixels = ' '.join(map(str, img_array.flatten()))
                    
                    data.append({
                        'emotion': emotion_id,
                        'pixels': pixels,
                        'Usage': 'Training'
                    })
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Process test data
    for emotion_name, emotion_id in emotion_map.items():
        emotion_path = os.path.join(test_path, emotion_name)
        if os.path.exists(emotion_path):
            images = glob.glob(os.path.join(emotion_path, '*.jpg'))
            for img_path in images:
                try:
                    # Load image and convert to grayscale
                    img = Image.open(img_path).convert('L')
                    img_array = np.array(img)
                    
                    # Flatten to pixel string
                    pixels = ' '.join(map(str, img_array.flatten()))
                    
                    data.append({
                        'emotion': emotion_id,
                        'pixels': pixels,
                        'Usage': 'PrivateTest'
                    })
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv('fer2013.csv', index=False)
    print(f"Created fer2013.csv with {len(df)} samples")
    print(f"Training samples: {len(df[df['Usage'] == 'Training'])}")
    print(f"Test samples: {len(df[df['Usage'] == 'PrivateTest'])}")

if __name__ == "__main__":
    create_fer2013_csv()
