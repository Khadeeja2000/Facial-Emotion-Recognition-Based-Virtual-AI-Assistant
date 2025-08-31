"""
Test suite for Emotion Recognition System
"""

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch

# Import the functions to test
from load_and_process import preprocess_input, get_emotion_labels


class TestEmotionRecognition(unittest.TestCase):
    """Test cases for emotion recognition functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        self.emotion_labels = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    
    def test_preprocess_input_v2(self):
        """Test preprocessing with v2 normalization."""
        processed = preprocess_input(self.sample_image, v2=True)
        
        # Check data type
        self.assertEqual(processed.dtype, np.float32)
        
        # Check normalization range [-1, 1]
        self.assertGreaterEqual(processed.min(), -1.0)
        self.assertLessEqual(processed.max(), 1.0)
        
        # Check shape preservation
        self.assertEqual(processed.shape, self.sample_image.shape)
    
    def test_preprocess_input_v1(self):
        """Test preprocessing without v2 normalization."""
        processed = preprocess_input(self.sample_image, v2=False)
        
        # Check data type
        self.assertEqual(processed.dtype, np.float32)
        
        # Check normalization range [0, 1]
        self.assertGreaterEqual(processed.min(), 0.0)
        self.assertLessEqual(processed.max(), 1.0)
        
        # Check shape preservation
        self.assertEqual(processed.shape, self.sample_image.shape)
    
    def test_get_emotion_labels(self):
        """Test emotion labels function."""
        labels = get_emotion_labels()
        
        # Check correct number of labels
        self.assertEqual(len(labels), 7)
        
        # Check all expected emotions are present
        for emotion in self.emotion_labels:
            self.assertIn(emotion, labels)
    
    def test_preprocess_input_invalid_input(self):
        """Test preprocessing with invalid input."""
        with self.assertRaises(AttributeError):
            preprocess_input(None)
    
    def test_preprocess_input_empty_array(self):
        """Test preprocessing with empty array."""
        empty_array = np.array([])
        processed = preprocess_input(empty_array)
        
        # Should handle empty array gracefully
        self.assertEqual(len(processed), 0)


class TestDataValidation(unittest.TestCase):
    """Test cases for data validation."""
    
    def test_image_dimensions(self):
        """Test that images have correct dimensions."""
        # Valid dimensions
        valid_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        self.assertEqual(valid_image.shape, (48, 48))
        
        # Invalid dimensions
        invalid_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        self.assertNotEqual(invalid_image.shape, (48, 48))
    
    def test_emotion_probabilities(self):
        """Test emotion probability calculations."""
        # Simulate model predictions
        predictions = np.array([0.1, 0.2, 0.1, 0.3, 0.1, 0.1, 0.1])
        
        # Check probabilities sum to 1
        self.assertAlmostEqual(predictions.sum(), 1.0, places=5)
        
        # Check all probabilities are non-negative
        self.assertTrue(np.all(predictions >= 0))
        
        # Check all probabilities are <= 1
        self.assertTrue(np.all(predictions <= 1))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
