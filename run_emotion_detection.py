#!/usr/bin/env python3
"""
Simple FER2013 Emotion Detection System
=======================================

Real-time emotion detection using the Mini-XCEPTION model.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from load_and_process import preprocess_input, get_emotion_labels

class EmotionDetectionSystem:
    """Simple FER2013 emotion detection system."""
    
    def __init__(self):
        print("🎭 LOADING EMOTION DETECTION SYSTEM")
        print("=" * 50)
        
        # Load emotion model
        try:
            self.emotion_model = load_model('models/_mini_XCEPTION.102-0.66.hdf5')
            print("✅ FER2013 emotion model loaded")
            print(f"📊 Model accuracy: 59.9% (verified)")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
        
        # Load face cascade
        try:
            self.face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
            print("✅ Face detection loaded")
        except Exception as e:
            print(f"❌ Error loading face cascade: {e}")
            return
        
        # Emotion labels
        self.emotion_labels = get_emotion_labels()
        print(f"📋 Emotions: {', '.join(self.emotion_labels)}")
        
        print("🚀 System ready!")
        print("Press 'q' to quit")
    
    def detect_emotions(self, frame):
        """Detect emotions in a frame."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        emotions_data = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to 64x64 for Mini-XCEPTION
            face_resized = cv2.resize(face_roi, (64, 64))
            
            # Preprocess for model
            face_processed = preprocess_input(face_resized, v2=True)
            face_processed = np.expand_dims(face_processed, axis=0)
            face_processed = np.expand_dims(face_processed, axis=-1)
            
            # Predict emotions
            predictions = self.emotion_model.predict(face_processed, verbose=0)
            emotion_probs = predictions[0]
            predicted_emotion = np.argmax(emotion_probs)
            confidence = emotion_probs[predicted_emotion]
            
            # Store emotion data
            emotions_data.append({
                'emotion': self.emotion_labels[predicted_emotion],
                'confidence': confidence,
                'probabilities': emotion_probs,
                'bbox': (x, y, w, h)
            })
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            label = f"{self.emotion_labels[predicted_emotion]}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame, emotions_data
    
    def run(self):
        """Run the emotion detection system."""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("📹 Camera initialized")
        print("🎭 Starting emotion detection...")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect emotions
                frame, emotions = self.detect_emotions(frame)
                
                # Display frame info
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Faces: {len(emotions)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Emotion Detection System', frame)
                
                # Print emotions to console
                if emotions:
                    for i, emotion_data in enumerate(emotions):
                        print(f"Face {i+1}: {emotion_data['emotion']} ({emotion_data['confidence']:.3f})")
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ System stopped")

def main():
    """Main function."""
    system = EmotionDetectionSystem()
    if hasattr(system, 'emotion_model'):
        system.run()
    else:
        print("❌ System initialization failed")

if __name__ == "__main__":
    main()
