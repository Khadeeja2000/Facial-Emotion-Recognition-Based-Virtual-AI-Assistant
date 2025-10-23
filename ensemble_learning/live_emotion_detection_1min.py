"""
Live Emotion Detection for 1 Minute using Mini-XCEPTION
Uses existing trained model from ensemble
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from collections import Counter

# Emotion mapping
emotion_map = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

def load_mini_xception_model():
    """Load the trained Mini-XCEPTION model"""
    print("Loading Mini-XCEPTION model...")
    
    try:
        # Try to load existing model from ensemble results
        model = keras.models.load_model('ensemble_learning/results/mini_xception_ensemble.h5')
        print("✓ Mini-XCEPTION model loaded successfully!")
        return model
    except Exception as e:
        try:
            # Try alternative path
            model = keras.models.load_model('results/mini_xception_ensemble.h5')
            print("✓ Mini-XCEPTION model loaded successfully!")
            return model
        except:
            print(f"❌ Could not find trained Mini-XCEPTION model")
            print("Searched: 'ensemble_learning/results/mini_xception_ensemble.h5'")
            print("Please ensure the model file exists.")
            return None

def preprocess_face(face_img):
    """Preprocess face for Mini-XCEPTION model"""
    # Resize to 48x48
    face_img = cv2.resize(face_img, (48, 48))
    
    # Convert to grayscale
    if len(face_img.shape) == 3:
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_img
    
    # Normalize
    face_gray = face_gray.astype('float32') / 255.0
    
    # Add channel dimension (grayscale = 1 channel)
    face_gray = np.expand_dims(face_gray, axis=-1)
    
    # Add batch dimension
    face_gray = np.expand_dims(face_gray, axis=0)
    
    return face_gray

def live_emotion_detection_1min():
    """Run live emotion detection for 1 minute"""
    print("=" * 60)
    print("LIVE EMOTION DETECTION - 1 MINUTE")
    print("=" * 60)
    
    # Load model
    model = load_mini_xception_model()
    if model is None:
        return
    
    # Load Haar Cascade for face detection
    print("Loading face detector...")
    try:
        face_cascade = cv2.CascadeClassifier('../haarcascade_files/haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("✓ Face detector loaded!")
    except:
        print("❌ Could not load face detector")
        return
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("✓ Webcam opened!")
    print("\n" + "=" * 60)
    print("STARTING LIVE DETECTION FOR 1 MINUTE")
    print("Press 'q' to quit early")
    print("=" * 60)
    
    # Track statistics
    start_time = time.time()
    duration = 60  # 1 minute
    emotion_counts = Counter()
    frame_count = 0
    detected_faces = 0
    
    while True:
        # Check if 1 minute has passed
        elapsed_time = time.time() - start_time
        if elapsed_time >= duration:
            break
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Process each face
        for (x, y, w, h) in faces:
            detected_faces += 1
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess face
            processed_face = preprocess_face(face_roi)
            
            # Predict emotion
            predictions = model.predict(processed_face, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            emotion_label = emotion_map[emotion_idx]
            confidence = predictions[0][emotion_idx]
            
            # Update statistics
            emotion_counts[emotion_label] += 1
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display emotion and confidence
            text = f"{emotion_label}: {confidence*100:.1f}%"
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display time remaining
        remaining_time = duration - elapsed_time
        time_text = f"Time remaining: {int(remaining_time)}s"
        cv2.putText(frame, time_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame count
        fps_text = f"FPS: {frame_count/elapsed_time:.1f}"
        cv2.putText(frame, fps_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Live Emotion Detection (1 min)', frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nDetection stopped by user")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Display statistics
    print("\n" + "=" * 60)
    print("DETECTION COMPLETED!")
    print("=" * 60)
    print(f"Duration: {elapsed_time:.1f} seconds")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {frame_count/elapsed_time:.1f}")
    print(f"Total faces detected: {detected_faces}")
    
    if emotion_counts:
        print("\n" + "=" * 60)
        print("EMOTION STATISTICS:")
        print("=" * 60)
        total_detections = sum(emotion_counts.values())
        for emotion, count in emotion_counts.most_common():
            percentage = (count / total_detections) * 100
            print(f"{emotion:12}: {count:4} ({percentage:5.1f}%)")
        
        print("\n" + "=" * 60)
        print(f"Most frequent emotion: {emotion_counts.most_common(1)[0][0]}")
        print("=" * 60)
    else:
        print("\nNo faces detected during the session")

if __name__ == "__main__":
    live_emotion_detection_1min()

