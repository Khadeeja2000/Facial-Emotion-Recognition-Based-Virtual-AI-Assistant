"""
Real-time Emotion Recognition System

This module provides real-time emotion detection using webcam input.
It detects faces, classifies emotions, and displays results in real-time.
"""

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils

# Configuration parameters
DETECTION_MODEL_PATH = 'haarcascade_files/haarcascade_frontalface_default.xml'
EMOTION_MODEL_PATH = 'models/_mini_XCEPTION.102-0.66.hdf5'
FRAME_WIDTH = 300
CANVAS_HEIGHT = 250
CANVAS_WIDTH = 300
BAR_HEIGHT = 35
BAR_SPACING = 35

# Emotion labels
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def initialize_models():
    """Initialize face detection and emotion classification models."""
    face_detection = cv2.CascadeClassifier(DETECTION_MODEL_PATH)
    emotion_classifier = load_model(EMOTION_MODEL_PATH, compile=False)
    return face_detection, emotion_classifier

def detect_faces(gray_image):
    """Detect faces in the grayscale image."""
    return face_detection.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

def process_face_roi(gray_image, face_coords):
    """Process the face region of interest for emotion classification."""
    fX, fY, fW, fH = face_coords
    roi = gray_image[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    return roi

def draw_emotion_bars(canvas, predictions):
    """Draw emotion probability bars on the canvas."""
    for i, (emotion, prob) in enumerate(zip(EMOTIONS, predictions)):
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        bar_width = int(prob * CANVAS_WIDTH)
        
        # Draw probability bar
        cv2.rectangle(canvas, 
                     (7, (i * BAR_SPACING) + 5),
                     (bar_width, (i * BAR_SPACING) + BAR_HEIGHT), 
                     (0, 0, 255), -1)
        
        # Draw text
        cv2.putText(canvas, text, 
                   (10, (i * BAR_SPACING) + 23),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                   (255, 255, 255), 2)

def draw_face_detection(frame, face_coords, emotion_label):
    """Draw face detection box and emotion label on frame."""
    fX, fY, fW, fH = face_coords
    
    # Draw emotion label above face
    cv2.putText(frame, emotion_label, 
               (fX, fY - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    # Draw face bounding box
    cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                  (0, 0, 255), 2)

def main():
    """Main function for real-time emotion recognition."""
    # Initialize models
    face_detection, emotion_classifier = initialize_models()
    
    # Initialize video capture
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('Emotion Recognition')
    
    print("Starting emotion recognition system...")
    print("Press 'q' to quit")
    
    try:
        while True:
            # Capture frame
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Resize frame
            frame = imutils.resize(frame, width=FRAME_WIDTH)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Create canvas for emotion bars
            canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype="uint8")
            frame_clone = frame.copy()
            
            # Detect faces
            faces = detect_faces(gray)
            
            if len(faces) > 0:
                # Get the largest face
                faces = sorted(faces, reverse=True,
                             key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                
                # Process face ROI
                roi = process_face_roi(gray, faces)
                
                # Predict emotions
                predictions = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(predictions)
                emotion_label = EMOTIONS[predictions.argmax()]
                
                # Draw emotion bars
                draw_emotion_bars(canvas, predictions)
                
                # Draw face detection
                draw_face_detection(frame_clone, faces, emotion_label)
            
            # Display results
            cv2.imshow('Emotion Recognition', frame_clone)
            cv2.imshow("Emotion Probabilities", canvas)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        print("System shutdown complete")

if __name__ == "__main__":
    main()
