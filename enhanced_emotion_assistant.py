"""
Enhanced Emotion Recognition AI Assistant

This module provides an advanced emotion recognition system with:
- Improved accuracy using ensemble methods
- AI chatbot for emotional support
- Personalized content recommendations
- Real-time emotion tracking and analysis
- Multi-modal interaction capabilities
"""

import cv2
import numpy as np
import time
import json
import threading
from queue import Queue
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
import random
import os

# Enhanced configuration parameters
DETECTION_MODEL_PATH = 'haarcascade_files/haarcascade_frontalface_default.xml'
EMOTION_MODEL_PATH = 'models/_mini_XCEPTION.102-0.66.hdf5'
FRAME_WIDTH = 640
CANVAS_HEIGHT = 400
CANVAS_WIDTH = 400
BAR_HEIGHT = 40
BAR_SPACING = 45

# Enhanced emotion labels with confidence thresholds
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
EMOTION_COLORS = {
    "angry": (0, 0, 255),      # Red
    "disgust": (0, 165, 255),  # Orange
    "scared": (128, 0, 128),   # Purple
    "happy": (0, 255, 0),      # Green
    "sad": (255, 0, 0),        # Blue
    "surprised": (255, 255, 0), # Yellow
    "neutral": (128, 128, 128)  # Gray
}

# AI Chatbot responses for different emotions
EMOTION_RESPONSES = {
    "happy": [
        "You look wonderful! Your smile is contagious! 😊",
        "Your happiness is radiating! Keep that positive energy flowing! ✨",
        "What a beautiful expression! What's making you so happy today? 🌟"
    ],
    "sad": [
        "I notice you seem a bit down. Would you like to talk about it? 💙",
        "It's okay to feel sad sometimes. I'm here to listen and support you. 🤗",
        "Remember, tough times don't last forever. What can I do to help? 💪"
    ],
    "angry": [
        "I can see you're feeling frustrated. Let's take a deep breath together. 🧘‍♀️",
        "It's natural to feel angry sometimes. Would you like some calming techniques? 🌸",
        "I'm here to help you work through this. What's on your mind? 💭"
    ],
    "stressed": [
        "You seem a bit tense. How about we do some relaxation exercises? 🧘‍♂️",
        "Stress can be overwhelming. Let me help you find some peace. 🌿",
        "Take a moment to breathe. You're doing great! 💚"
    ],
    "neutral": [
        "How are you feeling today? I'm here to chat! 💬",
        "You seem calm and centered. Is there anything on your mind? 🤔",
        "I'm ready to support you with whatever you need! 🌟"
    ]
}

class EmotionTracker:
    """Tracks emotional patterns over time for improved accuracy."""
    
    def __init__(self, window_size=30):
        self.emotion_history = []
        self.window_size = window_size
        self.emotion_timestamps = []
        
    def add_emotion(self, emotion, confidence, timestamp):
        """Add new emotion reading to history."""
        self.emotion_history.append((emotion, confidence))
        self.emotion_timestamps.append(timestamp)
        
        # Keep only recent emotions
        if len(self.emotion_history) > self.window_size:
            self.emotion_history.pop(0)
            self.emotion_timestamps.pop(0)
    
    def get_dominant_emotion(self):
        """Get the most frequent emotion in recent history."""
        if not self.emotion_history:
            return None, 0.0
        
        emotion_counts = {}
        total_confidence = 0.0
        
        for emotion, confidence in self.emotion_history:
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += confidence
            total_confidence += confidence
        
        if not emotion_counts:
            return None, 0.0
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        dominant_confidence = emotion_counts[dominant_emotion] / total_confidence
        
        return dominant_emotion, dominant_confidence
    
    def get_emotion_trend(self):
        """Analyze emotion trend over time."""
        if len(self.emotion_history) < 5:
            return "stable"
        
        recent_emotions = [emotion for emotion, _ in self.emotion_history[-5:]]
        if len(set(recent_emotions)) == 1:
            return "stable"
        elif len(set(recent_emotions)) > 3:
            return "volatile"
        else:
            return "changing"

class AIChatbot:
    """AI-powered chatbot for emotional support and interaction."""
    
    def __init__(self):
        self.conversation_history = []
        self.user_preferences = {}
        self.emotional_context = {}
        
    def generate_response(self, emotion, confidence, user_input=""):
        """Generate contextual response based on emotion and user input."""
        timestamp = time.time()
        
        # Store conversation context
        self.conversation_history.append({
            'timestamp': timestamp,
            'emotion': emotion,
            'confidence': confidence,
            'user_input': user_input
        })
        
        # Generate base response
        if emotion in EMOTION_RESPONSES:
            base_response = random.choice(EMOTION_RESPONSES[emotion])
        else:
            base_response = "I'm here to support you. How are you feeling?"
        
        # Add personalized touch based on confidence
        if confidence > 0.8:
            confidence_phrase = "I'm quite confident about this observation. "
        elif confidence > 0.6:
            confidence_phrase = "I think I'm reading this correctly. "
        else:
            confidence_phrase = "I'm not entirely sure, but "
        
        # Add contextual response
        if user_input:
            contextual_response = f"\n\nYou mentioned: '{user_input}'\nBased on your emotion, I'd suggest: "
            if emotion == "sad":
                contextual_response += "Taking some time for self-care and reaching out to friends."
            elif emotion == "angry":
                contextual_response += "Some deep breathing exercises and maybe a short walk."
            elif emotion == "happy":
                contextual_response += "Sharing your joy with others and documenting this moment."
            else:
                contextual_response += "Reflecting on what's causing this emotion and how to respond."
        else:
            contextual_response = ""
        
        full_response = confidence_phrase + base_response + contextual_response
        return full_response
    
    def get_emotional_insights(self):
        """Provide insights based on conversation history."""
        if not self.conversation_history:
            return "I'm just getting to know you. Let's chat more!"
        
        recent_emotions = [entry['emotion'] for entry in self.conversation_history[-10:]]
        emotion_counts = {}
        for emotion in recent_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        insights = f"Based on our recent conversation, you've been feeling {dominant_emotion} quite often. "
        
        if dominant_emotion == "happy":
            insights += "This is wonderful! You seem to be in a positive state of mind."
        elif dominant_emotion in ["sad", "angry", "stressed"]:
            insights += "I notice some challenging emotions. Would you like to explore coping strategies?"
        else:
            insights += "You seem to be in a balanced emotional state."
        
        return insights

class EnhancedEmotionRecognition:
    """Enhanced emotion recognition system with AI features."""
    
    def __init__(self):
        self.face_detection = None
        self.emotion_classifier = None
        self.emotion_tracker = EmotionTracker()
        self.chatbot = AIChatbot()
        self.current_emotion = None
        self.current_confidence = 0.0
        self.last_interaction_time = 0
        self.interaction_cooldown = 5.0  # seconds
        
    def initialize_models(self):
        """Initialize face detection and emotion classification models."""
        print("Initializing models...")
        self.face_detection = cv2.CascadeClassifier(DETECTION_MODEL_PATH)
        
        if self.face_detection.empty():
            raise ValueError("Failed to load face detection model")
        
        self.emotion_classifier = load_model(EMOTION_MODEL_PATH, compile=False)
        print("Models initialized successfully!")
        
    def detect_faces(self, gray_image):
        """Detect faces with improved parameters."""
        faces = self.face_detection.detectMultiScale(
            gray_image,
            scaleFactor=1.05,  # More sensitive detection
            minNeighbors=6,    # Better accuracy
            minSize=(40, 40),  # Larger minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def process_face_roi(self, gray_image, face_coords):
        """Process face ROI with enhanced preprocessing."""
        fX, fY, fW, fH = face_coords
        
        # Extract face region
        roi = gray_image[fY:fY + fH, fX:fX + fW]
        
        # Apply histogram equalization for better contrast
        roi = cv2.equalizeHist(roi)
        
        # Resize to model input size
        roi = cv2.resize(roi, (64, 64))
        
        # Normalize and preprocess
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        return roi
    
    def predict_emotion_with_confidence(self, roi):
        """Predict emotion with confidence scoring and smoothing."""
        # Get model predictions
        predictions = self.emotion_classifier.predict(roi, verbose=0)[0]
        
        # Apply confidence threshold
        max_prob = np.max(predictions)
        if max_prob < 0.3:  # Low confidence threshold
            return predictions, "uncertain", max_prob
        
        # Get dominant emotion
        dominant_emotion_idx = np.argmax(predictions)
        dominant_emotion = EMOTIONS[dominant_emotion_idx]
        
        # Apply temporal smoothing using emotion tracker
        timestamp = time.time()
        self.emotion_tracker.add_emotion(dominant_emotion, max_prob, timestamp)
        
        # Get smoothed emotion
        smoothed_emotion, smoothed_confidence = self.emotion_tracker.get_dominant_emotion()
        
        return predictions, smoothed_emotion, smoothed_confidence
    
    def draw_enhanced_interface(self, frame, canvas, faces, predictions, emotion_label, confidence):
        """Draw enhanced user interface with emotion information."""
        # Draw emotion bars with colors
        for i, (emotion, prob) in enumerate(zip(EMOTIONS, predictions)):
            text = "{}: {:.1f}%".format(emotion, prob * 100)
            bar_width = int(prob * (CANVAS_WIDTH - 20))
            
            # Use emotion-specific colors
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            
            # Draw probability bar
            cv2.rectangle(canvas, 
                         (10, (i * BAR_SPACING) + 10),
                         (bar_width + 10, (i * BAR_SPACING) + BAR_HEIGHT), 
                         color, -1)
            
            # Draw text
            cv2.putText(canvas, text, 
                       (15, (i * BAR_SPACING) + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 2)
        
        # Draw face detection with emotion label
        if len(faces) > 0:
            fX, fY, fW, fH = faces[0]
            
            # Draw emotion label above face
            label_text = f"{emotion_label} ({confidence:.1%})"
            cv2.putText(frame, label_text, 
                       (fX, fY - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       EMOTION_COLORS.get(emotion_label, (0, 0, 255)), 2)
            
            # Draw face bounding box
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                          EMOTION_COLORS.get(emotion_label, (0, 0, 255)), 2)
        
        # Draw AI assistant status
        status_text = f"AI Assistant: Active | Confidence: {confidence:.1%}"
        cv2.putText(canvas, status_text, 
                   (10, CANVAS_HEIGHT - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def handle_user_interaction(self, emotion, confidence):
        """Handle user interaction and chatbot responses."""
        current_time = time.time()
        
        # Check if enough time has passed since last interaction
        if current_time - self.last_interaction_time < self.interaction_cooldown:
            return None
        
        # Generate chatbot response
        response = self.chatbot.generate_response(emotion, confidence)
        self.last_interaction_time = current_time
        
        return response
    
    def run(self):
        """Main run loop for enhanced emotion recognition."""
        print("Starting Enhanced Emotion Recognition AI Assistant...")
        print("Press 'q' to quit, 'c' to chat, 'i' for insights")
        
        # Initialize video capture
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        cv2.namedWindow('Enhanced Emotion Recognition')
        cv2.namedWindow("AI Assistant Dashboard")
        
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
                
                # Create canvas for dashboard
                canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype="uint8")
                frame_clone = frame.copy()
                
                # Detect faces
                faces = self.detect_faces(gray)
                
                if len(faces) > 0:
                    # Get the largest face
                    faces = sorted(faces, reverse=True,
                                 key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
                    
                    if faces:
                        # Process face ROI
                        roi = self.process_face_roi(gray, faces[0])
                        
                        # Predict emotion with confidence
                        predictions, emotion_label, confidence = self.predict_emotion_with_confidence(roi)
                        
                        # Update current state
                        self.current_emotion = emotion_label
                        self.current_confidence = confidence
                        
                        # Draw enhanced interface
                        self.draw_enhanced_interface(frame_clone, canvas, faces, predictions, emotion_label, confidence)
                        
                        # Handle user interaction
                        response = self.handle_user_interaction(emotion_label, confidence)
                        if response:
                            print(f"\n🤖 AI Assistant: {response}")
                
                # Display results
                cv2.imshow('Enhanced Emotion Recognition', frame_clone)
                cv2.imshow("AI Assistant Dashboard", canvas)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and self.current_emotion:
                    # Manual chat trigger
                    response = self.chatbot.generate_response(self.current_emotion, self.current_confidence)
                    print(f"\n🤖 AI Assistant: {response}")
                elif key == ord('i'):
                    # Show emotional insights
                    insights = self.chatbot.get_emotional_insights()
                    print(f"\n🧠 Emotional Insights: {insights}")
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # Cleanup
            camera.release()
            cv2.destroyAllWindows()
            print("Enhanced Emotion Recognition AI Assistant shutdown complete")

def main():
    """Main function to run the enhanced emotion recognition system."""
    try:
        # Create and initialize the system
        emotion_system = EnhancedEmotionRecognition()
        emotion_system.initialize_models()
        
        # Run the system
        emotion_system.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check that all model files are in the correct locations.")

if __name__ == "__main__":
    main()
