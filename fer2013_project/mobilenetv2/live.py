#!/usr/bin/env python3
"""
Live MobileNetV2 Emotion Detection
=================================

Real-time emotion detection using MobileNetV2 model.
Monitors for 1 minute, then provides personalized recommendations.

Usage:
    python live.py
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pygame
import webbrowser
from collections import deque
import time
import os
from datetime import datetime
import pandas as pd

# Import shared utilities
import sys
sys.path.append('..')
from utils import get_personalized_content, grad_cam, overlay_heatmap, preprocess_image_for_model

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class LiveMobileNetV2System:
    """Live emotion detection with MobileNetV2 and personalized recommendations."""
    
    def __init__(self):
        print("Loading Live MobileNetV2 System")
        print("=" * 50)
        
        # Initialize pygame for GUI
        pygame.init()
        
        # Set up display
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Live MobileNetV2 Emotion Detection")
        
        # Colors
        self.colors = {
            'background': (20, 25, 40),
            'primary': (74, 144, 226),
            'secondary': (52, 73, 94),
            'accent': (46, 204, 113),
            'text': (236, 240, 241),
            'text_secondary': (149, 165, 166),
            'success': (39, 174, 96),
            'warning': (241, 196, 15),
            'danger': (231, 76, 60),
            'card': (44, 62, 80),
            'border': (52, 73, 94)
        }
        
        # Fonts
        self.fonts = {
            'title': pygame.font.Font(None, 48),
            'subtitle': pygame.font.Font(None, 32),
            'body': pygame.font.Font(None, 24),
            'small': pygame.font.Font(None, 18),
            'large': pygame.font.Font(None, 36)
        }
        
        # Load MobileNetV2 model
        try:
            self.emotion_model = load_model('best.h5')
            print("MobileNetV2 model loaded (60.5% accuracy)")
        except Exception as e:
            print(f"Error loading MobileNetV2 model: {e}")
            return
        
        # Load face cascade
        try:
            self.face_cascade = cv2.CascadeClassifier('../../haarcascade_files/haarcascade_frontalface_default.xml')
            print("Face detection loaded")
        except Exception as e:
            print(f"Error loading face cascade: {e}")
            return
        
        # Emotion labels
        self.emotion_labels = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']
        
        # System state
        self.frame_count = 0
        self.emotion_buffer = deque(maxlen=60)  # 60 seconds of emotion data
        self.current_emotion = 'neutral'
        self.emotion_confidence = 0.0
        self.monitoring_start_time = None
        self.monitoring_duration = 60  # 1 minute
        
        # UI State
        self.current_screen = 'monitoring'  # monitoring, analysis, recommendations
        self.recommendations = None
        self.dominant_emotion = 'neutral'
        self.avg_confidence = 0.0
        self.buttons = []
        
        # Personalized content database
        self.personalized_content = get_personalized_content()
        
        print("Personalized content database loaded")
        print("System ready!")
    
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
            
            # Resize to 224x224 for MobileNetV2
            face_resized = cv2.resize(face_roi, (224, 224))
            
            # Convert to RGB (3 channels)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
            
            # Preprocess for model
            face_processed = face_rgb.astype('float32') / 255.0
            face_processed = np.expand_dims(face_processed, axis=0)
            
            # Predict emotions
            predictions = self.emotion_model.predict(face_processed, verbose=0)
            emotion_probs = predictions[0]
            predicted_emotion_idx = np.argmax(emotion_probs)
            confidence = emotion_probs[predicted_emotion_idx]
            
            # Store emotion data
            emotion_name = self.emotion_labels[predicted_emotion_idx]
            emotions_data.append({
                'emotion': emotion_name,
                'confidence': confidence,
                'probabilities': emotion_probs,
                'bbox': (x, y, w, h)
            })
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            label = f"{emotion_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame, emotions_data
    
    def update_emotion_buffer(self, emotions):
        """Update emotion buffer with new data."""
        if emotions:
            # Use the most confident emotion
            best_emotion = max(emotions, key=lambda x: x['confidence'])
            self.emotion_buffer.append(best_emotion)
            
            # Update current emotion if confidence is high enough
            if best_emotion['confidence'] > 0.5:
                self.current_emotion = best_emotion['emotion']
                self.emotion_confidence = best_emotion['confidence']
    
    def analyze_emotions(self):
        """Analyze emotions from the 1-minute session."""
        if not self.emotion_buffer:
            return 'neutral', 0.0
        
        # Count emotions
        emotion_counts = {}
        total_confidence = 0.0
        
        for emotion_data in self.emotion_buffer:
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += confidence
        
        # Find most common emotion
        most_common_emotion = max(emotion_counts, key=emotion_counts.get)
        avg_confidence = total_confidence / len(self.emotion_buffer)
        
        return most_common_emotion, avg_confidence
    
    def save_session_data(self):
        """Save live session data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = f"results/live_sessions/sess_{timestamp}"
        os.makedirs(session_dir, exist_ok=True)
        
        # Save emotion data
        emotion_data = []
        for i, emotion_data_point in enumerate(self.emotion_buffer):
            emotion_data.append({
                'frame': i,
                'emotion': emotion_data_point['emotion'],
                'confidence': emotion_data_point['confidence'],
                'timestamp': time.time()
            })
        
        df = pd.DataFrame(emotion_data)
        df.to_csv(f"{session_dir}/live_log.csv", index=False)
        
        # Save summary
        with open(f"{session_dir}/summary.txt", 'w') as f:
            f.write(f"MobileNetV2 Live Session Summary\n")
            f.write(f"Session: {timestamp}\n")
            f.write(f"Duration: {self.monitoring_duration} seconds\n")
            f.write(f"Total Frames: {self.frame_count}\n")
            f.write(f"Emotion Samples: {len(self.emotion_buffer)}\n")
            f.write(f"Dominant Emotion: {self.dominant_emotion}\n")
            f.write(f"Average Confidence: {self.avg_confidence:.4f}\n")
        
        print(f"Session data saved to {session_dir}")
    
    def generate_gradcam(self, frame):
        """Generate Grad-CAM visualization for the current frame."""
        try:
            print("Generating Grad-CAM visualization...")
            
            # Create gradcam_live directory if it doesn't exist
            gradcam_dir = "results/gradcam_live"
            os.makedirs(gradcam_dir, exist_ok=True)
            
            # Preprocess the frame for the model
            processed_frame = preprocess_image_for_model(frame, 'mobilenetv2')
            
            # Generate Grad-CAM heatmap
            heatmap = grad_cam(self.model, processed_frame, 'block_16_project_BN')
            
            # Create overlay
            overlay = overlay_heatmap(heatmap, frame, alpha=0.4)
            
            # Save the overlay
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{gradcam_dir}/gradcam_frame_{timestamp}.png"
            cv2.imwrite(output_path, overlay)
            
            print(f"Grad-CAM visualization saved to: {output_path}")
            
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
    
    def run(self):
        """Run the live emotion detection system."""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera initialized")
        print("Starting 1-minute emotion detection...")
        print("Monitoring for exactly 60 seconds...")
        
        # Start monitoring
        self.monitoring_start_time = time.time()
        clock = pygame.time.Clock()
        running = True
        
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_g:  # Press 'g' for Grad-CAM
                            self.generate_gradcam(frame)
                
                # Camera processing
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Detect emotions
                frame, emotions = self.detect_emotions(frame)
                
                # Update emotion buffer
                self.update_emotion_buffer(emotions)
                
                # Calculate elapsed time
                elapsed_time = time.time() - self.monitoring_start_time
                remaining_time = self.monitoring_duration - elapsed_time
                
                if self.current_screen == 'monitoring':
                    # Draw monitoring screen with live camera
                    self.screen.fill(self.colors['background'])
                    
                    # Title
                    title_surface = self.fonts['title'].render("MobileNetV2 Live Detection", True, self.colors['text'])
                    self.screen.blit(title_surface, (50, 50))
                    
                    # Live camera feed (left side)
                    camera_width, camera_height = 400, 300
                    camera_frame_resized = cv2.resize(frame, (camera_width, camera_height))
                    
                    # Convert BGR to RGB for pygame
                    camera_frame_rgb = cv2.cvtColor(camera_frame_resized, cv2.COLOR_BGR2RGB)
                    camera_surface = pygame.surfarray.make_surface(camera_frame_rgb.swapaxes(0, 1))
                    
                    # Draw camera frame with border
                    camera_x = 50
                    camera_y = 120
                    self.screen.blit(camera_surface, (camera_x, camera_y))
                    pygame.draw.rect(self.screen, self.colors['border'], (camera_x, camera_y, camera_width, camera_height), 3, border_radius=8)
                    
                    # Live detection label
                    live_text = "LIVE DETECTION"
                    live_surface = self.fonts['body'].render(live_text, True, self.colors['danger'])
                    self.screen.blit(live_surface, (camera_x + 10, camera_y + 10))
                    
                    # Current emotion on camera
                    emotion_text = f"Emotion: {self.current_emotion.title()}"
                    emotion_surface = self.fonts['body'].render(emotion_text, True, self.colors['accent'])
                    self.screen.blit(emotion_surface, (camera_x + 10, camera_y + camera_height - 30))
                    
                    # Confidence on camera
                    confidence_text = f"Confidence: {self.emotion_confidence:.2f}"
                    confidence_surface = self.fonts['body'].render(confidence_text, True, self.colors['text'])
                    self.screen.blit(confidence_surface, (camera_x + 10, camera_y + camera_height - 60))
                    
                    # Status panel (right side)
                    status_x = camera_x + camera_width + 30
                    status_y = camera_y
                    
                    # Progress bar
                    progress = elapsed_time / self.monitoring_duration
                    pygame.draw.rect(self.screen, self.colors['secondary'], (status_x, status_y, 300, 30), border_radius=15)
                    pygame.draw.rect(self.screen, self.colors['primary'], (status_x, status_y, int(300 * progress), 30), border_radius=15)
                    
                    # Time remaining
                    time_text = f"Time Remaining: {remaining_time:.1f}s"
                    time_surface = self.fonts['subtitle'].render(time_text, True, self.colors['text'])
                    self.screen.blit(time_surface, (status_x, status_y + 50))
                    
                    # Frame count
                    frame_text = f"Frames Processed: {self.frame_count}"
                    frame_surface = self.fonts['body'].render(frame_text, True, self.colors['text_secondary'])
                    self.screen.blit(frame_surface, (status_x, status_y + 90))
                    
                    # Emotion buffer info
                    buffer_text = f"Emotion Samples: {len(self.emotion_buffer)}"
                    buffer_surface = self.fonts['body'].render(buffer_text, True, self.colors['text_secondary'])
                    self.screen.blit(buffer_surface, (status_x, status_y + 120))
                    
                    # Check if 1 minute is up
                    if elapsed_time >= self.monitoring_duration:
                        print("1 minute complete! Analyzing emotions...")
                        # Analyze emotions
                        if self.emotion_buffer:
                            self.dominant_emotion, self.avg_confidence = self.analyze_emotions()
                            self.recommendations = self.personalized_content[self.dominant_emotion]
                            self.current_screen = 'analysis'
                            self.save_session_data()
                        else:
                            print("No emotion data collected. Please try again.")
                            running = False
                
                elif self.current_screen == 'analysis':
                    # Show analysis results
                    self.screen.fill(self.colors['background'])
                    
                    # Title
                    title_surface = self.fonts['title'].render("Emotion Analysis Complete", True, self.colors['text'])
                    self.screen.blit(title_surface, (50, 50))
                    
                    # Analysis results
                    analysis_y = 120
                    pygame.draw.rect(self.screen, self.colors['card'], (50, analysis_y, 500, 200), border_radius=12)
                    pygame.draw.rect(self.screen, self.colors['border'], (50, analysis_y, 500, 200), 2, border_radius=12)
                    
                    # Dominant emotion
                    emotion_text = f"Dominant Emotion: {self.dominant_emotion.title()}"
                    emotion_surface = self.fonts['large'].render(emotion_text, True, self.colors['accent'])
                    self.screen.blit(emotion_surface, (70, analysis_y + 20))
                    
                    # Confidence
                    confidence_text = f"Confidence: {self.avg_confidence:.2f}"
                    confidence_surface = self.fonts['body'].render(confidence_text, True, self.colors['text_secondary'])
                    self.screen.blit(confidence_surface, (70, analysis_y + 60))
                    
                    # Samples
                    samples_text = f"Samples Analyzed: {len(self.emotion_buffer)}"
                    samples_surface = self.fonts['body'].render(samples_text, True, self.colors['text_secondary'])
                    self.screen.blit(samples_surface, (70, analysis_y + 90))
                    
                    # Continue button
                    continue_rect = pygame.Rect(200, analysis_y + 130, 200, 50)
                    pygame.draw.rect(self.screen, self.colors['primary'], continue_rect, border_radius=8)
                    pygame.draw.rect(self.screen, self.colors['border'], continue_rect, 2, border_radius=8)
                    continue_text = self.fonts['body'].render("View Recommendations", True, self.colors['text'])
                    continue_rect_text = continue_text.get_rect(center=continue_rect.center)
                    self.screen.blit(continue_text, continue_rect_text)
                    
                    # Check for continue button click
                    for event in pygame.event.get():
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if continue_rect.collidepoint(event.pos):
                                self.current_screen = 'recommendations'
                
                elif self.current_screen == 'recommendations':
                    # Show recommendations
                    self.screen.fill(self.colors['background'])
                    
                    # Title
                    title_surface = self.fonts['title'].render("Personalized Recommendations", True, self.colors['text'])
                    self.screen.blit(title_surface, (50, 30))
                    
                    # Emotion-specific message
                    if self.recommendations:
                        message = self.recommendations['title']
                        message_surface = self.fonts['subtitle'].render(message, True, self.colors['accent'])
                        self.screen.blit(message_surface, (50, 80))
                    
                    # Content cards
                    card_width = 300
                    card_height = 250
                    card_spacing = 20
                    start_x = 50
                    start_y = 130
                    
                    if self.recommendations:
                        # Videos card
                        pygame.draw.rect(self.screen, self.colors['card'], (start_x, start_y, card_width, card_height), border_radius=12)
                        pygame.draw.rect(self.screen, self.colors['border'], (start_x, start_y, card_width, card_height), 2, border_radius=12)
                        
                        # Card title
                        title_surface = self.fonts['subtitle'].render("Recommended Videos", True, self.colors['text'])
                        self.screen.blit(title_surface, (start_x + 20, start_y + 20))
                        
                        # Card content
                        y_offset = start_y + 60
                        for i, video in enumerate(self.recommendations['content']['videos'][:3]):
                            if y_offset < start_y + card_height - 20:
                                video_text = f"• {video}"
                                video_surface = self.fonts['body'].render(video_text, True, self.colors['text_secondary'])
                                self.screen.blit(video_surface, (start_x + 30, y_offset))
                                y_offset += 30
                        
                        # Music card
                        pygame.draw.rect(self.screen, self.colors['card'], (start_x + card_width + card_spacing, start_y, card_width, card_height), border_radius=12)
                        pygame.draw.rect(self.screen, self.colors['border'], (start_x + card_width + card_spacing, start_y, card_width, card_height), 2, border_radius=12)
                        
                        # Card title
                        title_surface = self.fonts['subtitle'].render("Recommended Music", True, self.colors['text'])
                        self.screen.blit(title_surface, (start_x + card_width + card_spacing + 20, start_y + 20))
                        
                        # Card content
                        y_offset = start_y + 60
                        for music in self.recommendations['content']['music']:
                            if y_offset < start_y + card_height - 20:
                                music_text = f"• {music}"
                                music_surface = self.fonts['body'].render(music_text, True, self.colors['text_secondary'])
                                self.screen.blit(music_surface, (start_x + card_width + card_spacing + 30, y_offset))
                                y_offset += 30
                        
                        # Activities card
                        pygame.draw.rect(self.screen, self.colors['card'], (start_x, start_y + card_height + card_spacing, card_width, card_height), border_radius=12)
                        pygame.draw.rect(self.screen, self.colors['border'], (start_x, start_y + card_height + card_spacing, card_width, card_height), 2, border_radius=12)
                        
                        # Card title
                        title_surface = self.fonts['subtitle'].render("Suggested Activities", True, self.colors['text'])
                        self.screen.blit(title_surface, (start_x + 20, start_y + card_height + card_spacing + 20))
                        
                        # Card content
                        y_offset = start_y + card_height + card_spacing + 60
                        for activity in self.recommendations['content']['activities'][:4]:
                            if y_offset < start_y + card_height + card_spacing + card_height - 20:
                                activity_text = f"• {activity}"
                                activity_surface = self.fonts['body'].render(activity_text, True, self.colors['text_secondary'])
                                self.screen.blit(activity_surface, (start_x + 30, y_offset))
                                y_offset += 30
                    
                    # Action buttons
                    button_y = start_y + card_height + card_spacing + card_height + 20
                    button_width = 180
                    button_height = 50
                    button_spacing = 15
                    
                    # Watch video button
                    video_rect = pygame.Rect(start_x, button_y, button_width, button_height)
                    pygame.draw.rect(self.screen, self.colors['primary'], video_rect, border_radius=8)
                    pygame.draw.rect(self.screen, self.colors['border'], video_rect, 2, border_radius=8)
                    video_text = self.fonts['body'].render("Watch Video", True, self.colors['text'])
                    video_rect_text = video_text.get_rect(center=video_rect.center)
                    self.screen.blit(video_text, video_rect_text)
                    
                    # Listen to music button
                    music_rect = pygame.Rect(start_x + button_width + button_spacing, button_y, button_width, button_height)
                    pygame.draw.rect(self.screen, self.colors['success'], music_rect, border_radius=8)
                    pygame.draw.rect(self.screen, self.colors['border'], music_rect, 2, border_radius=8)
                    music_text = self.fonts['body'].render("Listen to Music", True, self.colors['text'])
                    music_rect_text = music_text.get_rect(center=music_rect.center)
                    self.screen.blit(music_text, music_rect_text)
                    
                    # Try activity button
                    activity_rect = pygame.Rect(start_x + 2 * (button_width + button_spacing), button_y, button_width, button_height)
                    pygame.draw.rect(self.screen, self.colors['warning'], activity_rect, border_radius=8)
                    pygame.draw.rect(self.screen, self.colors['border'], activity_rect, 2, border_radius=8)
                    activity_text = self.fonts['body'].render("Try Activity", True, self.colors['text'])
                    activity_rect_text = activity_text.get_rect(center=activity_rect.center)
                    self.screen.blit(activity_text, activity_rect_text)
                    
                    # Exit button
                    exit_rect = pygame.Rect(start_x + 3 * (button_width + button_spacing), button_y, button_width, button_height)
                    pygame.draw.rect(self.screen, self.colors['danger'], exit_rect, border_radius=8)
                    pygame.draw.rect(self.screen, self.colors['border'], exit_rect, 2, border_radius=8)
                    exit_text = self.fonts['body'].render("Exit", True, self.colors['text'])
                    exit_rect_text = exit_text.get_rect(center=exit_rect.center)
                    self.screen.blit(exit_text, exit_rect_text)
                    
                    # Check for button clicks
                    for event in pygame.event.get():
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if video_rect.collidepoint(event.pos):
                                # Show video options
                                print("\nChoose a video:")
                                for i, video in enumerate(self.recommendations['content']['videos'], 1):
                                    print(f"  {i}. {video}")
                                try:
                                    choice = input("Enter video number (1-3): ").strip()
                                    if choice.isdigit() and 1 <= int(choice) <= 3:
                                        video_url = self.recommendations['content']['videos'][int(choice)-1]
                                        print(f"Opening video: {video_url}")
                                        webbrowser.open(video_url)
                                except:
                                    pass
                            elif music_rect.collidepoint(event.pos):
                                print("\nMusic Recommendations:")
                                for music in self.recommendations['content']['music']:
                                    print(f"  • {music}")
                                print("\nSearch for these on your favorite music platform!")
                            elif activity_rect.collidepoint(event.pos):
                                print("\nActivity Recommendations:")
                                for activity in self.recommendations['content']['activities']:
                                    print(f"  • {activity}")
                                print("\nTry any of these activities to help improve your mood!")
                            elif exit_rect.collidepoint(event.pos):
                                running = False
                
                # Update display
                pygame.display.flip()
                clock.tick(30)  # 30 FPS
                    
        except KeyboardInterrupt:
            print("Stopped by user")
        
        finally:
            cap.release()
            pygame.quit()
            print("System stopped")

def main():
    """Main function."""
    system = LiveMobileNetV2System()
    if hasattr(system, 'emotion_model'):
        system.run()
    else:
        print("System initialization failed")

if __name__ == "__main__":
    main()
