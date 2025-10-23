#!/usr/bin/env python3
"""
Improved Live Mini-XCEPTION Emotion Detection with Better UI/UX
=============================================================

Real-time emotion detection using Mini-XCEPTION model with modern, clean interface.
Monitors for 1 minute, then provides personalized recommendations.

Usage:
    python live_improved.py
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

class ImprovedLiveMiniXceptionSystem:
    """Improved live emotion detection with modern UI/UX."""
    
    def __init__(self):
        print("Loading Improved Live Mini-XCEPTION System")
        print("=" * 50)
        
        # Initialize pygame for GUI
        pygame.init()
        
        # Set up display with better resolution
        self.screen_width = 1400
        self.screen_height = 900
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Live Emotion Detection & Personalized Recommendations")
        
        # Modern color scheme
        self.colors = {
            'background': (15, 15, 25),      # Dark navy
            'surface': (25, 25, 40),         # Darker surface
            'primary': (0, 150, 255),        # Blue
            'secondary': (100, 200, 100),    # Green
            'accent': (255, 200, 50),        # Gold
            'text': (255, 255, 255),         # White
            'text_secondary': (180, 180, 180), # Light gray
            'success': (50, 200, 50),        # Green
            'warning': (255, 150, 50),       # Orange
            'error': (255, 50, 50),          # Red
            'border': (60, 60, 80)           # Dark border
        }
        
        # Modern fonts
        self.fonts = {
            'title': pygame.font.Font(None, 48),
            'heading': pygame.font.Font(None, 36),
            'subheading': pygame.font.Font(None, 28),
            'body': pygame.font.Font(None, 24),
            'small': pygame.font.Font(None, 20),
            'large': pygame.font.Font(None, 32)
        }
        
        # Load model
        print("Loading Mini-XCEPTION model...")
        self.model = load_model('best.h5')
        print("Mini-XCEPTION model loaded (68.6% accuracy)")
        
        # Load face detection
        self.face_cascade = cv2.CascadeClassifier('../../haarcascade_files/haarcascade_frontalface_default.xml')
        
        # Load personalized content
        self.personalized_content = get_personalized_content()
        
        # Emotion detection state
        self.emotion_buffer = deque(maxlen=300)  # 5 minutes at 1 FPS
        self.emotion_labels = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']
        self.current_emotion = 'neutral'
        self.monitoring_duration = 60  # 1 minute
        self.frame_count = 0
        
        # UI State
        self.current_screen = 'monitoring'  # monitoring, analysis, recommendations
        self.recommendations = None
        self.dominant_emotion = 'neutral'
        self.avg_confidence = 0.0
        self.buttons = []
        self.gradcam_generated = False
        self.gradcam_timer = 0
        
        print("System ready!")
    
    def draw_modern_card(self, x, y, width, height, title, content, color=None):
        """Draw a modern card with shadow and rounded corners."""
        if color is None:
            color = self.colors['surface']
        
        # Shadow
        shadow_offset = 4
        pygame.draw.rect(self.screen, (0, 0, 0, 50), 
                        (x + shadow_offset, y + shadow_offset, width, height), 
                        border_radius=12)
        
        # Card background
        pygame.draw.rect(self.screen, color, (x, y, width, height), border_radius=12)
        pygame.draw.rect(self.screen, self.colors['border'], (x, y, width, height), 2, border_radius=12)
        
        # Title
        title_surface = self.fonts['subheading'].render(title, True, self.colors['text'])
        self.screen.blit(title_surface, (x + 20, y + 15))
        
        # Content
        y_offset = y + 50
        for item in content:
            if y_offset < y + height - 20:
                item_surface = self.fonts['body'].render(f"• {item}", True, self.colors['text_secondary'])
                self.screen.blit(item_surface, (x + 20, y_offset))
                y_offset += 30
    
    def draw_modern_button(self, x, y, width, height, text, color, hover_color=None):
        """Draw a modern button with hover effect."""
        if hover_color is None:
            hover_color = tuple(min(255, c + 30) for c in color)
        
        # Get mouse position for hover effect
        mouse_x, mouse_y = pygame.mouse.get_pos()
        is_hover = x <= mouse_x <= x + width and y <= mouse_y <= y + height
        
        button_color = hover_color if is_hover else color
        
        # Button background
        pygame.draw.rect(self.screen, button_color, (x, y, width, height), border_radius=8)
        pygame.draw.rect(self.screen, self.colors['border'], (x, y, width, height), 2, border_radius=8)
        
        # Button text
        text_surface = self.fonts['body'].render(text, True, self.colors['text'])
        text_rect = text_surface.get_rect(center=(x + width//2, y + height//2))
        self.screen.blit(text_surface, text_rect)
        
        return pygame.Rect(x, y, width, height)
    
    def draw_progress_bar(self, x, y, width, height, progress, color):
        """Draw a modern progress bar."""
        # Background
        pygame.draw.rect(self.screen, self.colors['border'], (x, y, width, height), border_radius=height//2)
        
        # Progress fill
        fill_width = int(width * progress)
        if fill_width > 0:
            pygame.draw.rect(self.screen, color, (x, y, fill_width, height), border_radius=height//2)
        
        # Progress text
        progress_text = f"{int(progress * 100)}%"
        text_surface = self.fonts['small'].render(progress_text, True, self.colors['text'])
        text_rect = text_surface.get_rect(center=(x + width//2, y + height//2))
        self.screen.blit(text_surface, text_rect)
    
    def draw_emotion_analysis(self):
        """Draw emotion analysis screen."""
        self.screen.fill(self.colors['background'])
        
        # Header
        header_y = 50
        title_surface = self.fonts['title'].render("Emotion Analysis Complete", True, self.colors['text'])
        self.screen.blit(title_surface, (self.screen_width//2 - title_surface.get_width()//2, header_y))
        
        # Analysis results
        analysis_y = 150
        analysis_width = 600
        analysis_height = 200
        
        # Analysis card
        pygame.draw.rect(self.screen, self.colors['surface'], 
                       (self.screen_width//2 - analysis_width//2, analysis_y, analysis_width, analysis_height), 
                       border_radius=16)
        pygame.draw.rect(self.screen, self.colors['border'], 
                       (self.screen_width//2 - analysis_width//2, analysis_y, analysis_width, analysis_height), 
                       2, border_radius=16)
        
        # Dominant emotion
        emotion_text = f"Detected Emotion: {self.dominant_emotion.title()}"
        emotion_surface = self.fonts['heading'].render(emotion_text, True, self.colors['accent'])
        self.screen.blit(emotion_surface, (self.screen_width//2 - emotion_surface.get_width()//2, analysis_y + 30))
        
        # Confidence
        confidence_text = f"Confidence: {self.avg_confidence:.1%}"
        confidence_surface = self.fonts['subheading'].render(confidence_text, True, self.colors['text_secondary'])
        self.screen.blit(confidence_surface, (self.screen_width//2 - confidence_surface.get_width()//2, analysis_y + 80))
        
        # Continue button
        continue_rect = self.draw_modern_button(
            self.screen_width//2 - 100, analysis_y + 140, 200, 40, 
            "View Recommendations", self.colors['primary']
        )
        
        # Check for continue button click
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if continue_rect.collidepoint(event.pos):
                    self.current_screen = 'recommendations'
    
    def draw_recommendations(self):
        """Draw personalized recommendations screen with modern UI."""
        self.screen.fill(self.colors['background'])
        
        # Header with emotion context
        header_y = 30
        title_surface = self.fonts['title'].render("Personalized Recommendations", True, self.colors['text'])
        self.screen.blit(title_surface, (self.screen_width//2 - title_surface.get_width()//2, header_y))
        
        # Emotion context
        emotion_text = f"Based on your {self.dominant_emotion.title()} emotion"
        emotion_surface = self.fonts['subheading'].render(emotion_text, True, self.colors['accent'])
        self.screen.blit(emotion_surface, (self.screen_width//2 - emotion_surface.get_width()//2, header_y + 50))
        
        # Subtitle
        subtitle = self.recommendations['title']
        subtitle_surface = self.fonts['body'].render(subtitle, True, self.colors['text_secondary'])
        self.screen.blit(subtitle_surface, (self.screen_width//2 - subtitle_surface.get_width()//2, header_y + 85))
        
        # Main content area
        content_y = 150
        content_width = 1200
        content_height = 500
        
        # Background for content
        pygame.draw.rect(self.screen, self.colors['surface'], 
                       (self.screen_width//2 - content_width//2, content_y, content_width, content_height), 
                       border_radius=16)
        pygame.draw.rect(self.screen, self.colors['border'], 
                       (self.screen_width//2 - content_width//2, content_y, content_width, content_height), 
                       2, border_radius=16)
        
        # Three columns
        col_width = 350
        col_spacing = 50
        start_x = self.screen_width//2 - (col_width * 3 + col_spacing * 2) // 2
        col_y = content_y + 30
        
        # Videos column
        self.draw_recommendation_column(
            start_x, col_y, col_width, "Videos", 
            self.recommendations['content']['videos'][:3],
            self.colors['primary']
        )
        
        # Music column  
        self.draw_recommendation_column(
            start_x + col_width + col_spacing, col_y, col_width, "Music", 
            self.recommendations['content']['music'][:4],
            self.colors['secondary']
        )
        
        # Activities column
        self.draw_recommendation_column(
            start_x + (col_width + col_spacing) * 2, col_y, col_width, "Activities", 
            self.recommendations['content']['activities'][:5],
            self.colors['accent']
        )
        
        # Action buttons at bottom
        button_y = content_y + content_height + 30
        button_width = 200
        button_height = 60
        button_spacing = 30
        
        # Center buttons
        total_button_width = button_width * 4 + button_spacing * 3
        button_start_x = self.screen_width//2 - total_button_width//2
        
        # Video button
        video_rect = self.draw_modern_button(
            button_start_x, button_y, button_width, button_height,
            "🎬 Watch Video", self.colors['primary']
        )
        
        # Music button
        music_rect = self.draw_modern_button(
            button_start_x + button_width + button_spacing, button_y, button_width, button_height,
            "🎵 Listen Music", self.colors['secondary']
        )
        
        # Activity button
        activity_rect = self.draw_modern_button(
            button_start_x + (button_width + button_spacing) * 2, button_y, button_width, button_height,
            "🎯 Try Activity", self.colors['accent']
        )
        
        # Exit button
        exit_rect = self.draw_modern_button(
            button_start_x + (button_width + button_spacing) * 3, button_y, button_width, button_height,
            "❌ Exit", self.colors['error']
        )
        
        # Handle button clicks
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if video_rect.collidepoint(event.pos):
                    self.show_video_selection()
                
                elif music_rect.collidepoint(event.pos):
                    self.show_music_selection()
                
                elif activity_rect.collidepoint(event.pos):
                    self.show_activity_selection()
                
                elif exit_rect.collidepoint(event.pos):
                    return False
        
        return True
    
    def draw_recommendation_column(self, x, y, width, title, items, color):
        """Draw a recommendation column with modern styling."""
        # Column background
        pygame.draw.rect(self.screen, self.colors['background'], (x, y, width, 400), border_radius=12)
        pygame.draw.rect(self.screen, color, (x, y, width, 400), 2, border_radius=12)
        
        # Title
        title_surface = self.fonts['subheading'].render(title, True, color)
        self.screen.blit(title_surface, (x + 20, y + 20))
        
        # Items
        item_y = y + 60
        for i, item in enumerate(items):
            if item_y < y + 380:  # Don't overflow
                # Clean up item text (remove URLs, make it readable)
                clean_item = self.clean_recommendation_text(item)
                
                # Truncate if too long
                if len(clean_item) > 40:
                    clean_item = clean_item[:37] + "..."
                
                item_surface = self.fonts['body'].render(f"• {clean_item}", True, self.colors['text'])
                self.screen.blit(item_surface, (x + 20, item_y))
                item_y += 35
    
    def clean_recommendation_text(self, text):
        """Clean up recommendation text to make it more readable."""
        # Remove URLs and make text more readable
        if "youtube.com" in text.lower() or "watch?v=" in text.lower():
            # Extract video title from URL or use generic title
            if "watch?v=" in text:
                return "Recommended Video"
            return "YouTube Video"
        elif "spotify.com" in text.lower():
            return "Spotify Playlist"
        else:
            return text
    
    def show_video_selection(self):
        """Show video selection dialog."""
        print("\n🎬 Choose a video to watch:")
        for i, video in enumerate(self.recommendations['content']['videos'], 1):
            clean_title = self.clean_recommendation_text(video)
            print(f"  {i}. {clean_title}")
        print("  (Videos will open in your browser)")
        
        # Open first video as default
        if self.recommendations['content']['videos']:
            webbrowser.open(self.recommendations['content']['videos'][0])
    
    def show_music_selection(self):
        """Show music selection dialog."""
        print("\n🎵 Recommended Music:")
        for i, song in enumerate(self.recommendations['content']['music'], 1):
            print(f"  {i}. {song}")
        print("  (Search these songs on your music app)")
    
    def show_activity_selection(self):
        """Show activity selection dialog."""
        print("\n🎯 Suggested Activities:")
        for i, activity in enumerate(self.recommendations['content']['activities'], 1):
            print(f"  {i}. {activity}")
        print("  (Try any of these activities to improve your mood)")
    
    def update_emotion_buffer(self, emotions):
        """Update emotion buffer with new data."""
        if emotions:
            # Use the most confident emotion
            best_emotion = max(emotions, key=lambda x: x['confidence'])
            self.emotion_buffer.append(best_emotion)
            
            # Debug: Print emotion being added
            print(f"Adding emotion: {best_emotion['emotion']} (confidence: {best_emotion['confidence']:.3f})")
            
            # Update current emotion if confidence is high enough
            if best_emotion['confidence'] > 0.3:  # Lowered threshold from 0.5 to 0.3
                self.current_emotion = best_emotion['emotion']
    
    def analyze_emotions(self):
        """Analyze collected emotions to find dominant emotion."""
        if not self.emotion_buffer:
            return 'neutral', 0.0
        
        # Count all emotions (including neutral for debugging)
        all_emotion_counts = {}
        meaningful_emotion_counts = {}
        meaningful_emotions = []
        
        for emotion_data in self.emotion_buffer:
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            
            # Count all emotions
            all_emotion_counts[emotion] = all_emotion_counts.get(emotion, 0) + 1
            
            # Only consider meaningful emotions (not neutral)
            if emotion != 'neutral':
                meaningful_emotions.append(emotion_data)
                meaningful_emotion_counts[emotion] = meaningful_emotion_counts.get(emotion, 0) + 1
        
        print(f"All emotion counts: {all_emotion_counts}")
        print(f"Meaningful emotion counts: {meaningful_emotion_counts}")
        
        # If we have meaningful emotions, use the most frequent one
        if meaningful_emotion_counts:
            # Find the most frequent meaningful emotion
            dominant_emotion = max(meaningful_emotion_counts, key=meaningful_emotion_counts.get)
            
            # Calculate average confidence for this emotion
            emotion_confidences = [e['confidence'] for e in meaningful_emotions if e['emotion'] == dominant_emotion]
            avg_confidence = sum(emotion_confidences) / len(emotion_confidences) if emotion_confidences else 0.0
            
            print(f"Selected dominant emotion: {dominant_emotion} (count: {meaningful_emotion_counts[dominant_emotion]}, avg confidence: {avg_confidence:.3f})")
            return dominant_emotion, avg_confidence
        else:
            # Fallback to neutral if no meaningful emotions detected
            print("No meaningful emotions detected, using neutral")
            return 'neutral', 0.0
    
    def save_session_data(self):
        """Save live session data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = f"results/live_sessions/sess_{timestamp}"
        os.makedirs(session_dir, exist_ok=True)
        
        # Save emotion data
        emotion_data = []
        for emotion_info in self.emotion_buffer:
            emotion_data.append({
                'emotion': emotion_info['emotion'],
                'confidence': emotion_info['confidence'],
                'timestamp': emotion_info.get('timestamp', '')
            })
        
        df = pd.DataFrame(emotion_data)
        df.to_csv(f"{session_dir}/live_log.csv", index=False)
        
        # Save summary
        with open(f"{session_dir}/summary.txt", "w") as f:
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
            processed_frame = preprocess_image_for_model(frame, 'mini_xception')
            
            # Generate Grad-CAM heatmap
            heatmap = grad_cam(self.model, processed_frame, 'conv2d_2')
            
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
        """Run the improved live emotion detection system."""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera initialized")
        print("Starting 1-minute emotion detection...")
        
        # Main loop
        clock = pygame.time.Clock()
        running = True
        start_time = time.time()
        
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_g:  # Press 'g' for Grad-CAM
                            print("Generating Grad-CAM...")
                            self.generate_gradcam(frame)
                            self.gradcam_generated = True
                            self.gradcam_timer = time.time()
                            print("Grad-CAM generated! Check results/gradcam_live/ folder")
                
                # Camera processing
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                if self.current_screen == 'monitoring':
                    # Face detection and emotion recognition
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    emotions = []
                    for (x, y, w, h) in faces:
                        # Extract face region
                        face_roi = gray[y:y+h, x:x+w]
                        face_resized = cv2.resize(face_roi, (64, 64))
                        face_normalized = face_resized.astype('float32') / 255.0
                        face_input = np.expand_dims(face_normalized, axis=-1)
                        face_input = np.expand_dims(face_input, axis=0)
                        
                        # Predict emotion
                        predictions = self.model.predict(face_input, verbose=0)
                        predicted_class = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_class]
                        emotion = self.emotion_labels[predicted_class]
                        
                        emotions.append({
                            'emotion': emotion,
                            'confidence': confidence,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Draw face rectangle and emotion
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{emotion} ({confidence:.2f})", 
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Update emotion buffer
                    self.update_emotion_buffer(emotions)
                    
                    # Draw monitoring screen
                    self.screen.fill(self.colors['background'])
                    
                    # Header
                    title_surface = self.fonts['title'].render("Live Emotion Detection", True, self.colors['text'])
                    self.screen.blit(title_surface, (self.screen_width//2 - title_surface.get_width()//2, 30))
                    
                    # Progress bar
                    elapsed_time = time.time() - start_time
                    progress = min(elapsed_time / self.monitoring_duration, 1.0)
                    self.draw_progress_bar(100, 100, 1200, 30, progress, self.colors['primary'])
                    
                    # Status
                    status_text = f"Monitoring... {int(elapsed_time)}s / {self.monitoring_duration}s"
                    status_surface = self.fonts['subheading'].render(status_text, True, self.colors['text_secondary'])
                    self.screen.blit(status_surface, (self.screen_width//2 - status_surface.get_width()//2, 150))
                    
                    # Current emotion
                    if self.current_emotion != 'neutral':
                        emotion_text = f"Current: {self.current_emotion.title()}"
                        emotion_surface = self.fonts['heading'].render(emotion_text, True, self.colors['accent'])
                        self.screen.blit(emotion_surface, (self.screen_width//2 - emotion_surface.get_width()//2, 200))
                    
                    # Instructions
                    instruction_text = "Press 'G' for Grad-CAM visualization"
                    instruction_surface = self.fonts['body'].render(instruction_text, True, self.colors['text_secondary'])
                    self.screen.blit(instruction_surface, (self.screen_width//2 - instruction_surface.get_width()//2, 250))
                    
                    # Grad-CAM feedback
                    if self.gradcam_generated:
                        current_time = time.time()
                        if current_time - self.gradcam_timer < 3:  # Show for 3 seconds
                            feedback_text = "Grad-CAM Generated! Check results/gradcam_live/ folder"
                            feedback_surface = self.fonts['body'].render(feedback_text, True, self.colors['success'])
                            self.screen.blit(feedback_surface, (self.screen_width//2 - feedback_surface.get_width()//2, 280))
                        else:
                            self.gradcam_generated = False
                    
                    # Camera feed (small preview)
                    camera_surface = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0,1))
                    camera_surface = pygame.transform.scale(camera_surface, (400, 300))
                    self.screen.blit(camera_surface, (self.screen_width//2 - 200, 300))
                    
                    # Check if monitoring is complete
                    if elapsed_time >= self.monitoring_duration:
                        print("1 minute complete! Analyzing emotions...")
                        print(f"Emotion buffer size: {len(self.emotion_buffer)}")
                        
                        # Analyze emotions
                        if self.emotion_buffer:
                            self.dominant_emotion, self.avg_confidence = self.analyze_emotions()
                            print(f"Analyzed dominant emotion: {self.dominant_emotion} (confidence: {self.avg_confidence:.3f})")
                            
                            # Debug info is now handled in analyze_emotions function
                            
                            self.recommendations = self.personalized_content[self.dominant_emotion]
                            self.current_screen = 'analysis'
                            self.save_session_data()
                        else:
                            print("No emotions detected, defaulting to neutral")
                            self.dominant_emotion = 'neutral'
                            self.avg_confidence = 0.0
                            self.recommendations = self.personalized_content['neutral']
                            self.current_screen = 'analysis'
                
                elif self.current_screen == 'analysis':
                    self.draw_emotion_analysis()
                
                elif self.current_screen == 'recommendations':
                    running = self.draw_recommendations()
                
                pygame.display.flip()
                clock.tick(30)
        
        except KeyboardInterrupt:
            print("\nSystem stopped by user")
        
        finally:
            cap.release()
            pygame.quit()
            print("System stopped")

def main():
    """Main function."""
    system = ImprovedLiveMiniXceptionSystem()
    system.run()

if __name__ == "__main__":
    main()
