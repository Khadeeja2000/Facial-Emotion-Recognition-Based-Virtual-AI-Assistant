#!/usr/bin/env python3
"""
FER2013 Emotion Detection with Personalized Content System
==========================================================

Real-time emotion detection with personalized content recommendations.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pygame
import webbrowser
from collections import deque
import time
import random
import math

class EmotionPersonalizedSystem:
    """FER2013 emotion detection with personalized content recommendations."""
    
    def __init__(self):
        print("🎭 LOADING EMOTION PERSONALIZED SYSTEM")
        print("=" * 60)
        
        # Initialize pygame for GUI
        pygame.init()
        
        # Set up display
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Emotion Detection & Personalized Recommendations")
        
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
        
        # UI State
        self.current_screen = 'monitoring'  # monitoring, recommendations, choice
        self.recommendations = None
        self.dominant_emotion = 'neutral'
        self.avg_confidence = 0.0
        self.selected_option = 0
        self.buttons = []
        
        # Load FER2013 emotion model
        try:
            self.emotion_model = load_model('models/_mini_XCEPTION.102-0.66.hdf5')
            print("✅ FER2013 emotion model loaded (59.9% accuracy)")
        except Exception as e:
            print(f"❌ Error loading FER2013 model: {e}")
            return
        
        # Load face cascade
        try:
            self.face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
            print("✅ Face detection loaded")
        except Exception as e:
            print(f"❌ Error loading face cascade: {e}")
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
        
        # Personalized content database
        self.personalized_content = {
            'happy': {
                'title': '🎉 You look happy! Keep the positive vibes!',
                'content': {
                    'videos': [
                        'https://www.youtube.com/watch?v=dQw4w9WgXcQ',  # Fun music
                        'https://www.youtube.com/watch?v=9bZkp7q19f0',  # Dance music
                        'https://www.youtube.com/watch?v=L_jWHffIx5E'   # Upbeat content
                    ],
                    'activities': [
                        'Share your happiness with friends',
                        'Try a new hobby or activity',
                        'Go for a walk in nature',
                        'Listen to your favorite music'
                    ],
                    'music': [
                        'Upbeat pop songs',
                        'Happy electronic music',
                        'Positive rock music'
                    ]
                }
            },
            'sad': {
                'title': '😢 I notice you might be feeling down. Here are some uplifting options:',
                'content': {
                    'videos': [
                        'https://www.youtube.com/watch?v=inpok4MKVLM',  # Calming meditation
                        'https://www.youtube.com/watch?v=ZToicYcHIOU',  # Guided relaxation
                        'https://www.youtube.com/watch?v=1ZYbU82GVz4'   # Gentle breathing
                    ],
                    'activities': [
                        'Take a warm bath or shower',
                        'Call a friend or family member',
                        'Write in a journal',
                        'Go for a gentle walk',
                        'Watch a favorite movie'
                    ],
                    'music': [
                        'Soft instrumental music',
                        'Gentle acoustic songs',
                        'Calming nature sounds'
                    ]
                }
            },
            'angry': {
                'title': '😠 I see you might be feeling frustrated. Let\'s help you relax:',
                'content': {
                    'videos': [
                        'https://www.youtube.com/watch?v=inpok4MKVLM',  # Deep breathing
                        'https://www.youtube.com/watch?v=ZToicYcHIOU',  # Stress relief
                        'https://www.youtube.com/watch?v=1ZYbU82GVz4'   # Anger management
                    ],
                    'activities': [
                        'Take 10 deep breaths',
                        'Go for a brisk walk',
                        'Listen to calming music',
                        'Try progressive muscle relaxation',
                        'Write down what\'s bothering you'
                    ],
                    'music': [
                        'Calming classical music',
                        'Soft instrumental music',
                        'Nature sounds'
                    ]
                }
            },
            'scared': {
                'title': '😰 I notice you might be feeling anxious. Here are some calming options:',
                'content': {
                    'videos': [
                        'https://www.youtube.com/watch?v=inpok4MKVLM',  # Anxiety relief
                        'https://www.youtube.com/watch?v=ZToicYcHIOU',  # Grounding techniques
                        'https://www.youtube.com/watch?v=1ZYbU82GVz4'   # Breathing exercises
                    ],
                    'activities': [
                        'Practice 4-7-8 breathing',
                        'Use grounding techniques (5-4-3-2-1)',
                        'Listen to calming music',
                        'Take a warm bath',
                        'Call a trusted friend'
                    ],
                    'music': [
                        'Soft classical music',
                        'Ambient sounds',
                        'Calming nature sounds'
                    ]
                }
            },
            'surprised': {
                'title': '😲 You look surprised! Here are some fun options:',
                'content': {
                    'videos': [
                        'https://www.youtube.com/watch?v=dQw4w9WgXcQ',  # Fun surprises
                        'https://www.youtube.com/watch?v=9bZkp7q19f0',  # Exciting content
                        'https://www.youtube.com/watch?v=L_jWHffIx5E'   # Positive surprises
                    ],
                    'activities': [
                        'Explore something new',
                        'Try a new recipe',
                        'Visit a new place',
                        'Learn something interesting'
                    ],
                    'music': [
                        'Upbeat and energetic music',
                        'Positive pop songs',
                        'Exciting instrumental music'
                    ]
                }
            },
            'disgust': {
                'title': '🤢 I notice you might be feeling uncomfortable. Here are some helpful options:',
                'content': {
                    'videos': [
                        'https://www.youtube.com/watch?v=inpok4MKVLM',  # Calming content
                        'https://www.youtube.com/watch?v=ZToicYcHIOU',  # Relaxation
                        'https://www.youtube.com/watch?v=1ZYbU82GVz4'   # Gentle breathing
                    ],
                    'activities': [
                        'Take a break and rest',
                        'Listen to calming music',
                        'Go to a peaceful place',
                        'Practice mindfulness',
                        'Talk to someone you trust'
                    ],
                    'music': [
                        'Soft instrumental music',
                        'Calming nature sounds',
                        'Gentle acoustic music'
                    ]
                }
            },
            'neutral': {
                'title': '😐 You look calm and neutral. Here are some balanced options:',
                'content': {
                    'videos': [
                        'https://www.youtube.com/watch?v=inpok4MKVLM',  # Balanced content
                        'https://www.youtube.com/watch?v=ZToicYcHIOU',  # Mindful activities
                        'https://www.youtube.com/watch?v=1ZYbU82GVz4'   # Gentle relaxation
                    ],
                    'activities': [
                        'Read a good book',
                        'Take a peaceful walk',
                        'Practice meditation',
                        'Enjoy a hobby',
                        'Spend time in nature'
                    ],
                    'music': [
                        'Soft instrumental music',
                        'Ambient sounds',
                        'Gentle classical music'
                    ]
                }
            }
        }
        
        print("✅ Personalized content database loaded")
        print("🚀 System ready!")
    
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
            face_processed = face_resized.astype('float32') / 255.0
            face_processed = np.expand_dims(face_processed, axis=0)
            face_processed = np.expand_dims(face_processed, axis=-1)
            
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
    
    def get_personalized_recommendations(self):
        """Get personalized content recommendations based on current emotion."""
        if self.current_emotion in self.personalized_content:
            return self.personalized_content[self.current_emotion]
        else:
            return self.personalized_content['neutral']
    
    def show_recommendations(self):
        """Show personalized recommendations."""
        recommendations = self.get_personalized_recommendations()
        
        print(f"\n🎯 PERSONALIZED RECOMMENDATIONS")
        print("=" * 50)
        print(f"📊 Detected Emotion: {self.current_emotion} (confidence: {self.emotion_confidence:.2f})")
        print(f"💡 {recommendations['title']}")
        
        print(f"\n📺 Recommended Videos:")
        for i, video in enumerate(recommendations['content']['videos'][:3], 1):
            print(f"  {i}. {video}")
        
        print(f"\n🎵 Recommended Music:")
        for music in recommendations['content']['music']:
            print(f"  • {music}")
        
        print(f"\n🎯 Suggested Activities:")
        for activity in recommendations['content']['activities'][:3]:
            print(f"  • {activity}")
        
        print(f"\n🌐 Would you like me to open a recommended video? (y/n)")
        return recommendations
    
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
    
    def show_final_recommendations(self, dominant_emotion, confidence):
        """Show final recommendations after 1-minute analysis."""
        print(f"\n🎯 1-MINUTE EMOTION ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"📊 Dominant Emotion: {dominant_emotion}")
        print(f"📈 Confidence: {confidence:.2f}")
        print(f"📝 Total Samples: {len(self.emotion_buffer)}")
        
        recommendations = self.personalized_content[dominant_emotion]
        
        print(f"\n💡 {recommendations['title']}")
        print(f"\n🎯 PERSONALIZED RECOMMENDATIONS:")
        print("=" * 50)
        
        # Show videos
        print(f"\n📺 Recommended Videos:")
        for i, video in enumerate(recommendations['content']['videos'], 1):
            print(f"  {i}. {video}")
        
        # Show music
        print(f"\n🎵 Recommended Music:")
        for music in recommendations['content']['music']:
            print(f"  • {music}")
        
        # Show activities
        print(f"\n🏃 Recommended Activities:")
        for activity in recommendations['content']['activities']:
            print(f"  • {activity}")
        
        return recommendations
    
    def draw_button(self, x, y, width, height, text, color, hover_color, is_hover=False, is_selected=False):
        """Draw a button with hover effects."""
        button_color = hover_color if is_hover else color
        if is_selected:
            button_color = self.colors['accent']
        
        # Button background
        pygame.draw.rect(self.screen, button_color, (x, y, width, height), border_radius=8)
        pygame.draw.rect(self.screen, self.colors['border'], (x, y, width, height), 2, border_radius=8)
        
        # Button text
        text_surface = self.fonts['body'].render(text, True, self.colors['text'])
        text_rect = text_surface.get_rect(center=(x + width//2, y + height//2))
        self.screen.blit(text_surface, text_rect)
        
        return pygame.Rect(x, y, width, height)
    
    def draw_card(self, x, y, width, height, title, content, icon="📋"):
        """Draw a content card."""
        # Card background
        pygame.draw.rect(self.screen, self.colors['card'], (x, y, width, height), border_radius=12)
        pygame.draw.rect(self.screen, self.colors['border'], (x, y, width, height), 2, border_radius=12)
        
        # Card title
        title_surface = self.fonts['subtitle'].render(f"{icon} {title}", True, self.colors['text'])
        self.screen.blit(title_surface, (x + 20, y + 20))
        
        # Card content
        y_offset = y + 60
        for item in content:
            if y_offset < y + height - 20:
                item_surface = self.fonts['body'].render(f"• {item}", True, self.colors['text_secondary'])
                self.screen.blit(item_surface, (x + 30, y_offset))
                y_offset += 30
    
    def draw_progress_bar(self, x, y, width, height, progress, color):
        """Draw a progress bar."""
        # Background
        pygame.draw.rect(self.screen, self.colors['secondary'], (x, y, width, height), border_radius=height//2)
        
        # Progress
        progress_width = int(width * progress)
        pygame.draw.rect(self.screen, color, (x, y, progress_width, height), border_radius=height//2)
        
        # Border
        pygame.draw.rect(self.screen, self.colors['border'], (x, y, width, height), 2, border_radius=height//2)
    
    def draw_emotion_analysis(self):
        """Draw the emotion analysis screen."""
        self.screen.fill(self.colors['background'])
        
        # Title
        title_surface = self.fonts['title'].render("🎯 Emotion Analysis Complete", True, self.colors['text'])
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
        continue_rect = self.draw_button(200, analysis_y + 130, 200, 50, "View Recommendations", 
                                      self.colors['primary'], self.colors['accent'])
        self.buttons = [continue_rect]
    
    def draw_recommendations(self, camera_frame=None):
        """Draw the recommendations screen with live camera feed."""
        self.screen.fill(self.colors['background'])
        
        # Title
        title_surface = self.fonts['title'].render("💡 Personalized Recommendations", True, self.colors['text'])
        self.screen.blit(title_surface, (50, 30))
        
        # Emotion-specific message
        if self.recommendations:
            message = self.recommendations['title']
            message_surface = self.fonts['subtitle'].render(message, True, self.colors['accent'])
            self.screen.blit(message_surface, (50, 80))
        
        # Live camera feed (top right)
        if camera_frame is not None:
            # Resize camera frame to fit in the corner
            camera_width, camera_height = 200, 150
            camera_frame_resized = cv2.resize(camera_frame, (camera_width, camera_height))
            
            # Convert BGR to RGB for pygame
            camera_frame_rgb = cv2.cvtColor(camera_frame_resized, cv2.COLOR_BGR2RGB)
            camera_surface = pygame.surfarray.make_surface(camera_frame_rgb.swapaxes(0, 1))
            
            # Draw camera frame with border
            camera_x = self.screen_width - camera_width - 50
            camera_y = 50
            self.screen.blit(camera_surface, (camera_x, camera_y))
            pygame.draw.rect(self.screen, self.colors['border'], (camera_x, camera_y, camera_width, camera_height), 3, border_radius=8)
            
            # Live detection label
            live_text = "🔴 LIVE"
            live_surface = self.fonts['small'].render(live_text, True, self.colors['danger'])
            self.screen.blit(live_surface, (camera_x + 10, camera_y + 10))
            
            # Current emotion on camera
            emotion_text = f"Emotion: {self.current_emotion.title()}"
            emotion_surface = self.fonts['small'].render(emotion_text, True, self.colors['text'])
            self.screen.blit(emotion_surface, (camera_x + 10, camera_y + camera_height - 30))
        
        # Content cards (left side)
        card_width = 300
        card_height = 250
        card_spacing = 20
        start_x = 50
        start_y = 130
        
        if self.recommendations:
            # Videos card
            self.draw_card(start_x, start_y, card_width, card_height, 
                          "Recommended Videos", 
                          self.recommendations['content']['videos'][:3], "📺")
            
            # Music card
            self.draw_card(start_x + card_width + card_spacing, start_y, card_width, card_height,
                          "Recommended Music", 
                          self.recommendations['content']['music'], "🎵")
            
            # Activities card
            self.draw_card(start_x, start_y + card_height + card_spacing, card_width, card_height,
                          "Suggested Activities", 
                          self.recommendations['content']['activities'][:4], "🏃")
        
        # Action buttons
        button_y = start_y + card_height + card_spacing + card_height + 20
        button_width = 180
        button_height = 50
        button_spacing = 15
        
        buttons = []
        
        # Watch video button
        video_rect = self.draw_button(start_x, button_y, button_width, button_height, 
                                    "Watch Video", self.colors['primary'], self.colors['accent'])
        buttons.append(('video', video_rect))
        
        # Listen to music button
        music_rect = self.draw_button(start_x + button_width + button_spacing, button_y, button_width, button_height,
                                    "Listen to Music", self.colors['success'], self.colors['accent'])
        buttons.append(('music', music_rect))
        
        # Try activity button
        activity_rect = self.draw_button(start_x + 2 * (button_width + button_spacing), button_y, button_width, button_height,
                                       "Try Activity", self.colors['warning'], self.colors['accent'])
        buttons.append(('activity', activity_rect))
        
        # Exit button
        exit_rect = self.draw_button(start_x + 3 * (button_width + button_spacing), button_y, button_width, button_height,
                                   "Exit", self.colors['danger'], self.colors['accent'])
        buttons.append(('exit', exit_rect))
        
        self.buttons = buttons
    
    def draw_video_selection(self, camera_frame=None):
        """Draw the video selection screen with live camera feed."""
        self.screen.fill(self.colors['background'])
        
        # Title
        title_surface = self.fonts['title'].render("📺 Choose a Video", True, self.colors['text'])
        self.screen.blit(title_surface, (50, 50))
        
        # Live camera feed (top right)
        if camera_frame is not None:
            # Resize camera frame to fit in the corner
            camera_width, camera_height = 200, 150
            camera_frame_resized = cv2.resize(camera_frame, (camera_width, camera_height))
            
            # Convert BGR to RGB for pygame
            camera_frame_rgb = cv2.cvtColor(camera_frame_resized, cv2.COLOR_BGR2RGB)
            camera_surface = pygame.surfarray.make_surface(camera_frame_rgb.swapaxes(0, 1))
            
            # Draw camera frame with border
            camera_x = self.screen_width - camera_width - 50
            camera_y = 50
            self.screen.blit(camera_surface, (camera_x, camera_y))
            pygame.draw.rect(self.screen, self.colors['border'], (camera_x, camera_y, camera_width, camera_height), 3, border_radius=8)
            
            # Live detection label
            live_text = "🔴 LIVE"
            live_surface = self.fonts['small'].render(live_text, True, self.colors['danger'])
            self.screen.blit(live_surface, (camera_x + 10, camera_y + 10))
            
            # Current emotion on camera
            emotion_text = f"Emotion: {self.current_emotion.title()}"
            emotion_surface = self.fonts['small'].render(emotion_text, True, self.colors['text'])
            self.screen.blit(emotion_surface, (camera_x + 10, camera_y + camera_height - 30))
        
        if self.recommendations:
            # Video options
            video_y = 120
            for i, video in enumerate(self.recommendations['content']['videos']):
                # Video card
                card_rect = pygame.Rect(50, video_y + i * 120, 800, 100)
                pygame.draw.rect(self.screen, self.colors['card'], card_rect, border_radius=8)
                pygame.draw.rect(self.screen, self.colors['border'], card_rect, 2, border_radius=8)
                
                # Video number and URL
                video_text = f"{i+1}. {video}"
                video_surface = self.fonts['body'].render(video_text, True, self.colors['text'])
                self.screen.blit(video_surface, (70, video_y + i * 120 + 30))
                
                # Select button
                select_rect = self.draw_button(700, video_y + i * 120 + 25, 100, 50, "Select", 
                                             self.colors['primary'], self.colors['accent'])
                self.buttons.append(('select_video', select_rect, i))
        
        # Back button
        back_rect = self.draw_button(50, 600, 150, 50, "← Back", 
                                   self.colors['secondary'], self.colors['accent'])
        self.buttons.append(('back', back_rect))
    
    def handle_click(self, pos):
        """Handle mouse clicks."""
        for button_info in self.buttons:
            if len(button_info) == 2:
                button_type, button_rect = button_info
                extra = None
            else:
                button_type, button_rect, extra = button_info
            
            if button_rect.collidepoint(pos):
                if button_type == 'video':
                    self.current_screen = 'video_selection'
                    self.buttons = []
                elif button_type == 'music':
                    self.show_music_recommendations()
                elif button_type == 'activity':
                    self.show_activity_recommendations()
                elif button_type == 'exit':
                    return False
                elif button_type == 'select_video':
                    video_url = self.recommendations['content']['videos'][extra]
                    webbrowser.open(video_url)
                elif button_type == 'back':
                    self.current_screen = 'recommendations'
                    self.buttons = []
        return True
    
    def show_music_recommendations(self):
        """Show music recommendations."""
        print("\n🎵 Music Recommendations:")
        if self.recommendations:
            for music in self.recommendations['content']['music']:
                print(f"  • {music}")
        print("\n💡 Search for these on your favorite music platform!")
    
    def show_activity_recommendations(self):
        """Show activity recommendations."""
        print("\n🏃 Activity Recommendations:")
        if self.recommendations:
            for activity in self.recommendations['content']['activities']:
                print(f"  • {activity}")
        print("\n💡 Try any of these activities to help improve your mood!")
    
    def run(self):
        """Run the emotion detection system with beautiful GUI."""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("📹 Camera initialized")
        print("🎭 Starting 1-minute emotion detection...")
        print("⏱️  Monitoring for exactly 60 seconds...")
        
        # Start monitoring
        self.monitoring_start_time = time.time()
        clock = pygame.time.Clock()
        running = True
        
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if not self.handle_click(event.pos):
                            running = False
                
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
                    title_surface = self.fonts['title'].render("🎭 Emotion Detection", True, self.colors['text'])
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
                    live_text = "🔴 LIVE DETECTION"
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
                    self.draw_progress_bar(status_x, status_y, 300, 30, progress, self.colors['primary'])
                    
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
                        print(f"\n⏰ 1 minute complete! Analyzing emotions...")
                        # Analyze emotions
                        if self.emotion_buffer:
                            self.dominant_emotion, self.avg_confidence = self.analyze_emotions()
                            self.recommendations = self.personalized_content[self.dominant_emotion]
                            self.current_screen = 'analysis'
                        else:
                            print("❌ No emotion data collected. Please try again.")
                            running = False
                
                elif self.current_screen == 'analysis':
                    self.draw_emotion_analysis()
                    # Check for continue button click
                    for event in pygame.event.get():
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if self.handle_click(event.pos):
                                self.current_screen = 'recommendations'
                            else:
                                running = False
                
                elif self.current_screen == 'recommendations':
                    # Continue emotion detection during recommendations
                    frame, emotions = self.detect_emotions(frame)
                    self.update_emotion_buffer(emotions)
                    self.draw_recommendations(frame)
                
                elif self.current_screen == 'video_selection':
                    # Continue emotion detection during video selection
                    frame, emotions = self.detect_emotions(frame)
                    self.update_emotion_buffer(emotions)
                    self.draw_video_selection(frame)
                
                # Update display
                pygame.display.flip()
                clock.tick(30)  # 30 FPS
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        
        finally:
            cap.release()
            pygame.quit()
            print("✅ System stopped")

def main():
    """Main function."""
    system = EmotionPersonalizedSystem()
    if hasattr(system, 'emotion_model'):
        system.run()
    else:
        print("❌ System initialization failed")

if __name__ == "__main__":
    main()
