"""
Real-World Testing System for Ensemble Emotion Recognition
Tests ensemble model performance with live camera feed
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

class RealWorldEnsembleTester:
    """
    Real-world testing system for ensemble emotion recognition
    """
    
    def __init__(self, models_path: str = "ensemble_learning/results"):
        self.models_path = Path(models_path)
        self.models = {}
        self.ensemble_weights = {}
        self.emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
        self.face_cascade = None
        self.test_results = {}
        self.session_data = []
        
    def load_models(self):
        """Load trained ensemble models"""
        print("Loading ensemble models for real-world testing...")
        
        model_files = {
            'mini_xception': 'mini_xception_ensemble.h5',
            'mobilenetv2': 'mobilenetv2_ensemble.h5',
            'efficientnetb0': 'efficientnetb0_ensemble.h5'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_path / filename
            if model_path.exists():
                self.models[model_name] = keras.models.load_model(model_path)
                print(f"✓ Loaded {model_name}")
            else:
                print(f"✗ Model not found: {model_name}")
        
        # Load ensemble weights
        weights_path = self.models_path / "ensemble_weights.json"
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                self.ensemble_weights = json.load(f)
            print("✓ Loaded ensemble weights")
        else:
            print("Warning: Using equal weights for ensemble")
            available_models = list(self.models.keys())
            self.ensemble_weights = {model: 1.0/len(available_models) for model in available_models}
        
        # Load face cascade
        cascade_path = "../haarcascade_files/haarcascade_frontalface_default.xml"
        if Path(cascade_path).exists():
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✓ Loaded face cascade")
        else:
            print("Warning: Face cascade not found, trying OpenCV default")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def preprocess_face(self, face_img: np.ndarray, target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
        """Preprocess face image for model input"""
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        face_img = cv2.resize(face_img, target_size)
        
        # Normalize
        face_img = face_img.astype('float32') / 255.0
        
        # Add channel dimension for grayscale
        face_img = np.expand_dims(face_img, axis=-1)
        
        # Add batch dimension
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img
    
    def prepare_rgb_face(self, face_img: np.ndarray) -> np.ndarray:
        """Prepare RGB version for transfer learning models"""
        # Remove batch dimension if present
        if face_img.ndim == 4:
            face_img = face_img[0]
        
        # Convert grayscale to RGB
        face_rgb = np.repeat(face_img, 3, axis=-1)
        
        # Add batch dimension
        face_rgb = np.expand_dims(face_rgb, axis=0)
        
        return face_rgb
    
    def get_ensemble_prediction(self, face_img: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Get ensemble prediction from all models"""
        predictions = {}
        
        # Mini-XCEPTION prediction (grayscale)
        if 'mini_xception' in self.models:
            predictions['mini_xception'] = self.models['mini_xception'].predict(face_img)
        
        # Transfer learning models predictions (RGB)
        face_rgb = self.prepare_rgb_face(face_img)
        
        if 'mobilenetv2' in self.models:
            predictions['mobilenetv2'] = self.models['mobilenetv2'].predict(face_rgb)
        
        if 'efficientnetb0' in self.models:
            predictions['efficientnetb0'] = self.models['efficientnetb0'].predict(face_rgb)
        
        # Create ensemble prediction
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        
        for model_name, pred in predictions.items():
            if model_name in self.ensemble_weights:
                ensemble_pred += self.ensemble_weights[model_name] * pred
        
        return ensemble_pred, predictions
    
    def run_live_test(self, duration_seconds: int = 60, save_session: bool = True):
        """Run live emotion recognition test"""
        print(f"Starting live emotion recognition test for {duration_seconds} seconds...")
        print("Press 'q' to quit early, 's' to save current session")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Session metadata
        session_start = datetime.now()
        session_id = session_start.strftime("%Y%m%d_%H%M%S")
        
        # Performance tracking
        frame_count = 0
        detection_count = 0
        fps_times = []
        prediction_times = []
        
        # Emotion tracking
        emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
        confidence_scores = []
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_start = time.time()
            frame_count += 1
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Process each face
            for (x, y, w, h) in faces:
                detection_count += 1
                
                # Extract face region
                face_img = gray[y:y+h, x:x+w]
                
                # Preprocess face
                processed_face = self.preprocess_face(face_img)
                
                # Get ensemble prediction
                pred_start = time.time()
                ensemble_pred, individual_preds = self.get_ensemble_prediction(processed_face)
                pred_time = time.time() - pred_start
                prediction_times.append(pred_time)
                
                # Get predicted emotion and confidence
                predicted_class = np.argmax(ensemble_pred[0])
                confidence = np.max(ensemble_pred[0])
                predicted_emotion = self.emotion_labels[predicted_class]
                
                # Update tracking
                emotion_counts[predicted_emotion] += 1
                confidence_scores.append(confidence)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw emotion label and confidence
                label = f"{predicted_emotion}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw individual model predictions (optional)
                if len(individual_preds) > 1:
                    y_offset = 30
                    for model_name, pred in individual_preds.items():
                        model_pred = np.argmax(pred[0])
                        model_conf = np.max(pred[0])
                        model_emotion = self.emotion_labels[model_pred]
                        model_label = f"{model_name}: {model_emotion} ({model_conf:.2f})"
                        cv2.putText(frame, model_label, (x, y+y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_offset += 20
            
            # Calculate and display FPS
            fps_time = time.time() - frame_start
            fps_times.append(fps_time)
            current_fps = 1.0 / fps_time if fps_time > 0 else 0
            
            # Display FPS and session info
            fps_text = f"FPS: {current_fps:.1f}"
            session_text = f"Session: {int(time.time() - start_time)}s"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, session_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display current emotion distribution
            if detection_count > 0:
                most_common = max(emotion_counts.items(), key=lambda x: x[1])
                distribution_text = f"Most detected: {most_common[0]} ({most_common[1]})"
                cv2.putText(frame, distribution_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Ensemble Emotion Recognition', frame)
            
            # Check for quit or save
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested by user")
                break
            elif key == ord('s'):
                print("Save session requested")
                if save_session:
                    self._save_session_data(session_id, emotion_counts, confidence_scores, 
                                          prediction_times, fps_times, detection_count)
            
            # Check duration
            if time.time() - start_time >= duration_seconds:
                print(f"Test completed after {duration_seconds} seconds")
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate session statistics
        session_end = datetime.now()
        total_time = session_end - session_start
        
        avg_fps = np.mean([1.0/t for t in fps_times]) if fps_times else 0
        avg_prediction_time = np.mean(prediction_times) if prediction_times else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        session_stats = {
            'session_id': session_id,
            'duration_seconds': total_time.total_seconds(),
            'total_frames': frame_count,
            'face_detections': detection_count,
            'avg_fps': avg_fps,
            'avg_prediction_time': avg_prediction_time,
            'avg_confidence': avg_confidence,
            'emotion_distribution': emotion_counts,
            'detection_rate': detection_count / frame_count if frame_count > 0 else 0
        }
        
        self.test_results[session_id] = session_stats
        
        # Save session data
        if save_session:
            self._save_session_data(session_id, emotion_counts, confidence_scores, 
                                  prediction_times, fps_times, detection_count)
        
        # Print session summary
        self._print_session_summary(session_stats)
        
        return session_stats
    
    def _save_session_data(self, session_id: str, emotion_counts: Dict, confidence_scores: List,
                          prediction_times: List, fps_times: List, detection_count: int):
        """Save session data to files"""
        session_dir = self.models_path / "live_sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save session metadata
        session_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'emotion_counts': emotion_counts,
            'confidence_scores': confidence_scores,
            'prediction_times': prediction_times,
            'fps_times': fps_times,
            'detection_count': detection_count
        }
        
        with open(session_dir / "session_data.json", 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Save emotion distribution as CSV
        emotion_df = pd.DataFrame(list(emotion_counts.items()), columns=['emotion', 'count'])
        emotion_df['percentage'] = emotion_df['count'] / emotion_df['count'].sum() * 100
        emotion_df.to_csv(session_dir / "emotion_distribution.csv", index=False)
        
        print(f"Session data saved to {session_dir}")
    
    def _print_session_summary(self, session_stats: Dict):
        """Print session summary"""
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Duration: {session_stats['duration_seconds']:.1f} seconds")
        print(f"Total frames: {session_stats['total_frames']}")
        print(f"Face detections: {session_stats['face_detections']}")
        print(f"Detection rate: {session_stats['detection_rate']:.2%}")
        print(f"Average FPS: {session_stats['avg_fps']:.1f}")
        print(f"Average prediction time: {session_stats['avg_prediction_time']*1000:.1f} ms")
        print(f"Average confidence: {session_stats['avg_confidence']:.3f}")
        
        print("\nEmotion Distribution:")
        sorted_emotions = sorted(session_stats['emotion_distribution'].items(), 
                               key=lambda x: x[1], reverse=True)
        for emotion, count in sorted_emotions:
            percentage = count / session_stats['face_detections'] * 100 if session_stats['face_detections'] > 0 else 0
            print(f"  {emotion}: {count} ({percentage:.1f}%)")
    
    def analyze_multiple_sessions(self):
        """Analyze results from multiple test sessions"""
        sessions_dir = self.models_path / "live_sessions"
        if not sessions_dir.exists():
            print("No session data found")
            return
        
        print("Analyzing multiple test sessions...")
        
        session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]
        
        if not session_dirs:
            print("No session directories found")
            return
        
        # Collect data from all sessions
        all_sessions = []
        emotion_totals = {emotion: 0 for emotion in self.emotion_labels}
        
        for session_dir in session_dirs:
            session_file = session_dir / "session_data.json"
            if session_file.exists():
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    all_sessions.append(session_data)
                    
                    # Add to emotion totals
                    for emotion, count in session_data['emotion_counts'].items():
                        emotion_totals[emotion] += count
        
        if not all_sessions:
            print("No valid session data found")
            return
        
        # Create analysis visualizations
        self._create_session_analysis_plots(all_sessions, emotion_totals)
        
        # Save summary report
        self._save_session_analysis_report(all_sessions, emotion_totals)
    
    def _create_session_analysis_plots(self, all_sessions: List[Dict], emotion_totals: Dict):
        """Create analysis plots for multiple sessions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Real-World Testing Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall emotion distribution
        ax1 = axes[0, 0]
        emotions = list(emotion_totals.keys())
        counts = list(emotion_totals.values())
        bars = ax1.bar(emotions, counts, color=sns.color_palette("husl", len(emotions)))
        ax1.set_title('Overall Emotion Distribution')
        ax1.set_ylabel('Total Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            if total > 0:
                percentage = count / total * 100
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{percentage:.1f}%', ha='center', va='bottom')
        
        # 2. Performance metrics over sessions
        ax2 = axes[0, 1]
        session_ids = [s['session_id'] for s in all_sessions]
        fps_values = [np.mean([1.0/t for t in s['fps_times']]) if s['fps_times'] else 0 for s in all_sessions]
        ax2.plot(session_ids, fps_values, marker='o', linewidth=2, markersize=6)
        ax2.set_title('FPS Performance Across Sessions')
        ax2.set_ylabel('Average FPS')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Confidence scores distribution
        ax3 = axes[1, 0]
        all_confidences = []
        for session in all_sessions:
            all_confidences.extend(session['confidence_scores'])
        
        if all_confidences:
            ax3.hist(all_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_title('Confidence Score Distribution')
            ax3.set_xlabel('Confidence Score')
            ax3.set_ylabel('Frequency')
            ax3.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_confidences):.3f}')
            ax3.legend()
        
        # 4. Detection rates
        ax4 = axes[1, 1]
        detection_rates = []
        for session in all_sessions:
            if session['total_frames'] > 0:
                rate = session['detection_count'] / session['total_frames']
                detection_rates.append(rate)
            else:
                detection_rates.append(0)
        
        ax4.bar(session_ids, detection_rates, color='lightgreen', alpha=0.7)
        ax4.set_title('Face Detection Rate Across Sessions')
        ax4.set_ylabel('Detection Rate')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.models_path / "real_world_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_session_analysis_report(self, all_sessions: List[Dict], emotion_totals: Dict):
        """Save comprehensive session analysis report"""
        report_path = self.models_path / "real_world_testing_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("REAL-WORLD TESTING ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Sessions: {len(all_sessions)}\n")
            f.write(f"Total Face Detections: {sum(emotion_totals.values())}\n")
            
            if all_sessions:
                avg_fps = np.mean([np.mean([1.0/t for t in s['fps_times']]) if s['fps_times'] else 0 for s in all_sessions])
                avg_confidence = np.mean([np.mean(s['confidence_scores']) if s['confidence_scores'] else 0 for s in all_sessions])
                f.write(f"Average FPS: {avg_fps:.2f}\n")
                f.write(f"Average Confidence: {avg_confidence:.3f}\n")
            
            f.write("\nOVERALL EMOTION DISTRIBUTION\n")
            f.write("-" * 30 + "\n")
            total_detections = sum(emotion_totals.values())
            for emotion, count in sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_detections * 100 if total_detections > 0 else 0
                f.write(f"{emotion}: {count} ({percentage:.1f}%)\n")
            
            f.write("\nPER-SESSION DETAILS\n")
            f.write("-" * 20 + "\n")
            for session in all_sessions:
                f.write(f"\nSession {session['session_id']}:\n")
                f.write(f"  Detections: {session['detection_count']}\n")
                if session['fps_times']:
                    avg_fps = np.mean([1.0/t for t in session['fps_times']])
                    f.write(f"  Average FPS: {avg_fps:.2f}\n")
                if session['confidence_scores']:
                    avg_conf = np.mean(session['confidence_scores'])
                    f.write(f"  Average Confidence: {avg_conf:.3f}\n")
                
                # Top emotion for this session
                top_emotion = max(session['emotion_counts'].items(), key=lambda x: x[1])
                f.write(f"  Most detected emotion: {top_emotion[0]} ({top_emotion[1]})\n")
        
        print(f"Real-world testing report saved to {report_path}")

def main():
    """Main real-world testing pipeline"""
    print("=== Real-World Ensemble Testing ===")
    
    # Initialize tester
    tester = RealWorldEnsembleTester()
    
    # Load models
    tester.load_models()
    
    if not tester.models:
        print("No models found! Please train models first using ensemble_trainer.py")
        return
    
    # Run live test
    print("\nStarting live emotion recognition test...")
    print("Instructions:")
    print("- Make different facial expressions")
    print("- Try different lighting conditions")
    print("- Move your face around")
    print("- Press 'q' to quit, 's' to save session")
    
    session_stats = tester.run_live_test(duration_seconds=60, save_session=True)
    
    # Analyze sessions
    print("\nAnalyzing test sessions...")
    tester.analyze_multiple_sessions()
    
    print("\n=== REAL-WORLD TESTING COMPLETED ===")

if __name__ == "__main__":
    main()
