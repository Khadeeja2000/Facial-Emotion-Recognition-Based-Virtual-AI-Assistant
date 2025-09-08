"""
Content Recommendation System for Emotion Recognition

This module provides personalized content recommendations based on detected emotions:
- Music recommendations
- Video content suggestions
- Activity recommendations
- Wellness tips and exercises
- Social interaction suggestions
"""

import json
import random
import time
from datetime import datetime
import requests
from typing import Dict, List, Tuple, Optional

class ContentRecommendationEngine:
    """AI-powered content recommendation engine based on emotional states."""
    
    def __init__(self):
        self.emotion_content_map = self._initialize_content_map()
        self.user_preferences = {}
        self.interaction_history = []
        self.content_ratings = {}
        
    def _initialize_content_map(self) -> Dict:
        """Initialize the emotion-to-content mapping."""
        return {
            "happy": {
                "music": [
                    {"title": "Happy", "artist": "Pharrell Williams", "genre": "Pop", "mood": "uplifting"},
                    {"title": "Good Life", "artist": "OneRepublic", "genre": "Pop Rock", "mood": "positive"},
                    {"title": "Walking on Sunshine", "artist": "Katrina & The Waves", "genre": "Pop", "mood": "energetic"},
                    {"title": "Don't Stop Believin'", "artist": "Journey", "genre": "Rock", "mood": "inspiring"},
                    {"title": "I Gotta Feeling", "artist": "The Black Eyed Peas", "genre": "Pop", "mood": "celebratory"}
                ],
                "videos": [
                    {"title": "Funny Cat Compilation", "type": "entertainment", "duration": "5 min", "mood": "humorous"},
                    {"title": "Travel Adventures", "type": "travel", "duration": "10 min", "mood": "inspiring"},
                    {"title": "Dance Tutorial", "type": "fitness", "duration": "15 min", "mood": "energetic"},
                    {"title": "Success Stories", "type": "motivational", "duration": "8 min", "mood": "inspiring"},
                    {"title": "Comedy Skits", "type": "comedy", "duration": "6 min", "mood": "humorous"}
                ],
                "activities": [
                    {"name": "Social Gathering", "type": "social", "duration": "2 hours", "benefit": "connection"},
                    {"name": "Creative Project", "type": "creative", "duration": "1 hour", "benefit": "expression"},
                    {"name": "Outdoor Exercise", "type": "physical", "duration": "45 min", "benefit": "energy"},
                    {"name": "Learning New Skill", "type": "educational", "duration": "1 hour", "benefit": "growth"},
                    {"name": "Gratitude Journaling", "type": "mindfulness", "duration": "15 min", "benefit": "reflection"}
                ],
                "wellness_tips": [
                    "Share your joy with others - happiness multiplies when shared",
                    "Document this positive moment in a gratitude journal",
                    "Use this energy to help someone else feel good",
                    "Channel your happiness into creative expression",
                    "Plan something fun for the future to maintain momentum"
                ]
            },
            "sad": {
                "music": [
                    {"title": "Fix You", "artist": "Coldplay", "genre": "Alternative", "mood": "comforting"},
                    {"title": "Bridge Over Troubled Water", "artist": "Simon & Garfunkel", "genre": "Folk", "mood": "soothing"},
                    {"title": "Lean On Me", "artist": "Bill Withers", "genre": "Soul", "mood": "supportive"},
                    {"title": "Don't Give Up", "artist": "Peter Gabriel", "genre": "Rock", "mood": "encouraging"},
                    {"title": "You've Got a Friend", "artist": "James Taylor", "genre": "Folk", "mood": "comforting"}
                ],
                "videos": [
                    {"title": "Gentle Yoga Flow", "type": "wellness", "duration": "20 min", "mood": "calming"},
                    {"title": "Nature Sounds", "type": "relaxation", "duration": "30 min", "mood": "soothing"},
                    {"title": "Heartwarming Stories", "type": "inspirational", "duration": "12 min", "mood": "uplifting"},
                    {"title": "Meditation Guide", "type": "mindfulness", "duration": "15 min", "mood": "centering"},
                    {"title": "Pet Therapy", "type": "comfort", "duration": "8 min", "mood": "healing"}
                ],
                "activities": [
                    {"name": "Gentle Walk", "type": "physical", "duration": "30 min", "benefit": "mood_lift"},
                    {"name": "Warm Bath", "type": "self_care", "duration": "20 min", "benefit": "relaxation"},
                    {"name": "Call a Friend", "type": "social", "duration": "15 min", "benefit": "connection"},
                    {"name": "Creative Writing", "type": "expressive", "duration": "30 min", "benefit": "release"},
                    {"name": "Mindful Breathing", "type": "meditation", "duration": "10 min", "benefit": "calm"}
                ],
                "wellness_tips": [
                    "It's okay to feel sad - emotions are temporary visitors",
                    "Reach out to someone you trust - you don't have to face this alone",
                    "Practice self-compassion - treat yourself as you would a dear friend",
                    "Small acts of self-care can make a big difference",
                    "Remember that difficult times often lead to growth and strength"
                ]
            },
            "angry": {
                "music": [
                    {"title": "Let It Be", "artist": "The Beatles", "genre": "Rock", "mood": "calming"},
                    {"title": "Don't Worry, Be Happy", "artist": "Bobby McFerrin", "genre": "Jazz", "mood": "cheerful"},
                    {"title": "Three Little Birds", "artist": "Bob Marley", "genre": "Reggae", "mood": "soothing"},
                    {"title": "Here Comes the Sun", "artist": "The Beatles", "genre": "Rock", "mood": "uplifting"},
                    {"title": "What a Wonderful World", "artist": "Louis Armstrong", "genre": "Jazz", "mood": "peaceful"}
                ],
                "videos": [
                    {"title": "Breathing Exercises", "type": "calming", "duration": "10 min", "mood": "centering"},
                    {"title": "Progressive Relaxation", "type": "stress_relief", "duration": "15 min", "mood": "relaxing"},
                    {"title": "Anger Management Tips", "type": "educational", "duration": "8 min", "mood": "informative"},
                    {"title": "Peaceful Nature Scenes", "type": "relaxation", "duration": "12 min", "mood": "soothing"},
                    {"title": "Mindfulness Practice", "type": "meditation", "duration": "20 min", "mood": "centering"}
                ],
                "activities": [
                    {"name": "Deep Breathing", "type": "calming", "duration": "5 min", "benefit": "immediate_relief"},
                    {"name": "Physical Exercise", "type": "physical", "duration": "30 min", "benefit": "energy_release"},
                    {"name": "Journal Writing", "type": "expressive", "duration": "20 min", "benefit": "emotional_release"},
                    {"name": "Cold Shower", "type": "physical", "duration": "5 min", "benefit": "physiological_calm"},
                    {"name": "Counting Exercise", "type": "focusing", "duration": "2 min", "benefit": "mental_distraction"}
                ],
                "wellness_tips": [
                    "Take 10 deep breaths - oxygen helps calm the nervous system",
                    "Count to 10 slowly - this gives your brain time to process",
                    "Remove yourself from the situation if possible",
                    "Express your feelings in a safe, constructive way",
                    "Remember that anger is often a secondary emotion - what's underneath?"
                ]
            },
            "stressed": {
                "music": [
                    {"title": "Weightless", "artist": "Marconi Union", "genre": "Ambient", "mood": "relaxing"},
                    {"title": "Claire de Lune", "artist": "Debussy", "genre": "Classical", "mood": "soothing"},
                    {"title": "The Sound of Silence", "artist": "Disturbed", "genre": "Rock", "mood": "contemplative"},
                    {"title": "River Flows in You", "artist": "Yiruma", "genre": "Piano", "mood": "peaceful"},
                    {"title": "Adagio for Strings", "artist": "Barber", "genre": "Classical", "mood": "calming"}
                ],
                "videos": [
                    {"title": "Guided Meditation", "type": "mindfulness", "duration": "20 min", "mood": "centering"},
                    {"title": "Stress Relief Techniques", "type": "wellness", "duration": "15 min", "mood": "educational"},
                    {"title": "Nature Walks", "type": "relaxation", "duration": "25 min", "mood": "soothing"},
                    {"title": "Progressive Muscle Relaxation", "type": "relaxation", "duration": "18 min", "mood": "calming"},
                    {"title": "Mindful Movement", "type": "exercise", "duration": "30 min", "mood": "gentle"}
                ],
                "activities": [
                    {"name": "Progressive Relaxation", "type": "relaxation", "duration": "15 min", "benefit": "muscle_relief"},
                    {"name": "Mindful Walking", "type": "exercise", "duration": "20 min", "benefit": "mental_clarity"},
                    {"name": "Tea Ritual", "type": "self_care", "duration": "10 min", "benefit": "slowing_down"},
                    {"name": "Stress Ball Exercise", "type": "physical", "duration": "5 min", "benefit": "tension_release"},
                    {"name": "Gratitude Practice", "type": "mindfulness", "duration": "10 min", "benefit": "perspective"}
                ],
                "wellness_tips": [
                    "Practice the 4-7-8 breathing technique: inhale 4, hold 7, exhale 8",
                    "Take regular breaks - even 5 minutes can reset your stress levels",
                    "Prioritize tasks - not everything needs to be done right now",
                    "Use the 5-4-3-2-1 grounding technique to stay present",
                    "Remember that stress is temporary - this too shall pass"
                ]
            },
            "neutral": {
                "music": [
                    {"title": "The Scientist", "artist": "Coldplay", "genre": "Alternative", "mood": "contemplative"},
                    {"title": "Hallelujah", "artist": "Jeff Buckley", "genre": "Folk", "mood": "reflective"},
                    {"title": "Mad World", "artist": "Gary Jules", "genre": "Alternative", "mood": "thoughtful"},
                    {"title": "Creep", "artist": "Radiohead", "genre": "Alternative", "mood": "introspective"},
                    {"title": "Fix You", "artist": "Coldplay", "genre": "Alternative", "mood": "melancholic"}
                ],
                "videos": [
                    {"title": "Educational Content", "type": "learning", "duration": "15 min", "mood": "informative"},
                    {"title": "Documentary", "type": "educational", "duration": "45 min", "mood": "engaging"},
                    {"title": "Tutorial", "type": "learning", "duration": "20 min", "mood": "practical"},
                    {"title": "News Analysis", "type": "informative", "duration": "12 min", "mood": "current"},
                    {"title": "Skill Development", "type": "educational", "duration": "25 min", "mood": "productive"}
                ],
                "activities": [
                    {"name": "Goal Setting", "type": "planning", "duration": "30 min", "benefit": "direction"},
                    {"name": "Skill Learning", "type": "educational", "duration": "1 hour", "benefit": "growth"},
                    {"name": "Planning Session", "type": "organizational", "duration": "45 min", "benefit": "clarity"},
                    {"name": "Research Project", "type": "intellectual", "duration": "1 hour", "benefit": "knowledge"},
                    {"name": "Creative Exploration", "type": "artistic", "duration": "40 min", "benefit": "expression"}
                ],
                "wellness_tips": [
                    "Use this balanced state to set intentions for the day",
                    "Channel your focus into productive activities",
                    "Take advantage of your mental clarity for decision-making",
                    "Consider what would bring you joy or fulfillment",
                    "This is a great time for self-reflection and planning"
                ]
            }
        }
    
    def get_recommendations(self, emotion: str, confidence: float, user_input: str = "") -> Dict:
        """Get personalized content recommendations based on emotion."""
        timestamp = datetime.now().isoformat()
        
        # Store interaction
        self.interaction_history.append({
            'timestamp': timestamp,
            'emotion': emotion,
            'confidence': confidence,
            'user_input': user_input
        })
        
        # Get base recommendations
        if emotion not in self.emotion_content_map:
            emotion = "neutral"  # Default fallback
        
        base_recommendations = self.emotion_content_map[emotion]
        
        # Personalize based on confidence and user input
        personalized_recommendations = self._personalize_recommendations(
            base_recommendations, emotion, confidence, user_input
        )
        
        # Add emotional context and tips
        emotional_context = self._get_emotional_context(emotion, confidence)
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': timestamp,
            'recommendations': personalized_recommendations,
            'emotional_context': emotional_context,
            'interaction_suggestions': self._get_interaction_suggestions(emotion, user_input)
        }
    
    def _personalize_recommendations(self, base_recs: Dict, emotion: str, confidence: float, user_input: str) -> Dict:
        """Personalize recommendations based on user context."""
        personalized = {}
        
        for category, items in base_recs.items():
            if category == "wellness_tips":
                # Select most relevant tips
                personalized[category] = random.sample(items, min(3, len(items)))
            else:
                # Select items based on confidence and user preferences
                num_items = min(3, len(items))
                if confidence > 0.8:
                    # High confidence - more specific recommendations
                    personalized[category] = random.sample(items, num_items)
                else:
                    # Lower confidence - broader recommendations
                    personalized[category] = random.sample(items, num_items)
        
        return personalized
    
    def _get_emotional_context(self, emotion: str, confidence: float) -> Dict:
        """Provide emotional context and understanding."""
        context_map = {
            "happy": {
                "description": "You're experiencing positive emotions",
                "intensity": "high" if confidence > 0.8 else "moderate",
                "suggestion": "Embrace and share this positive energy",
                "duration_estimate": "2-4 hours"
            },
            "sad": {
                "description": "You're feeling down or melancholic",
                "intensity": "high" if confidence > 0.8 else "moderate",
                "suggestion": "Be gentle with yourself and seek support",
                "duration_estimate": "1-3 hours"
            },
            "angry": {
                "description": "You're experiencing frustration or anger",
                "intensity": "high" if confidence > 0.8 else "moderate",
                "suggestion": "Take time to cool down before acting",
                "duration_estimate": "30 min - 2 hours"
            },
            "stressed": {
                "description": "You're feeling overwhelmed or tense",
                "intensity": "high" if confidence > 0.8 else "moderate",
                "suggestion": "Focus on relaxation and stress management",
                "duration_estimate": "1-4 hours"
            },
            "neutral": {
                "description": "You're in a balanced emotional state",
                "intensity": "low",
                "suggestion": "Use this stability for productive activities",
                "duration_estimate": "variable"
            }
        }
        
        return context_map.get(emotion, context_map["neutral"])
    
    def _get_interaction_suggestions(self, emotion: str, user_input: str) -> List[str]:
        """Get suggestions for how to interact with the user."""
        suggestions = []
        
        if user_input:
            suggestions.append(f"Based on your input '{user_input}', I can provide more specific recommendations.")
        
        if emotion in ["sad", "angry", "stressed"]:
            suggestions.append("Would you like to talk about what's on your mind?")
            suggestions.append("I can suggest some calming activities if you'd like.")
        
        elif emotion == "happy":
            suggestions.append("What's making you feel so good? I'd love to hear about it!")
            suggestions.append("This is a great time to plan something fun for later.")
        
        else:
            suggestions.append("How can I best support you right now?")
            suggestions.append("Is there something specific you'd like to focus on?")
        
        return suggestions
    
    def get_emotional_insights(self) -> Dict:
        """Provide insights based on emotional history."""
        if not self.interaction_history:
            return {"message": "Not enough data yet. Keep using the system to get insights!"}
        
        # Analyze recent emotions
        recent_emotions = [entry['emotion'] for entry in self.interaction_history[-10:]]
        emotion_counts = {}
        for emotion in recent_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        total_interactions = len(self.interaction_history)
        
        # Calculate emotional patterns
        positive_emotions = ["happy"]
        challenging_emotions = ["sad", "angry", "stressed"]
        
        positive_count = sum(1 for e in recent_emotions if e in positive_emotions)
        challenging_count = sum(1 for e in recent_emotions if e in challenging_emotions)
        
        insights = {
            "dominant_emotion": dominant_emotion,
            "total_interactions": total_interactions,
            "recent_pattern": {
                "positive_ratio": positive_count / len(recent_emotions),
                "challenging_ratio": challenging_count / len(recent_emotions),
                "neutral_ratio": (len(recent_emotions) - positive_count - challenging_count) / len(recent_emotions)
            },
            "recommendations": self._get_insight_recommendations(dominant_emotion, positive_count, challenging_count)
        }
        
        return insights
    
    def _get_insight_recommendations(self, dominant_emotion: str, positive_count: int, challenging_count: int) -> List[str]:
        """Get recommendations based on emotional insights."""
        recommendations = []
        
        if challenging_count > positive_count:
            recommendations.append("Consider incorporating more stress-relief activities into your routine")
            recommendations.append("You might benefit from talking to a friend or professional about recent challenges")
            recommendations.append("Try to identify triggers for difficult emotions and develop coping strategies")
        
        elif positive_count > challenging_count:
            recommendations.append("Great job maintaining positive emotions! Keep up the good work")
            recommendations.append("Consider sharing your positive energy with others")
            recommendations.append("Document what's working well for you")
        
        else:
            recommendations.append("You seem to have a balanced emotional state")
            recommendations.append("This is a good time to set goals and plan ahead")
            recommendations.append("Consider exploring new activities or hobbies")
        
        return recommendations
    
    def rate_content(self, content_id: str, rating: int, feedback: str = ""):
        """Allow users to rate content and provide feedback."""
        if content_id not in self.content_ratings:
            self.content_ratings[content_id] = []
        
        self.content_ratings[content_id].append({
            'rating': rating,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_user_preferences(self) -> Dict:
        """Get user preferences based on interaction history."""
        if not self.interaction_history:
            return {}
        
        # Analyze preferred content types
        content_preferences = {}
        for entry in self.interaction_history:
            emotion = entry['emotion']
            if emotion not in content_preferences:
                content_preferences[emotion] = 0
            content_preferences[emotion] += 1
        
        # Get most preferred emotion
        preferred_emotion = max(content_preferences, key=content_preferences.get)
        
        return {
            'preferred_emotion': preferred_emotion,
            'total_interactions': len(self.interaction_history),
            'emotion_distribution': content_preferences,
            'suggestions': self._get_preference_suggestions(preferred_emotion)
        }
    
    def _get_preference_suggestions(self, preferred_emotion: str) -> List[str]:
        """Get suggestions based on user preferences."""
        suggestions = []
        
        if preferred_emotion == "happy":
            suggestions.append("You seem to enjoy positive content - consider creating more joyful experiences")
            suggestions.append("Share your happiness with others to multiply the positive impact")
        
        elif preferred_emotion in ["sad", "angry", "stressed"]:
            suggestions.append("You might benefit from exploring stress management techniques")
            suggestions.append("Consider seeking support from friends, family, or professionals")
            suggestions.append("Focus on building resilience and coping strategies")
        
        else:
            suggestions.append("You have a balanced approach to emotional content")
            suggestions.append("Consider exploring new types of content to expand your interests")
        
        return suggestions

def main():
    """Test the content recommendation system."""
    print("Content Recommendation System Test")
    print("="*40)
    
    # Initialize the system
    recommender = ContentRecommendationEngine()
    
    # Test different emotions
    test_emotions = ["happy", "sad", "angry", "stressed", "neutral"]
    
    for emotion in test_emotions:
        print(f"\n🎭 Testing emotion: {emotion.upper()}")
        print("-" * 30)
        
        # Get recommendations
        recommendations = recommender.get_recommendations(emotion, 0.85)
        
        # Display music recommendations
        if 'music' in recommendations['recommendations']:
            print("🎵 Music Recommendations:")
            for music in recommendations['recommendations']['music'][:2]:
                print(f"  • {music['title']} by {music['artist']} ({music['mood']})")
        
        # Display wellness tips
        if 'wellness_tips' in recommendations['recommendations']:
            print("💡 Wellness Tips:")
            for tip in recommendations['recommendations']['wellness_tips'][:2]:
                print(f"  • {tip}")
        
        print(f"📊 Emotional Context: {recommendations['emotional_context']['description']}")
    
    # Test insights
    print(f"\n🧠 Emotional Insights:")
    insights = recommender.get_emotional_insights()
    print(f"Dominant emotion: {insights['dominant_emotion']}")
    print(f"Total interactions: {insights['total_interactions']}")
    
    # Test user preferences
    print(f"\n👤 User Preferences:")
    preferences = recommender.get_user_preferences()
    print(f"Preferred emotion: {preferences.get('preferred_emotion', 'None')}")
    print(f"Total interactions: {preferences.get('total_interactions', 0)}")

if __name__ == "__main__":
    main()
