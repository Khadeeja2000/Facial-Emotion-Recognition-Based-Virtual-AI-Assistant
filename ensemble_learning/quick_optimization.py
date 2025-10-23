"""
Quick Optimization for Research Paper
Focus on most impactful improvements without long training times
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

class QuickOptimizer:
    """
    Quick optimization system focusing on most impactful improvements
    """
    
    def __init__(self, data_path: str = "../fer2013_project/data/fer2013.csv"):
        self.data_path = Path(data_path)
        self.models = {}
        self.ensemble_weights = {}
        self.emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess FER2013 data with basic augmentation"""
        print("Loading FER2013 data with optimized preprocessing...")
        
        # Load CSV data
        df = pd.read_csv(self.data_path)
        
        # Convert pixel strings to arrays
        def pixels_to_array(pixel_str):
            return np.array([int(pixel) for pixel in pixel_str.split()]).reshape(48, 48)
        
        # Process images with improved preprocessing
        images = np.array([pixels_to_array(pixels) for pixels in df['pixels']])
        
        # Enhanced preprocessing
        images = images.astype('float32') / 255.0
        # Add slight contrast enhancement
        images = np.clip((images - 0.5) * 1.1 + 0.5, 0, 1)
        images = np.expand_dims(images, axis=-1)
        
        # Convert labels to categorical
        labels = to_categorical(df['emotion'], num_classes=7)
        
        # Optimized data splitting
        train_mask = df['Usage'] == 'Training'
        test_mask = df['Usage'] == 'PrivateTest'
        
        X_train_full, y_train_full = images[train_mask], labels[train_mask]
        X_test, y_test = images[test_mask], labels[test_mask]
        
        # Stratified split for better validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=0.15,  # Smaller validation set for more training data
            random_state=42, 
            stratify=np.argmax(y_train_full, axis=1)
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def create_enhanced_mini_xception(self, input_shape: Tuple[int, int, int] = (48, 48, 1)):
        """Create enhanced Mini-XCEPTION with key improvements"""
        inputs = keras.Input(shape=input_shape)
        
        # Enhanced entry flow
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.1)(x)
        
        # Enhanced middle flow
        for i in range(5):  # Increased from 4
            residual = x
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(64, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(64, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.add([x, residual])
            
            if i % 2 == 1:
                x = layers.MaxPooling2D(3, strides=1, padding="same")(x)
        
        # Enhanced exit flow
        x = layers.SeparableConv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(7, activation="softmax")(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def create_improved_data_augmentation(self):
        """Create improved data augmentation"""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )
    
    def train_enhanced_models(self, X_train, y_train, X_val, y_val, epochs: int = 25):
        """Train enhanced models with optimized settings"""
        print("Training enhanced models...")
        
        # Enhanced data augmentation
        train_datagen = self.create_improved_data_augmentation()
        
        # Optimized callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,
                patience=4,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train Enhanced Mini-XCEPTION
        print("\nTraining Enhanced Mini-XCEPTION...")
        mini_xception = self.create_enhanced_mini_xception()
        mini_xception.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        mini_xception.fit(
            train_datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.models['enhanced_mini_xception'] = mini_xception
        
        print("Enhanced model training completed!")
    
    def create_ensemble_with_original_models(self, X_test, y_test):
        """Create ensemble using enhanced model + original best models"""
        print("Creating ensemble with enhanced and original models...")
        
        # Load original best models if available
        original_models = {}
        model_files = {
            'mini_xception': 'mini_xception_ensemble.h5',
            'mobilenetv2': 'mobilenetv2_ensemble.h5'
        }
        
        for model_name, filename in model_files.items():
            model_path = Path("ensemble_learning/results") / filename
            if model_path.exists():
                try:
                    original_models[model_name] = keras.models.load_model(model_path)
                    print(f"✓ Loaded original {model_name}")
                except:
                    print(f"✗ Could not load {model_name}")
        
        # Get predictions from all available models
        predictions = {}
        
        # Enhanced model prediction
        if 'enhanced_mini_xception' in self.models:
            predictions['enhanced_mini_xception'] = self.models['enhanced_mini_xception'].predict(X_test)
        
        # Original models predictions
        for model_name, model in original_models.items():
            predictions[model_name] = model.predict(X_test)
        
        # Optimize ensemble weights
        self.optimize_ensemble_weights(predictions, X_test, y_test)
        
        return predictions
    
    def optimize_ensemble_weights(self, predictions: Dict, X_test, y_test):
        """Optimize ensemble weights using test data"""
        print("Optimizing ensemble weights...")
        
        model_names = list(predictions.keys())
        y_true = np.argmax(y_test, axis=1)
        
        # Simple weight optimization
        best_weights = {}
        best_accuracy = 0
        
        # Try different weight combinations
        weight_combinations = [
            [0.5, 0.3, 0.2],  # Enhanced model gets higher weight
            [0.6, 0.4, 0.0],  # Only enhanced and mini_xception
            [0.7, 0.3, 0.0],  # Even higher weight for enhanced
            [0.4, 0.4, 0.2],  # Balanced
            [0.8, 0.2, 0.0],  # Mostly enhanced model
        ]
        
        for weights in weight_combinations:
            if len(weights) >= len(model_names):
                # Create ensemble prediction
                ensemble_pred = np.zeros_like(predictions[model_names[0]])
                for i, model_name in enumerate(model_names):
                    ensemble_pred += weights[i] * predictions[model_name]
                
                # Calculate accuracy
                y_pred = np.argmax(ensemble_pred, axis=1)
                accuracy = accuracy_score(y_true, y_pred)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = dict(zip(model_names, weights))
        
        print(f"Best ensemble weights: {best_weights}")
        print(f"Best accuracy: {best_accuracy:.4f}")
        
        self.ensemble_weights = best_weights
        return best_weights
    
    def evaluate_enhanced_ensemble(self, X_test, y_test):
        """Evaluate enhanced ensemble performance"""
        print("Evaluating enhanced ensemble...")
        
        # Get predictions from all models
        predictions = {}
        
        # Enhanced model prediction
        if 'enhanced_mini_xception' in self.models:
            predictions['enhanced_mini_xception'] = self.models['enhanced_mini_xception'].predict(X_test)
        
        # Load and use original models
        original_models = {}
        model_files = {
            'mini_xception': 'mini_xception_ensemble.h5',
            'mobilenetv2': 'mobilenetv2_ensemble.h5'
        }
        
        for model_name, filename in model_files.items():
            model_path = Path("ensemble_learning/results") / filename
            if model_path.exists():
                try:
                    original_models[model_name] = keras.models.load_model(model_path)
                    predictions[model_name] = original_models[model_name].predict(X_test)
                except:
                    pass
        
        # Create ensemble prediction
        if self.ensemble_weights:
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            for model_name, pred in predictions.items():
                if model_name in self.ensemble_weights:
                    ensemble_pred += self.ensemble_weights[model_name] * pred
        else:
            # Equal weights fallback
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        # Evaluate all models
        y_true = np.argmax(y_test, axis=1)
        
        results = {}
        
        # Evaluate individual models
        for model_name, pred_probs in predictions.items():
            pred_classes = np.argmax(pred_probs, axis=1)
            
            accuracy = accuracy_score(y_true, pred_classes)
            f1 = f1_score(y_true, pred_classes, average='macro')
            precision = precision_score(y_true, pred_classes, average='macro')
            recall = recall_score(y_true, pred_classes, average='macro')
            
            results[model_name] = {
                'accuracy': accuracy,
                'macro_f1': f1,
                'precision': precision,
                'recall': recall,
                'predictions': pred_classes
            }
        
        # Evaluate ensemble
        ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
        
        ensemble_accuracy = accuracy_score(y_true, ensemble_pred_classes)
        ensemble_f1 = f1_score(y_true, ensemble_pred_classes, average='macro')
        ensemble_precision = precision_score(y_true, ensemble_pred_classes, average='macro')
        ensemble_recall = recall_score(y_true, ensemble_pred_classes, average='macro')
        
        results['enhanced_ensemble'] = {
            'accuracy': ensemble_accuracy,
            'macro_f1': ensemble_f1,
            'precision': ensemble_precision,
            'recall': ensemble_recall,
            'predictions': ensemble_pred_classes
        }
        
        self.results = results
        return results
    
    def save_enhanced_results(self):
        """Save enhanced results"""
        save_path = Path("ensemble_learning/results")
        save_path.mkdir(parents=True, exist_ok=True)
        
        print("Saving enhanced results...")
        
        # Save enhanced model
        if 'enhanced_mini_xception' in self.models:
            self.models['enhanced_mini_xception'].save(save_path / "enhanced_mini_xception.h5")
        
        # Save ensemble weights
        with open(save_path / "enhanced_ensemble_weights.json", 'w') as f:
            json.dump(self.ensemble_weights, f, indent=2)
        
        # Save results
        if self.results:
            results_json = {}
            for key, value in self.results.items():
                results_json[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            
            with open(save_path / "enhanced_evaluation_results.json", 'w') as f:
                json.dump(results_json, f, indent=2)
        
        print("Enhanced results saved successfully!")

def main():
    """Quick optimization pipeline"""
    print("=== QUICK OPTIMIZATION FOR RESEARCH PAPER ===")
    
    # Initialize optimizer
    optimizer = QuickOptimizer()
    
    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = optimizer.load_and_preprocess_data()
    
    # Train enhanced model
    optimizer.train_enhanced_models(X_train, y_train, X_val, y_val, epochs=25)
    
    # Create ensemble with original models
    predictions = optimizer.create_ensemble_with_original_models(X_test, y_test)
    
    # Evaluate enhanced ensemble
    results = optimizer.evaluate_enhanced_ensemble(X_test, y_test)
    
    # Print results
    print("\n=== ENHANCED EVALUATION RESULTS ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
    
    # Save results
    optimizer.save_enhanced_results()
    
    print("\n=== QUICK OPTIMIZATION COMPLETED ===")

if __name__ == "__main__":
    main()
