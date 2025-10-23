"""
Research Paper Optimization System
Advanced techniques to improve ensemble performance for publication
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

class ResearchOptimizer:
    """
    Advanced optimization system for research paper results
    """
    
    def __init__(self, data_path: str = "../fer2013_project/data/fer2013.csv"):
        self.data_path = Path(data_path)
        self.models = {}
        self.optimized_models = {}
        self.ensemble_weights = {}
        self.emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess FER2013 data with advanced augmentation"""
        print("Loading and preprocessing FER2013 data with advanced augmentation...")
        
        # Load CSV data
        df = pd.read_csv(self.data_path)
        
        # Convert pixel strings to arrays
        def pixels_to_array(pixel_str):
            return np.array([int(pixel) for pixel in pixel_str.split()]).reshape(48, 48)
        
        # Process images
        images = np.array([pixels_to_array(pixels) for pixels in df['pixels']])
        images = images.astype('float32') / 255.0
        images = np.expand_dims(images, axis=-1)  # Add channel dimension
        
        # Convert labels to categorical
        labels = to_categorical(df['emotion'], num_classes=7)
        
        # Advanced data splitting with stratification
        train_mask = df['Usage'] == 'Training'
        test_mask = df['Usage'] == 'PrivateTest'
        
        X_train_full, y_train_full = images[train_mask], labels[train_mask]
        X_test, y_test = images[test_mask], labels[test_mask]
        
        # Create stratified train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=0.2, 
            random_state=42, 
            stratify=np.argmax(y_train_full, axis=1)
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def create_advanced_data_augmentation(self):
        """Create advanced data augmentation pipeline"""
        return ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            channel_shift_range=0.1
        )
    
    def create_optimized_mini_xception(self, input_shape: Tuple[int, int, int] = (48, 48, 1)):
        """Create optimized Mini-XCEPTION with improvements"""
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
        
        # Enhanced middle flow with residual connections
        for i in range(6):  # Increased from 4 to 6
            residual = x
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(64, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(64, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            
            # Add residual connection
            x = layers.add([x, residual])
            
            # Pooling every 2 blocks
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
    
    def create_improved_transfer_learning_model(self, base_model_name: str, input_shape: Tuple[int, int, int] = (48, 48, 3)):
        """Create improved transfer learning model with better fine-tuning"""
        inputs = keras.Input(shape=input_shape)
        
        # Load pre-trained model
        if base_model_name == 'mobilenetv2':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
        elif base_model_name == 'efficientnetb0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)
        elif base_model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
        
        # Gradual unfreezing strategy
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - 30
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(7, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def prepare_rgb_data(self, X_gray: np.ndarray) -> np.ndarray:
        """Convert grayscale to RGB for transfer learning models"""
        return np.repeat(X_gray, 3, axis=-1)
    
    def train_optimized_models(self, X_train, y_train, X_val, y_val, epochs: int = 50):
        """Train optimized models with advanced techniques"""
        print("Training optimized models with advanced techniques...")
        
        # Advanced data augmentation
        train_datagen = self.create_advanced_data_augmentation()
        
        # Advanced callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath='ensemble_learning/results/optimized_mini_xception_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # 1. Train Optimized Mini-XCEPTION
        print("\n1. Training Optimized Mini-XCEPTION...")
        mini_xception = self.create_optimized_mini_xception()
        mini_xception.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
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
        
        self.optimized_models['mini_xception'] = mini_xception
        
        # 2. Train Improved Transfer Learning Models
        transfer_models = ['mobilenetv2', 'efficientnetb0', 'vgg16']
        X_train_rgb = self.prepare_rgb_data(X_train)
        X_val_rgb = self.prepare_rgb_data(X_val)
        
        for model_name in transfer_models:
            print(f"\n{len(self.optimized_models)+1}. Training Improved {model_name.upper()}...")
            
            model = self.create_improved_transfer_learning_model(model_name)
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            model.fit(
                train_datagen.flow(X_train_rgb, y_train, batch_size=16),  # Smaller batch size
                steps_per_epoch=len(X_train_rgb) // 16,
                epochs=epochs,
                validation_data=(X_val_rgb, y_val),
                callbacks=callbacks_list,
                verbose=1
            )
            
            self.optimized_models[model_name] = model
        
        print("\nAll optimized models trained successfully!")
    
    def optimize_ensemble_weights(self, X_val, y_val):
        """Optimize ensemble weights using validation data"""
        print("Optimizing ensemble weights...")
        
        # Get predictions from all models
        predictions = {}
        
        # Mini-XCEPTION prediction
        if 'mini_xception' in self.optimized_models:
            predictions['mini_xception'] = self.optimized_models['mini_xception'].predict(X_val)
        
        # Transfer learning models predictions
        X_val_rgb = self.prepare_rgb_data(X_val)
        for model_name in ['mobilenetv2', 'efficientnetb0', 'vgg16']:
            if model_name in self.optimized_models:
                predictions[model_name] = self.optimized_models[model_name].predict(X_val_rgb)
        
        # Grid search for optimal weights
        best_weights = {}
        best_accuracy = 0
        
        # Generate weight combinations
        weight_combinations = []
        for w1 in np.arange(0.1, 1.0, 0.1):
            for w2 in np.arange(0.1, 1.0, 0.1):
                for w3 in np.arange(0.1, 1.0, 0.1):
                    for w4 in np.arange(0.1, 1.0, 0.1):
                        if abs(w1 + w2 + w3 + w4 - 1.0) < 0.01:  # Sum to 1
                            weight_combinations.append([w1, w2, w3, w4])
        
        model_names = list(predictions.keys())
        
        for weights in weight_combinations[:100]:  # Limit to 100 combinations for speed
            if len(weights) >= len(model_names):
                # Create ensemble prediction
                ensemble_pred = np.zeros_like(predictions[model_names[0]])
                for i, model_name in enumerate(model_names):
                    ensemble_pred += weights[i] * predictions[model_name]
                
                # Calculate accuracy
                y_pred = np.argmax(ensemble_pred, axis=1)
                y_true = np.argmax(y_val, axis=1)
                accuracy = accuracy_score(y_true, y_pred)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = dict(zip(model_names, weights))
        
        print(f"Best ensemble weights: {best_weights}")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        
        self.ensemble_weights = best_weights
        return best_weights
    
    def create_advanced_ensemble_prediction(self, X_test):
        """Create advanced ensemble prediction"""
        predictions = {}
        
        # Mini-XCEPTION prediction
        if 'mini_xception' in self.optimized_models:
            predictions['mini_xception'] = self.optimized_models['mini_xception'].predict(X_test)
        
        # Transfer learning models predictions
        X_test_rgb = self.prepare_rgb_data(X_test)
        for model_name in ['mobilenetv2', 'efficientnetb0', 'vgg16']:
            if model_name in self.optimized_models:
                predictions[model_name] = self.optimized_models[model_name].predict(X_test_rgb)
        
        # Create ensemble prediction with optimized weights
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        
        for model_name, pred in predictions.items():
            if model_name in self.ensemble_weights:
                ensemble_pred += self.ensemble_weights[model_name] * pred
        
        return ensemble_pred, predictions
    
    def evaluate_optimized_models(self, X_test, y_test):
        """Evaluate optimized models with comprehensive metrics"""
        print("Evaluating optimized models...")
        
        # Get predictions
        ensemble_pred, individual_preds = self.create_advanced_ensemble_prediction(X_test)
        
        # Convert to class predictions
        y_true = np.argmax(y_test, axis=1)
        ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
        
        results = {}
        
        # Evaluate ensemble
        ensemble_accuracy = accuracy_score(y_true, ensemble_pred_classes)
        ensemble_f1 = f1_score(y_true, ensemble_pred_classes, average='macro')
        ensemble_precision = precision_score(y_true, ensemble_pred_classes, average='macro')
        ensemble_recall = recall_score(y_true, ensemble_pred_classes, average='macro')
        
        results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'macro_f1': ensemble_f1,
            'precision': ensemble_precision,
            'recall': ensemble_recall,
            'predictions': ensemble_pred_classes
        }
        
        # Evaluate individual models
        for model_name, pred_probs in individual_preds.items():
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
        
        self.results = results
        return results
    
    def save_optimized_results(self):
        """Save optimized results and models"""
        save_path = Path("ensemble_learning/results")
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving optimized models and results...")
        
        # Save optimized models
        for model_name, model in self.optimized_models.items():
            model.save(save_path / f"optimized_{model_name}_final.h5")
        
        # Save ensemble weights
        with open(save_path / "optimized_ensemble_weights.json", 'w') as f:
            json.dump(self.ensemble_weights, f, indent=2)
        
        # Save results
        if self.results:
            results_json = {}
            for key, value in self.results.items():
                results_json[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            
            with open(save_path / "optimized_evaluation_results.json", 'w') as f:
                json.dump(results_json, f, indent=2)
        
        print("Optimized models and results saved successfully!")

def main():
    """Main optimization pipeline"""
    print("=== RESEARCH PAPER OPTIMIZATION PIPELINE ===")
    
    # Initialize optimizer
    optimizer = ResearchOptimizer()
    
    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = optimizer.load_and_preprocess_data()
    
    # Train optimized models
    optimizer.train_optimized_models(X_train, y_train, X_val, y_val, epochs=50)
    
    # Optimize ensemble weights
    optimizer.optimize_ensemble_weights(X_val, y_val)
    
    # Evaluate optimized models
    results = optimizer.evaluate_optimized_models(X_test, y_test)
    
    # Print results
    print("\n=== OPTIMIZED EVALUATION RESULTS ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
    
    # Save results
    optimizer.save_optimized_results()
    
    print("\n=== OPTIMIZATION COMPLETED ===")

if __name__ == "__main__":
    main()
