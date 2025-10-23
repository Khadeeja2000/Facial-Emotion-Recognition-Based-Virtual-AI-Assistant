"""
Ensemble Learning Trainer for Emotion Recognition
Combines Mini-XCEPTION, MobileNetV2, and EfficientNetB0 models
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

class EnsembleEmotionRecognizer:
    """
    Ensemble model combining three CNN architectures for emotion recognition
    """
    
    def __init__(self, data_path: str = "../fer2013_project/data/fer2013.csv"):
        self.data_path = Path(data_path)
        self.models = {}
        self.ensemble_weights = {
            'mini_xception': 0.5,  # Best individual accuracy
            'mobilenetv2': 0.25,   # Transfer learning benefits
            'efficientnetb0': 0.25 # State-of-the-art architecture
        }
        self.emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
        self.results = {}
        
    def load_fer2013_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess FER2013 dataset"""
        print("Loading FER2013 dataset...")
        
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
        labels = keras.utils.to_categorical(df['emotion'], num_classes=7)
        
        # Split data - create validation from training data
        
        train_mask = df['Usage'] == 'Training'
        test_mask = df['Usage'] == 'PrivateTest'
        
        # Get training data
        X_train_full, y_train_full = images[train_mask], labels[train_mask]
        
        # Split training data into train/validation (80/20)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=0.2, 
            random_state=42, 
            stratify=np.argmax(y_train_full, axis=1)
        )
        
        # Get test data
        X_test, y_test = images[test_mask], labels[test_mask]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def create_mini_xception(self, input_shape: Tuple[int, int, int] = (48, 48, 1)) -> keras.Model:
        """Create Mini-XCEPTION model architecture"""
        inputs = keras.Input(shape=input_shape)
        
        # Entry flow
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Middle flow
        for _ in range(4):
            residual = x
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(64, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(64, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(3, strides=1, padding="same")(x)
            x = layers.add([x, residual])
        
        # Exit flow
        x = layers.SeparableConv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(7, activation="softmax")(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def create_mobilenetv2(self, input_shape: Tuple[int, int, int] = (48, 48, 3)) -> keras.Model:
        """Create MobileNetV2 transfer learning model"""
        # Convert grayscale to RGB for transfer learning
        inputs = keras.Input(shape=input_shape)
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(7, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def create_efficientnetb0(self, input_shape: Tuple[int, int, int] = (48, 48, 3)) -> keras.Model:
        """Create EfficientNetB0 transfer learning model"""
        # Convert grayscale to RGB for transfer learning
        inputs = keras.Input(shape=input_shape)
        
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(7, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def prepare_rgb_data(self, X_gray: np.ndarray) -> np.ndarray:
        """Convert grayscale to RGB for transfer learning models"""
        X_rgb = np.repeat(X_gray, 3, axis=-1)
        return X_rgb
    
    def train_individual_models(self, X_train, y_train, X_val, y_val, epochs: int = 30):
        """Train all three individual models"""
        print("Training individual models...")
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
        
        # 1. Train Mini-XCEPTION
        print("\n1. Training Mini-XCEPTION...")
        mini_xception = self.create_mini_xception()
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
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.models['mini_xception'] = mini_xception
        
        # 2. Train MobileNetV2
        print("\n2. Training MobileNetV2...")
        X_train_rgb = self.prepare_rgb_data(X_train)
        X_val_rgb = self.prepare_rgb_data(X_val)
        
        mobilenetv2 = self.create_mobilenetv2()
        mobilenetv2.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        mobilenetv2.fit(
            train_datagen.flow(X_train_rgb, y_train, batch_size=32),
            steps_per_epoch=len(X_train_rgb) // 32,
            epochs=epochs,
            validation_data=(X_val_rgb, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.models['mobilenetv2'] = mobilenetv2
        
        # 3. Train EfficientNetB0
        print("\n3. Training EfficientNetB0...")
        efficientnetb0 = self.create_efficientnetb0()
        efficientnetb0.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        efficientnetb0.fit(
            train_datagen.flow(X_train_rgb, y_train, batch_size=32),
            steps_per_epoch=len(X_train_rgb) // 32,
            epochs=epochs,
            validation_data=(X_val_rgb, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.models['efficientnetb0'] = efficientnetb0
        
        print("\nAll models trained successfully!")
    
    def create_ensemble_prediction(self, X_test) -> np.ndarray:
        """Create ensemble prediction using weighted voting"""
        print("Creating ensemble predictions...")
        
        # Get predictions from all models
        predictions = {}
        
        # Mini-XCEPTION prediction
        predictions['mini_xception'] = self.models['mini_xception'].predict(X_test)
        
        # Transfer learning models predictions (RGB)
        X_test_rgb = self.prepare_rgb_data(X_test)
        predictions['mobilenetv2'] = self.models['mobilenetv2'].predict(X_test_rgb)
        predictions['efficientnetb0'] = self.models['efficientnetb0'].predict(X_test_rgb)
        
        # Weighted ensemble prediction
        ensemble_pred = (
            self.ensemble_weights['mini_xception'] * predictions['mini_xception'] +
            self.ensemble_weights['mobilenetv2'] * predictions['mobilenetv2'] +
            self.ensemble_weights['efficientnetb0'] * predictions['efficientnetb0']
        )
        
        return ensemble_pred, predictions
    
    def evaluate_models(self, X_test, y_test) -> Dict:
        """Evaluate individual models and ensemble"""
        print("Evaluating models...")
        
        # Get predictions
        ensemble_pred, individual_preds = self.create_ensemble_prediction(X_test)
        
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
    
    def save_models(self, save_path: str = "ensemble_learning/results"):
        """Save trained models"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving models to {save_path}...")
        
        # Save individual models
        for model_name, model in self.models.items():
            model.save(save_path / f"{model_name}_ensemble.h5")
        
        # Save ensemble weights
        with open(save_path / "ensemble_weights.json", 'w') as f:
            json.dump(self.ensemble_weights, f, indent=2)
        
        # Save results
        if self.results:
            # Convert numpy arrays to lists for JSON serialization
            results_json = {}
            for key, value in self.results.items():
                results_json[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            
            with open(save_path / "evaluation_results.json", 'w') as f:
                json.dump(results_json, f, indent=2)
        
        print("Models and results saved successfully!")
    
    def create_comparison_visualization(self):
        """Create comparison visualization of all models"""
        if not self.results:
            print("No results available for visualization")
            return
        
        # Prepare data for plotting
        models = list(self.results.keys())
        metrics = ['accuracy', 'macro_f1', 'precision', 'recall']
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            values = [self.results[model][metric] for model in models]
            bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('ensemble_learning/results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparison visualization saved!")

def main():
    """Main training pipeline"""
    print("=== Ensemble Learning for Emotion Recognition ===")
    
    # Initialize ensemble trainer
    trainer = EnsembleEmotionRecognizer()
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_fer2013_data()
    
    # Train individual models
    trainer.train_individual_models(X_train, y_train, X_val, y_val, epochs=30)
    
    # Evaluate models
    results = trainer.evaluate_models(X_test, y_test)
    
    # Print results
    print("\n=== EVALUATION RESULTS ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
    
    # Save models and results
    trainer.save_models()
    
    # Create visualization
    trainer.create_comparison_visualization()
    
    print("\n=== ENSEMBLE TRAINING COMPLETED ===")

if __name__ == "__main__":
    main()
