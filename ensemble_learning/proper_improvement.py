"""
Proper Improvement - Build on existing good results
Fix the issues and actually improve performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class ProperImprovement:
    """Proper improvement that builds on good results"""
    
    def __init__(self):
        self.emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
        self.num_classes = len(self.emotion_labels)
        
    def load_data_properly(self):
        """Load data with proper preprocessing (like original)"""
        print("Loading FER2013 data with proper preprocessing...")
        
        df = pd.read_csv('../fer2013_project/data/fer2013.csv')
        
        images = []
        labels = []
        
        for idx, row in df.iterrows():
            pixels = np.array(row['pixels'].split(), dtype='uint8')
            image = pixels.reshape(48, 48, 1)
            # Simple normalization like original
            images.append(image.astype('float32') / 255.0)
            labels.append(row['emotion'])
        
        images = np.array(images)
        labels = to_categorical(labels, self.num_classes)
        
        # Split data
        train_mask = df['Usage'] == 'Training'
        test_mask = df['Usage'] == 'PrivateTest'
        
        X_train_full, y_train_full = images[train_mask], labels[train_mask]
        X_test, y_test = images[test_mask], labels[test_mask]
        
        # Create validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.2,
            random_state=42,
            stratify=np.argmax(y_train_full, axis=1)
        )
        
        print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_enhanced_mini_xception(self):
        """Create enhanced Mini-XCEPTION based on original"""
        print("Creating Enhanced Mini-XCEPTION...")
        
        inputs = keras.Input(shape=(48, 48, 1))
        
        # Entry flow
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Separable convolutions with residual
        residual = layers.Conv2D(128, 1, strides=2, padding="same")(x)
        residual = layers.BatchNormalization()(residual)
        
        x = layers.SeparableConv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.add([x, residual])
        
        # Multiple residual blocks (increased from 4 to 6)
        for _ in range(6):
            residual = x
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(128, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(128, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.add([x, residual])
        
        # Exit flow
        residual = layers.Conv2D(256, 1, strides=2, padding="same")(x)
        residual = layers.BatchNormalization()(residual)
        
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(256, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(256, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.add([x, residual])
        
        # Enhanced classifier
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        
        model = keras.Model(inputs, outputs)
        
        # Better optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def create_improved_cnn(self):
        """Create improved CNN that actually works"""
        print("Creating Improved CNN...")
        
        model = keras.Sequential([
            # Block 1
            layers.Conv2D(32, 3, activation='relu', input_shape=(48, 48, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Enhanced classifier
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def train_model_properly(self, model, X_train, y_train, X_val, y_val, model_name, epochs=30):
        """Train model with proper settings"""
        print(f"Training {model_name} properly...")
        
        # Conservative data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,  # Reduced from 15
            width_shift_range=0.05,  # Reduced from 0.1
            height_shift_range=0.05,  # Reduced from 0.1
            shear_range=0.05,  # Reduced from 0.1
            zoom_range=0.05,  # Reduced from 0.1
            horizontal_flip=True,
            brightness_range=[0.95, 1.05],  # Reduced from 0.9-1.1
            fill_mode='nearest'
        )
        
        # Better callbacks
        callbacks_list = [
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=8,  # Increased patience
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,  # Increased patience
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train with proper settings
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def create_smart_ensemble(self, models, X_test):
        """Create smart ensemble with confidence weighting"""
        print("Creating Smart Ensemble...")
        
        predictions = []
        confidences = []
        
        for model_name, model in models.items():
            pred = model.predict(X_test)
            predictions.append(pred)
            
            # Calculate confidence as max probability
            confidence = np.mean(np.max(pred, axis=1))
            confidences.append(confidence)
        
        # Weight by confidence
        weights = np.array(confidences)
        weights = weights / np.sum(weights)
        
        print(f"Ensemble weights: {weights}")
        
        # Weighted ensemble
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_pred += pred * weight
        
        return ensemble_pred
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models"""
        print("Evaluating Models...")
        
        results = {}
        
        # Individual models
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            accuracy = accuracy_score(y_true_classes, y_pred_classes)
            macro_f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
            precision = precision_score(y_true_classes, y_pred_classes, average='macro')
            recall = recall_score(y_true_classes, y_pred_classes, average='macro')
            
            results[model_name] = {
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'precision': precision,
                'recall': recall
            }
            
            print(f"{model_name}: Accuracy={accuracy:.4f}, F1={macro_f1:.4f}")
        
        # Smart ensemble
        ensemble_pred = self.create_smart_ensemble(models, X_test)
        y_pred_classes = np.argmax(ensemble_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        macro_f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
        precision = precision_score(y_true_classes, y_pred_classes, average='macro')
        recall = recall_score(y_true_classes, y_pred_classes, average='macro')
        
        results['smart_ensemble'] = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'precision': precision,
            'recall': recall
        }
        
        print(f"Smart Ensemble: Accuracy={accuracy:.4f}, F1={macro_f1:.4f}")
        return results
    
    def create_visualization(self, results):
        """Create visualization"""
        print("Creating Visualization...")
        
        models = list(results.keys())
        metrics = ['accuracy', 'macro_f1', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Proper Improvement - Model Performance', fontsize=16, fontweight='bold')
        
        colors = ['#2E8B57', '#FF6347', '#4169E1', '#9370DB']
        
        for idx, metric in enumerate(metrics):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            values = [results[model][metric] for model in models]
            bars = ax.bar(models, values, color=colors[:len(models)], alpha=0.8)
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/PROPER_IMPROVEMENT_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results):
        """Save results"""
        save_path = Path("results")
        save_path.mkdir(exist_ok=True)
        
        with open(save_path / "proper_improvement_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary
        report_path = Path("results/PROPER_IMPROVEMENT_SUMMARY.txt")
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("PROPER IMPROVEMENT RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
            f.write(f"BEST MODEL: {best_model[0].upper()}\n")
            f.write(f"BEST ACCURACY: {best_model[1]['accuracy']:.4f}\n\n")
            
            f.write("ALL RESULTS:\n")
            f.write("-" * 15 + "\n")
            for model_name, metrics in results.items():
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  F1-Score: {metrics['macro_f1']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n\n")
        
        print("✓ Results saved!")

def main():
    """Main pipeline"""
    print("=== PROPER IMPROVEMENT FOR RESEARCH ===")
    print("Building on existing good results, not destroying them!")
    
    improver = ProperImprovement()
    
    # Load data properly
    X_train, X_val, X_test, y_train, y_val, y_test = improver.load_data_properly()
    
    # Create models
    models = {}
    
    # Enhanced Mini-XCEPTION
    enhanced_mini = improver.create_enhanced_mini_xception()
    improver.train_model_properly(enhanced_mini, X_train, y_train, X_val, y_val, "enhanced_mini_xception", epochs=25)
    models['enhanced_mini_xception'] = enhanced_mini
    
    # Improved CNN
    improved_cnn = improver.create_improved_cnn()
    improver.train_model_properly(improved_cnn, X_train, y_train, X_val, y_val, "improved_cnn", epochs=25)
    models['improved_cnn'] = improved_cnn
    
    # Evaluate
    results = improver.evaluate_models(models, X_test, y_test)
    
    # Visualize
    improver.create_visualization(results)
    
    # Save
    improver.save_results(results)
    
    # Print results
    print("\n=== PROPER IMPROVEMENT RESULTS ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
    
    best_accuracy = max(results.values(), key=lambda x: x['accuracy'])['accuracy']
    print(f"\n🎯 BEST ACCURACY: {best_accuracy:.4f}")
    
    if best_accuracy > 0.70:
        print("🎉 EXCELLENT! Research-quality results achieved!")
    elif best_accuracy > 0.65:
        print("📈 GOOD! Significant improvement achieved!")
    elif best_accuracy > 0.60:
        print("✅ DECENT! Results are better than before!")
    else:
        print("❌ STILL NOT GOOD ENOUGH!")
    
    print("\n🚀 PROPER IMPROVEMENT COMPLETED!")

if __name__ == "__main__":
    main()
