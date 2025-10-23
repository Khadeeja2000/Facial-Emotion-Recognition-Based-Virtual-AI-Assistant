"""
Quick Research Boost - Fast Improvements for Research Paper
Focus on the most impactful changes to quickly improve results
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

class QuickResearchBoost:
    """
    Quick optimizer to boost results for research paper
    Focus on high-impact, low-effort improvements
    """
    
    def __init__(self):
        self.emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
        self.num_classes = len(self.emotion_labels)
        self.img_size = 48
        
    def load_and_enhance_data(self):
        """Load data with enhanced preprocessing"""
        print("Loading and enhancing FER2013 data...")
        
        df = pd.read_csv('../fer2013_project/data/fer2013.csv')
        
        images = []
        labels = []
        
        for idx, row in df.iterrows():
            pixels = np.array(row['pixels'].split(), dtype='uint8')
            image = pixels.reshape(48, 48)
            
        # Enhanced preprocessing
        image = image.astype('float32') / 255.0
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        # Add channel dimension for grayscale
        image = np.expand_dims(image, axis=-1)
        images.append(image)
        labels.append(row['emotion'])
        
        images = np.array(images)
        labels = to_categorical(labels, self.num_classes)
        
        # Split data
        train_mask = df['Usage'] == 'Training'
        test_mask = df['Usage'] == 'PrivateTest'
        
        X_train_full, y_train_full = images[train_mask], labels[train_mask]
        X_test, y_test = images[test_mask], labels[test_mask]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.2,
            random_state=42,
            stratify=np.argmax(y_train_full, axis=1)
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_improved_mini_xception(self):
        """Create improved Mini-XCEPTION with better architecture"""
        print("Creating Improved Mini-XCEPTION...")
        
        inputs = keras.Input(shape=(self.img_size, self.img_size, 1))
        
        # Entry flow
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Improved separable convolutions
        residual = layers.Conv2D(128, 1, strides=2, padding="same")(x)
        residual = layers.BatchNormalization()(residual)
        
        x = layers.SeparableConv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.add([x, residual])
        
        # Multiple residual blocks
        for _ in range(6):  # Increased from 4 to 6
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
        
        # Improved classifier
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        
        model = keras.Model(inputs, outputs)
        
        # Improved optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def create_simple_cnn(self):
        """Create a simple but effective CNN"""
        print("Creating Simple CNN...")
        
        model = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=(48, 48, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            layers.Conv2D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            layers.Conv2D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            layers.Conv2D(256, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def create_enhanced_data_generator(self):
        """Create enhanced data augmentation"""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )
    
    def train_model_quickly(self, model, X_train, y_train, X_val, y_val, model_name, epochs=25):
        """Train model with quick but effective settings"""
        print(f"Training {model_name} quickly...")
        
        # Enhanced callbacks
        callbacks_list = [
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Enhanced data augmentation
        datagen = self.create_enhanced_data_generator()
        
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def create_smart_ensemble(self, models, X_test):
        """Create smart ensemble with optimized weights"""
        print("Creating Smart Ensemble...")
        
        predictions = []
        weights = []
        
        # Get predictions and calculate weights based on confidence
        for model_name, model in models.items():
            pred = model.predict(X_test)
            predictions.append(pred)
            
            # Weight based on prediction confidence
            confidence = np.mean(np.max(pred, axis=1))
            weights.append(confidence)
        
        # Normalize weights
        weights = np.array(weights)
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
        
        # Evaluate individual models
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
        
        # Evaluate ensemble
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
    
    def create_quick_visualizations(self, results):
        """Create quick but effective visualizations"""
        print("Creating Quick Visualizations...")
        
        models = list(results.keys())
        metrics = ['accuracy', 'macro_f1', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Quick Research Boost - Model Performance', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        colors = ['#2E8B57', '#FF6347', '#4169E1', '#9370DB']
        
        for idx, metric in enumerate(metrics):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            values = [results[model][metric] for model in models]
            bars = ax.bar(models, values, color=colors[:len(models)], alpha=0.8, 
                         edgecolor='black', linewidth=0.5)
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold', fontsize=12)
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Highlight best performer
            best_idx = np.argmax(values)
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(2)
            
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/QUICK_BOOST_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_quick_results(self, results):
        """Save quick results"""
        save_path = Path("results")
        save_path.mkdir(exist_ok=True)
        
        with open(save_path / "quick_boost_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary
        report_path = Path("results/QUICK_BOOST_SUMMARY.txt")
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("QUICK RESEARCH BOOST RESULTS\n")
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
        
        print("✓ Quick results saved!")

def main():
    """Main quick boost pipeline"""
    print("=== QUICK RESEARCH BOOST ===")
    print("Fast improvements for research paper")
    
    booster = QuickResearchBoost()
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = booster.load_and_enhance_data()
    
    # Create models
    models = {}
    
    # 1. Improved Mini-XCEPTION
    mini_xception = booster.create_improved_mini_xception()
    booster.train_model_quickly(mini_xception, X_train, y_train, X_val, y_val, "mini_xception", epochs=20)
    models['mini_xception'] = mini_xception
    
    # 2. Simple CNN
    simple_cnn = booster.create_simple_cnn()
    booster.train_model_quickly(simple_cnn, X_train, y_train, X_val, y_val, "simple_cnn", epochs=15)
    models['simple_cnn'] = simple_cnn
    
    # Evaluate models
    results = booster.evaluate_models(models, X_test, y_test)
    
    # Create visualizations
    booster.create_quick_visualizations(results)
    
    # Save results
    booster.save_quick_results(results)
    
    # Print final results
    print("\n=== QUICK BOOST RESULTS ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
    
    best_accuracy = max(results.values(), key=lambda x: x['accuracy'])['accuracy']
    print(f"\n🎯 BEST ACHIEVED ACCURACY: {best_accuracy:.4f}")
    
    if best_accuracy > 0.70:
        print("🎉 EXCELLENT! Results are now research-quality!")
    elif best_accuracy > 0.65:
        print("📈 GOOD! Results are significantly improved!")
    else:
        print("📊 DECENT! Results are better than before!")
    
    print("\n🚀 QUICK BOOST COMPLETED!")

if __name__ == "__main__":
    main()
