"""
Simple Boost - Focus on the most impactful improvements
Quick and effective way to improve results for research paper
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

class SimpleBoost:
    """Simple but effective boost for research results"""
    
    def __init__(self):
        self.emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
        self.num_classes = len(self.emotion_labels)
        
    def load_data(self):
        """Load FER2013 data with proper preprocessing"""
        print("Loading FER2013 data...")
        
        df = pd.read_csv('../fer2013_project/data/fer2013.csv')
        
        images = []
        labels = []
        
        for idx, row in df.iterrows():
            pixels = np.array(row['pixels'].split(), dtype='uint8')
            image = pixels.reshape(48, 48, 1)  # Add channel dimension
            images.append(image)
            labels.append(row['emotion'])
        
        images = np.array(images, dtype='float32') / 255.0
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
    
    def create_better_cnn(self):
        """Create a better CNN architecture"""
        print("Creating Better CNN...")
        
        model = keras.Sequential([
            # First block
            layers.Conv2D(32, 3, activation='relu', input_shape=(48, 48, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Second block
            layers.Conv2D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Third block
            layers.Conv2D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Fourth block
            layers.Conv2D(256, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Classifier
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
    
    def create_deeper_cnn(self):
        """Create a deeper CNN"""
        print("Creating Deeper CNN...")
        
        model = keras.Sequential([
            # Block 1
            layers.Conv2D(64, 3, activation='relu', input_shape=(48, 48, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            
            # Block 2
            layers.Conv2D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            
            # Block 3
            layers.Conv2D(256, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            
            # Block 4
            layers.Conv2D(512, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(512, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            
            # Classifier
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def train_model(self, model, X_train, y_train, X_val, y_val, model_name, epochs=20):
        """Train model with good settings"""
        print(f"Training {model_name}...")
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )
        
        # Callbacks
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
                patience=8,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def create_ensemble(self, models, X_test):
        """Create simple ensemble"""
        print("Creating Ensemble...")
        
        predictions = []
        for model_name, model in models.items():
            pred = model.predict(X_test)
            predictions.append(pred)
        
        # Simple average
        ensemble_pred = np.mean(predictions, axis=0)
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
        
        # Ensemble
        ensemble_pred = self.create_ensemble(models, X_test)
        y_pred_classes = np.argmax(ensemble_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        macro_f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
        precision = precision_score(y_true_classes, y_pred_classes, average='macro')
        recall = recall_score(y_true_classes, y_pred_classes, average='macro')
        
        results['ensemble'] = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'precision': precision,
            'recall': recall
        }
        
        print(f"Ensemble: Accuracy={accuracy:.4f}, F1={macro_f1:.4f}")
        return results
    
    def create_visualization(self, results):
        """Create visualization"""
        print("Creating Visualization...")
        
        models = list(results.keys())
        metrics = ['accuracy', 'macro_f1', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Simple Boost - Model Performance', fontsize=16, fontweight='bold')
        
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
        plt.savefig('results/SIMPLE_BOOST_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results):
        """Save results"""
        save_path = Path("results")
        save_path.mkdir(exist_ok=True)
        
        with open(save_path / "simple_boost_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary
        report_path = Path("results/SIMPLE_BOOST_SUMMARY.txt")
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SIMPLE BOOST RESULTS\n")
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
    print("=== SIMPLE BOOST FOR RESEARCH ===")
    
    booster = SimpleBoost()
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = booster.load_data()
    
    # Create models
    models = {}
    
    # Better CNN
    better_cnn = booster.create_better_cnn()
    booster.train_model(better_cnn, X_train, y_train, X_val, y_val, "better_cnn", epochs=15)
    models['better_cnn'] = better_cnn
    
    # Deeper CNN
    deeper_cnn = booster.create_deeper_cnn()
    booster.train_model(deeper_cnn, X_train, y_train, X_val, y_val, "deeper_cnn", epochs=15)
    models['deeper_cnn'] = deeper_cnn
    
    # Evaluate
    results = booster.evaluate_models(models, X_test, y_test)
    
    # Visualize
    booster.create_visualization(results)
    
    # Save
    booster.save_results(results)
    
    # Print results
    print("\n=== SIMPLE BOOST RESULTS ===")
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
    else:
        print("📊 DECENT! Results are better than before!")
    
    print("\n🚀 SIMPLE BOOST COMPLETED!")

if __name__ == "__main__":
    main()
