"""
Train VGG16 and VGG19 models separately for ensemble integration
Uses existing data preprocessing and adds VGG models to ensemble
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG16, VGG19
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class VGGTrainer:
    def __init__(self, data_path="../fer2013_project/data/fer2013.csv"):
        self.data_path = Path(data_path)
        self.models = {}
        
        # Emotion mapping
        self.emotion_map = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        
    def load_and_preprocess_data(self):
        """Load and preprocess FER2013 data for VGG models"""
        print("Loading and preprocessing FER2013 data for VGG models...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}")
        
        # Convert pixel strings to arrays
        images = []
        labels = []
        
        for idx, row in df.iterrows():
            if idx % 5000 == 0:
                print(f"Processing {idx}/{len(df)} samples...")
            
            # Convert pixel string to array
            pixels = np.array(row['pixels'].split(), dtype='uint8')
            image = pixels.reshape(48, 48)
            
            # Convert to 3-channel for VGG models
            image_rgb = np.stack([image, image, image], axis=-1)
            
            images.append(image_rgb)
            labels.append(row['emotion'])
        
        images = np.array(images, dtype='float32')
        labels = np.array(labels)
        
        # Normalize images
        images = images / 255.0
        
        # Encode labels
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        labels_categorical = tf.keras.utils.to_categorical(labels_encoded, 7)
        
        # Split data
        train_mask = df['Usage'] == 'Training'
        test_mask = df['Usage'] == 'PrivateTest'
        
        X_train_full = images[train_mask]
        y_train_full = labels_categorical[train_mask]
        X_test = images[test_mask]
        y_test = labels_categorical[test_mask]
        
        # Create validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.2,
            random_state=42,
            stratify=np.argmax(y_train_full, axis=1)
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_augmentation(self):
        """Create data augmentation pipeline for VGG models"""
        return keras.Sequential([
            layers.RandomRotation(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
            layers.RandomFlip("horizontal"),
            layers.RandomContrast(0.1),
        ])
    
    def create_vgg16_model(self, input_shape=(48, 48, 3)):
        """Create VGG16 model with custom top for FER"""
        print("Creating VGG16 model...")
        
        # Load VGG16 without top layers
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze early layers (first 10 layers)
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        
        # Add custom classifier
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(7, activation='softmax')(x)
        
        model = keras.Model(base_model.input, outputs)
        return model
    
    def create_vgg19_model(self, input_shape=(48, 48, 3)):
        """Create VGG19 model with custom top for FER"""
        print("Creating VGG19 model...")
        
        # Load VGG19 without top layers
        base_model = VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze early layers (first 10 layers)
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        
        # Add custom classifier
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(7, activation='softmax')(x)
        
        model = keras.Model(base_model.input, outputs)
        return model
    
    def train_vgg_model(self, model, model_name, X_train, y_train, X_val, y_val):
        """Train VGG model with optimized configuration"""
        print(f"\nTraining {model_name}...")
        
        # Compile model with lower learning rate for transfer learning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR for transfer learning
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,  # More patience for transfer learning
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-8
            ),
            callbacks.ModelCheckpoint(
                filepath=f'vgg_models/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Data augmentation
        data_augmentation = self.create_data_augmentation()
        
        # Training with more epochs for transfer learning
        history = model.fit(
            data_augmentation(X_train),
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,  # More epochs for transfer learning
            batch_size=16,  # Smaller batch size for VGG
            callbacks=callbacks_list,
            verbose=1
        )
        
        return model, history
    
    def evaluate_vgg_model(self, model, model_name, X_test, y_test):
        """Evaluate VGG model"""
        print(f"Evaluating {model_name}...")
        
        # Get predictions
        pred = model.predict(X_test, verbose=0)
        pred_labels = np.argmax(pred, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        
        print(f"{model_name} test accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': pred_labels,
            'true_labels': true_labels,
            'probabilities': pred
        }
    
    def create_vgg_visualizations(self, results):
        """Create visualizations for VGG models"""
        print("Creating VGG model visualizations...")
        
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        ax = axes[0, 0]
        colors = ['#E74C3C', '#9B59B6']  # Red for VGG16, Purple for VGG19
        bars = ax.bar(models, accuracies, color=colors)
        ax.set_title('VGG Models Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Confusion matrix for VGG16
        ax = axes[0, 1]
        cm = confusion_matrix(results['VGG16']['true_labels'], 
                             results['VGG16']['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
        ax.set_title('VGG16 Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        # Confusion matrix for VGG19
        ax = axes[1, 0]
        cm = confusion_matrix(results['VGG19']['true_labels'], 
                             results['VGG19']['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)
        ax.set_title('VGG19 Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        # Per-class accuracy comparison
        ax = axes[1, 1]
        emotions = list(self.emotion_map.values())
        
        vgg16_per_class = []
        vgg19_per_class = []
        
        for i in range(7):
            # VGG16
            mask = results['VGG16']['true_labels'] == i
            if mask.sum() > 0:
                class_acc = (results['VGG16']['predictions'][mask] == i).sum() / mask.sum()
                vgg16_per_class.append(class_acc)
            else:
                vgg16_per_class.append(0)
            
            # VGG19
            mask = results['VGG19']['true_labels'] == i
            if mask.sum() > 0:
                class_acc = (results['VGG19']['predictions'][mask] == i).sum() / mask.sum()
                vgg19_per_class.append(class_acc)
            else:
                vgg19_per_class.append(0)
        
        x = np.arange(len(emotions))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, vgg16_per_class, width, label='VGG16', color='#E74C3C', alpha=0.7)
        bars2 = ax.bar(x + width/2, vgg19_per_class, width, label='VGG19', color='#9B59B6', alpha=0.7)
        
        ax.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('Emotion Class', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('vgg_models_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ VGG visualizations created: vgg_models_results.png")
    
    def run_vgg_training(self):
        """Run complete VGG training pipeline"""
        print("=" * 80)
        print("TRAINING VGG16 AND VGG19 MODELS FOR ENSEMBLE")
        print("=" * 80)
        
        # Create directories
        Path("vgg_models").mkdir(exist_ok=True)
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess_data()
        
        # Create VGG models
        vgg16_model = self.create_vgg16_model()
        vgg19_model = self.create_vgg19_model()
        
        # Train VGG16
        print(f"\n{'='*50}")
        print("TRAINING VGG16")
        print(f"{'='*50}")
        trained_vgg16, history_vgg16 = self.train_vgg_model(
            vgg16_model, 'VGG16', X_train, y_train, X_val, y_val
        )
        
        # Train VGG19
        print(f"\n{'='*50}")
        print("TRAINING VGG19")
        print(f"{'='*50}")
        trained_vgg19, history_vgg19 = self.train_vgg_model(
            vgg19_model, 'VGG19', X_train, y_train, X_val, y_val
        )
        
        # Evaluate models
        results = {}
        results['VGG16'] = self.evaluate_vgg_model(trained_vgg16, 'VGG16', X_test, y_test)
        results['VGG19'] = self.evaluate_vgg_model(trained_vgg19, 'VGG19', X_test, y_test)
        
        # Create visualizations
        self.create_vgg_visualizations(results)
        
        # Save results
        self.save_vgg_results(results)
        
        print("\n" + "=" * 80)
        print("VGG TRAINING COMPLETED!")
        print("=" * 80)
        
        return results, trained_vgg16, trained_vgg19
    
    def save_vgg_results(self, results):
        """Save VGG results to file"""
        print("\nSaving VGG results...")
        
        # Create summary
        summary = {
            'vgg_models_results': {
                'VGG16': {'accuracy': results['VGG16']['accuracy']},
                'VGG19': {'accuracy': results['VGG19']['accuracy']},
                'best_vgg': max(results.keys(), key=lambda x: results[x]['accuracy']),
                'best_accuracy': max([results[model]['accuracy'] for model in results.keys()])
            }
        }
        
        # Save to file
        with open('vgg_models_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✓ VGG results saved to vgg_models_results.json")
        
        # Print summary
        print("\n" + "="*60)
        print("VGG MODELS RESULTS SUMMARY")
        print("="*60)
        
        for name, result in results.items():
            print(f"{name:10}: {result['accuracy']:.4f}")
        
        best_vgg = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print(f"\nBest VGG Model: {best_vgg} ({results[best_vgg]['accuracy']:.4f})")
        print("="*60)

def main():
    """Run VGG training pipeline"""
    trainer = VGGTrainer()
    results, vgg16_model, vgg19_model = trainer.run_vgg_training()
    
    print("\n🎉 VGG16 and VGG19 training completed!")
    print("Models saved in 'vgg_models/' directory")
    print("Check 'vgg_models_results.png' for visualizations")
    print("Check 'vgg_models_results.json' for detailed results")
    
    return results, vgg16_model, vgg19_model

if __name__ == "__main__":
    main()
