"""
Train VGG16 model separately for emotion recognition
Focused training to see VGG16 accuracy on FER2013
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class VGG16Trainer:
    def __init__(self, data_path="../fer2013_project/data/fer2013.csv"):
        self.data_path = Path(data_path)
        self.model = None
        
        # Emotion mapping
        self.emotion_map = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        
    def load_and_preprocess_data(self):
        """Load and preprocess FER2013 data for VGG16"""
        print("Loading and preprocessing FER2013 data for VGG16...")
        
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
            
            # Convert to 3-channel for VGG16
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
        """Create data augmentation pipeline for VGG16"""
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
        
        print(f"Total layers in VGG16: {len(base_model.layers)}")
        print(f"Frozen layers: {len([l for l in base_model.layers if not l.trainable])}")
        print(f"Trainable layers: {len([l for l in base_model.layers if l.trainable])}")
        
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
        
        # Print model summary
        print(f"\nVGG16 Model Summary:")
        print(f"Total parameters: {model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        return model
    
    def train_vgg16_model(self, model, X_train, y_train, X_val, y_val):
        """Train VGG16 model with optimized configuration"""
        print(f"\nTraining VGG16 model...")
        
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
                patience=15,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-8
            ),
            callbacks.ModelCheckpoint(
                filepath='vgg16_best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Data augmentation
        data_augmentation = self.create_data_augmentation()
        
        # Training
        print("Starting VGG16 training...")
        history = model.fit(
            data_augmentation(X_train),
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Reasonable number of epochs
            batch_size=16,  # Smaller batch size for VGG
            callbacks=callbacks_list,
            verbose=1
        )
        
        return model, history
    
    def evaluate_vgg16_model(self, model, X_test, y_test):
        """Evaluate VGG16 model"""
        print(f"\nEvaluating VGG16 model...")
        
        # Get predictions
        pred = model.predict(X_test, verbose=0)
        pred_labels = np.argmax(pred, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        
        print(f"VGG16 test accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(true_labels, pred_labels, 
                                  target_names=list(self.emotion_map.values())))
        
        return {
            'accuracy': accuracy,
            'predictions': pred_labels,
            'true_labels': true_labels,
            'probabilities': pred
        }
    
    def create_vgg16_visualizations(self, results, history):
        """Create visualizations for VGG16 model"""
        print("Creating VGG16 visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training history
        ax = axes[0, 0]
        ax.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
        ax.set_title('VGG16 Training History', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss history
        ax = axes[0, 1]
        ax.plot(history.history['loss'], label='Training Loss', color='blue')
        ax.plot(history.history['val_loss'], label='Validation Loss', color='red')
        ax.set_title('VGG16 Loss History', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Confusion matrix
        ax = axes[1, 0]
        cm = confusion_matrix(results['true_labels'], results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('VGG16 Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        # Per-class accuracy
        ax = axes[1, 1]
        emotions = list(self.emotion_map.values())
        per_class_acc = []
        for i in range(7):
            mask = results['true_labels'] == i
            if mask.sum() > 0:
                class_acc = (results['predictions'][mask] == i).sum() / mask.sum()
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0)
        
        bars = ax.bar(emotions, per_class_acc, color='lightblue')
        ax.set_title('VGG16 Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, per_class_acc):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('vgg16_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ VGG16 visualizations created: vgg16_results.png")
    
    def save_vgg16_results(self, results, history):
        """Save VGG16 results to file"""
        print("\nSaving VGG16 results...")
        
        # Create summary
        summary = {
            'vgg16_results': {
                'test_accuracy': results['accuracy'],
                'training_epochs': len(history.history['accuracy']),
                'best_val_accuracy': max(history.history['val_accuracy']),
                'final_train_accuracy': history.history['accuracy'][-1],
                'final_val_accuracy': history.history['val_accuracy'][-1],
                'model_parameters': self.model.count_params(),
                'emotion_classes': list(self.emotion_map.values())
            }
        }
        
        # Save to file
        with open('vgg16_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✓ VGG16 results saved to vgg16_results.json")
        
        # Print summary
        print("\n" + "="*60)
        print("VGG16 RESULTS SUMMARY")
        print("="*60)
        print(f"Test Accuracy:        {results['accuracy']:.4f}")
        print(f"Best Val Accuracy:    {max(history.history['val_accuracy']):.4f}")
        print(f"Training Epochs:      {len(history.history['accuracy'])}")
        print(f"Model Parameters:     {self.model.count_params():,}")
        print("="*60)
    
    def run_vgg16_training(self):
        """Run complete VGG16 training pipeline"""
        print("=" * 80)
        print("TRAINING VGG16 MODEL FOR EMOTION RECOGNITION")
        print("=" * 80)
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess_data()
        
        # Create VGG16 model
        self.model = self.create_vgg16_model()
        
        # Train VGG16
        print(f"\n{'='*50}")
        print("TRAINING VGG16")
        print(f"{'='*50}")
        trained_model, history = self.train_vgg16_model(
            self.model, X_train, y_train, X_val, y_val
        )
        
        # Evaluate VGG16
        results = self.evaluate_vgg16_model(trained_model, X_test, y_test)
        
        # Create visualizations
        self.create_vgg16_visualizations(results, history)
        
        # Save results
        self.save_vgg16_results(results, history)
        
        print("\n" + "=" * 80)
        print("VGG16 TRAINING COMPLETED!")
        print("=" * 80)
        
        return results, history

def main():
    """Run VGG16 training pipeline"""
    trainer = VGG16Trainer()
    results, history = trainer.run_vgg16_training()
    
    print("\n🎉 VGG16 training completed!")
    print("Model saved as 'vgg16_best_model.h5'")
    print("Check 'vgg16_results.png' for visualizations")
    print("Check 'vgg16_results.json' for detailed results")
    
    return results, history

if __name__ == "__main__":
    main()
