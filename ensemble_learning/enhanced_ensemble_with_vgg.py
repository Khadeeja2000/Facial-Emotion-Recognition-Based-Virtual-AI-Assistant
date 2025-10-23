"""
Enhanced Ensemble Learning with VGG16/19 Integration
Includes VGG16, VGG19, and optimized configurations for better accuracy
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

class EnhancedEnsembleWithVGG:
    def __init__(self, data_path="../fer2013_project/data/fer2013.csv"):
        self.data_path = Path(data_path)
        self.models = {}
        self.ensemble_weights = {}
        self.results = {}
        
        # Emotion mapping
        self.emotion_map = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        
    def load_and_preprocess_data(self):
        """Load and preprocess FER2013 data with enhanced augmentation"""
        print("Loading and preprocessing FER2013 data...")
        
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
        """Create data augmentation pipeline"""
        return keras.Sequential([
            layers.RandomRotation(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
            layers.RandomFlip("horizontal"),
            layers.RandomContrast(0.1),
        ])
    
    def create_mini_xception(self, input_shape=(48, 48, 3)):
        """Create enhanced Mini-XCEPTION model"""
        print("Creating enhanced Mini-XCEPTION model...")
        
        inputs = keras.Input(shape=input_shape)
        
        # Entry flow
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # First residual block - no pooling
        residual = x
        x = layers.SeparableConv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        
        # Second residual block with pooling
        residual = x
        x = layers.SeparableConv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        
        # Adjust residual for pooling
        residual = layers.MaxPooling2D(3, strides=2, padding="same")(residual)
        residual = layers.Conv2D(128, 1, padding="same")(residual)
        x = layers.add([x, residual])
        
        # Exit flow
        x = layers.SeparableConv2D(256, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(256, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Classifier
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(7, activation="softmax")(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def create_vgg16_model(self, input_shape=(48, 48, 3)):
        """Create VGG16 model with custom top for FER"""
        print("Creating VGG16 model...")
        
        # Load VGG16 without top layers
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze early layers
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
        
        # Freeze early layers
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
    
    def create_enhanced_mobilenet(self, input_shape=(48, 48, 3)):
        """Create enhanced MobileNetV2 model"""
        print("Creating enhanced MobileNetV2 model...")
        
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
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
    
    def create_enhanced_efficientnet(self, input_shape=(48, 48, 3)):
        """Create enhanced EfficientNetB0 model"""
        print("Creating enhanced EfficientNetB0 model...")
        
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-8]:
            layer.trainable = False
        
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
    
    def train_model(self, model, model_name, X_train, y_train, X_val, y_val):
        """Train individual model with enhanced configuration"""
        print(f"\nTraining {model_name}...")
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
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
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                filepath=f'enhanced_models/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Data augmentation
        data_augmentation = self.create_data_augmentation()
        
        # Training
        history = model.fit(
            data_augmentation(X_train),
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return model, history
    
    def create_enhanced_ensemble(self, models, X_val, y_val):
        """Create enhanced ensemble with VGG models"""
        print("\nCreating enhanced ensemble...")
        
        # Get predictions from all models
        predictions = {}
        for name, model in models.items():
            print(f"Getting predictions from {name}...")
            pred = model.predict(X_val)
            predictions[name] = pred
        
        # Calculate individual accuracies for weighting
        accuracies = {}
        for name, pred in predictions.items():
            pred_labels = np.argmax(pred, axis=1)
            true_labels = np.argmax(y_val, axis=1)
            acc = accuracy_score(true_labels, pred_labels)
            accuracies[name] = acc
            print(f"{name} validation accuracy: {acc:.4f}")
        
        # Enhanced weighting strategy
        total_acc = sum(accuracies.values())
        weights = {name: acc/total_acc for name, acc in accuracies.items()}
        
        print(f"\nEnsemble weights: {weights}")
        
        # Create ensemble prediction function
        def ensemble_predict(X):
            predictions = {}
            for name, model in models.items():
                pred = model.predict(X, verbose=0)
                predictions[name] = pred
            
            # Weighted ensemble
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            for name, pred in predictions.items():
                ensemble_pred += weights[name] * pred
            
            return ensemble_pred
        
        return ensemble_predict, weights
    
    def evaluate_models(self, models, ensemble_predict, X_test, y_test):
        """Evaluate all models and ensemble"""
        print("\nEvaluating models...")
        
        results = {}
        
        # Individual models
        for name, model in models.items():
            print(f"Evaluating {name}...")
            pred = model.predict(X_test, verbose=0)
            pred_labels = np.argmax(pred, axis=1)
            true_labels = np.argmax(y_test, axis=1)
            
            acc = accuracy_score(true_labels, pred_labels)
            results[name] = {
                'accuracy': acc,
                'predictions': pred_labels,
                'true_labels': true_labels
            }
            print(f"{name} test accuracy: {acc:.4f}")
        
        # Ensemble
        print("Evaluating ensemble...")
        ensemble_pred = ensemble_predict(X_test)
        ensemble_labels = np.argmax(ensemble_pred, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        
        ensemble_acc = accuracy_score(true_labels, ensemble_labels)
        results['ensemble'] = {
            'accuracy': ensemble_acc,
            'predictions': ensemble_labels,
            'true_labels': true_labels
        }
        
        print(f"Ensemble test accuracy: {ensemble_acc:.4f}")
        
        return results
    
    def create_visualizations(self, results):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        # Model comparison
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        ax = axes[0, 0]
        bars = ax.bar(models, accuracies, color=['#4472C4', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6', '#E67E22'])
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Confusion matrix for ensemble
        ax = axes[0, 1]
        cm = confusion_matrix(results['ensemble']['true_labels'], 
                             results['ensemble']['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Ensemble Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        # Per-class accuracy
        ax = axes[1, 0]
        emotions = list(self.emotion_map.values())
        per_class_acc = []
        for i in range(7):
            mask = results['ensemble']['true_labels'] == i
            if mask.sum() > 0:
                class_acc = (results['ensemble']['predictions'][mask] == i).sum() / mask.sum()
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0)
        
        bars = ax.bar(emotions, per_class_acc, color='skyblue')
        ax.set_title('Per-Class Accuracy (Ensemble)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, acc in zip(bars, per_class_acc):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Model performance comparison
        ax = axes[1, 1]
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, accuracies, color=['#4472C4', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6', '#E67E22'])
        ax.set_title('Enhanced Ensemble Performance', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('enhanced_ensemble_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Visualizations created: enhanced_ensemble_results.png")
    
    def run_complete_pipeline(self):
        """Run the complete enhanced ensemble pipeline"""
        print("=" * 80)
        print("ENHANCED ENSEMBLE LEARNING WITH VGG16/19")
        print("=" * 80)
        
        # Create directories
        Path("enhanced_models").mkdir(exist_ok=True)
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess_data()
        
        # Create models
        models = {
            'Mini-XCEPTION': self.create_mini_xception(),
            'VGG16': self.create_vgg16_model(),
            'VGG19': self.create_vgg19_model(),
            'MobileNetV2': self.create_enhanced_mobilenet(),
            'EfficientNetB0': self.create_enhanced_efficientnet()
        }
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            print(f"\n{'='*50}")
            print(f"TRAINING {name.upper()}")
            print(f"{'='*50}")
            
            trained_model, history = self.train_model(
                model, name, X_train, y_train, X_val, y_val
            )
            trained_models[name] = trained_model
        
        # Create ensemble
        ensemble_predict, weights = self.create_enhanced_ensemble(
            trained_models, X_val, y_val
        )
        
        # Evaluate
        results = self.evaluate_models(trained_models, ensemble_predict, X_test, y_test)
        
        # Create visualizations
        self.create_visualizations(results)
        
        # Save results
        self.save_results(results, weights)
        
        print("\n" + "=" * 80)
        print("ENHANCED ENSEMBLE PIPELINE COMPLETED!")
        print("=" * 80)
        
        return results, weights
    
    def save_results(self, results, weights):
        """Save results to file"""
        print("\nSaving results...")
        
        # Create summary
        summary = {
            'enhanced_ensemble_results': {
                'individual_models': {name: {'accuracy': results[name]['accuracy']} 
                                   for name in results.keys() if name != 'ensemble'},
                'ensemble': {'accuracy': results['ensemble']['accuracy']},
                'weights': weights,
                'improvement_over_best_individual': (
                    results['ensemble']['accuracy'] - 
                    max([results[name]['accuracy'] for name in results.keys() if name != 'ensemble'])
                )
            }
        }
        
        # Save to file
        with open('enhanced_ensemble_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✓ Results saved to enhanced_ensemble_results.json")
        
        # Print summary
        print("\n" + "="*60)
        print("ENHANCED ENSEMBLE RESULTS SUMMARY")
        print("="*60)
        
        for name, result in results.items():
            print(f"{name:15}: {result['accuracy']:.4f}")
        
        best_individual = max([results[name]['accuracy'] for name in results.keys() if name != 'ensemble'])
        improvement = results['ensemble']['accuracy'] - best_individual
        
        print(f"\nBest Individual Model: {best_individual:.4f}")
        print(f"Enhanced Ensemble:     {results['ensemble']['accuracy']:.4f}")
        print(f"Improvement:          +{improvement:.4f} ({improvement*100:.2f}%)")
        print("="*60)

def main():
    """Run the enhanced ensemble pipeline"""
    ensemble = EnhancedEnsembleWithVGG()
    results, weights = ensemble.run_complete_pipeline()
    
    print("\n🎉 Enhanced ensemble with VGG16/19 completed!")
    print("Check 'enhanced_ensemble_results.png' for visualizations")
    print("Check 'enhanced_ensemble_results.json' for detailed results")

if __name__ == "__main__":
    main()
