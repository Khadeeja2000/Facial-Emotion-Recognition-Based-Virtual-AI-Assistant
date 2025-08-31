"""
Emotion Classification Model Training Script

This script trains a CNN model for emotion classification using the FER2013 dataset.
It includes data augmentation, callbacks for monitoring, and model checkpointing.
"""

import os
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from load_and_process import load_fer2013, preprocess_input
from models.cnn import mini_XCEPTION

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10000
INPUT_SHAPE = (48, 48, 1)
VALIDATION_SPLIT = 0.2
VERBOSE = 1
NUM_CLASSES = 7
PATIENCE = 50
BASE_PATH = 'models/'

def create_data_generator():
    """Create data generator with augmentation."""
    return ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

def create_callbacks():
    """Create training callbacks."""
    # Ensure models directory exists
    os.makedirs(BASE_PATH, exist_ok=True)
    
    # Logging callback
    log_file_path = os.path.join(BASE_PATH, 'emotion_training.log')
    csv_logger = CSVLogger(log_file_path, append=False)
    
    # Early stopping callback
    early_stop = EarlyStopping('val_loss', patience=PATIENCE)
    
    # Learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(
        'val_loss', 
        factor=0.1,
        patience=int(PATIENCE/4), 
        verbose=1
    )
    
    # Model checkpointing callback
    trained_models_path = os.path.join(BASE_PATH, 'mini_XCEPTION')
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(
        model_names, 
        'val_loss', 
        verbose=1,
        save_best_only=True
    )
    
    return [model_checkpoint, csv_logger, early_stop, reduce_lr]

def prepare_data():
    """Load and prepare training data."""
    print("Loading FER2013 dataset...")
    faces, emotions = load_fer2013()
    
    print("Preprocessing input data...")
    faces = preprocess_input(faces)
    
    print(f"Dataset shape: {faces.shape}")
    print(f"Number of samples: {len(faces)}")
    print(f"Number of classes: {emotions.shape[1]}")
    
    return faces, emotions

def train_model():
    """Main training function."""
    print("Starting emotion classification model training...")
    
    # Create data generator
    data_generator = create_data_generator()
    
    # Create and compile model
    print("Creating model...")
    model = mini_XCEPTION(INPUT_SHAPE, NUM_CLASSES)
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Prepare data
    faces, emotions = prepare_data()
    
    # Split data
    print("Splitting data into train/validation sets...")
    xtrain, xtest, ytrain, ytest = train_test_split(
        faces, emotions, 
        test_size=VALIDATION_SPLIT, 
        shuffle=True
    )
    
    print(f"Training samples: {len(xtrain)}")
    print(f"Validation samples: {len(xtest)}")
    
    # Train model
    print("Starting training...")
    model.fit(
        data_generator.flow(xtrain, ytrain, batch_size=BATCH_SIZE),
        steps_per_epoch=len(xtrain) // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=VERBOSE,
        callbacks=callbacks,
        validation_data=(xtest, ytest)
    )
    
    print("Training completed!")

if __name__ == "__main__":
    train_model()
