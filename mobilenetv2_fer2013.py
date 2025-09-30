#!/usr/bin/env python3
"""
FER2013 Facial Emotion Recognition — Transfer Learning Pipeline (MobileNetV2)
- Preprocessing (FER2013 image folders -> tensors)
- Training phase 1: freeze base (ImageNet) + train classifier head
- Training phase 2: unfreeze top layers (fine-tuning)
- Evaluation (metrics + confusion matrices)
- Live inference from webcam (OpenCV Haar cascade)

Usage examples:
  python mobilenetv2_fer2013.py --mode train --epochs_frozen 10 --epochs_finetune 20
  python mobilenetv2_fer2013.py --mode evaluate --model_path ./mobilenetv2_models/fer_mnv2_best.h5
  python mobilenetv2_fer2013.py --mode live --model_path ./mobilenetv2_models/fer_mnv2_best.h5
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# For nicer logs if installed
try:
    from rich import print
except Exception:
    pass

# -------------------------
# Config / Labels
# -------------------------
EMOTION_LABELS = [
    "Angry",    # 0
    "Disgust",  # 1
    "Fear",     # 2
    "Happy",    # 3
    "Sad",      # 4
    "Surprise", # 5
    "Neutral",  # 6
]
NUM_CLASSES = len(EMOTION_LABELS)

IMG_SIZE_RAW = 48          # raw FER2013 images
IMG_RESIZED = 160          # MobileNetV2 supports 96/128/160/192/224; 160 = good speed/accuracy
INPUT_CHANNELS = 3         # MobileNetV2 expects RGB
RANDOM_SEED = 42

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mnv2_preprocess


def set_seeds(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_fer2013_from_folders():
    """Load FER2013 from image folders structure."""
    data_path = Path("fer2013/fer2013/archive-2")
    train_path = data_path / "train"
    test_path = data_path / "test"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"FER2013 folders not found at {data_path}")
    
    print(f"Loading FER2013 from folders...")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")
    
    def load_images_from_folder(folder_path, subset_name):
        images = []
        labels = []
        
        for emotion_idx, emotion in enumerate(EMOTION_LABELS):
            emotion_path = folder_path / emotion
            if emotion_path.exists():
                image_files = list(emotion_path.glob("*.jpg")) + list(emotion_path.glob("*.png"))
                print(f"  {emotion}: {len(image_files)} images")
                
                for img_file in image_files:
                    # Load image
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize to 48x48 (original FER2013 size)
                        img = cv2.resize(img, (IMG_SIZE_RAW, IMG_SIZE_RAW))
                        # Normalize to [0, 1]
                        img = img.astype('float32') / 255.0
                        # Add channel dimension
                        img = np.expand_dims(img, axis=-1)  # (48, 48, 1)
                        images.append(img)
                        labels.append(emotion_idx)
        
        return np.array(images), np.array(labels)
    
    # Load training and test images
    x_train, y_train = load_images_from_folder(train_path, "train")
    x_test, y_test = load_images_from_folder(test_path, "test")
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    # Split training data into train/validation
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"Loaded FER2013: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def to_rgb_and_resize(img_48x48x1):
    """Map fn for tf.data: grayscale [0,1] -> RGB, resize, MobileNetV2 preprocess."""
    # Convert 1-channel to 3-channel by repeating
    img = tf.tile(img_48x48x1, [1,1,3])                   # (48,48,3)
    img = tf.image.resize(img, (IMG_RESIZED, IMG_RESIZED))# (H,W,3) in [0,1]
    # MobilenetV2 preprocess expects float in [0,255] or [-1,1] depending; use keras preprocessing fn
    img = img * 255.0
    img = mnv2_preprocess(img)                            # scale to [-1,1]
    return img


def make_datasets(x_train, y_train, x_val, y_val, x_test, y_test, batch_size=128):
    """Create tf.data with augmentation on train only; resize & preprocess for MobileNetV2."""
    AUTOTUNE = tf.data.AUTOTUNE

    def augment(img, label):
        # Basic photometric + geometric augs
        img = tf.image.random_flip_left_right(img)
        # jitter (on [0,1] grayscale before RGB conversion is fine)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        # small random crop/zoom
        pad = 4
        img = tf.image.resize_with_pad(img, IMG_SIZE_RAW + pad, IMG_SIZE_RAW + pad)
        img = tf.image.random_crop(img, size=(IMG_SIZE_RAW, IMG_SIZE_RAW, 1))
        return img, label

    def pipe(x, y, training=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(8192, seed=RANDOM_SEED, reshuffle_each_iteration=True)
            ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
        # Convert to RGB, resize, then preprocess for MobileNetV2
        ds = ds.map(lambda img, lbl: (to_rgb_and_resize(img), lbl), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = pipe(x_train, y_train, training=True)
    val_ds   = pipe(x_val,   y_val,   training=False)
    test_ds  = pipe(x_test,  y_test,  training=False)
    return train_ds, val_ds, test_ds


def build_transfer_model(trainable_backbone=False, dropout=0.4):
    """MobileNetV2 backbone (ImageNet weights) + GAP + Dense head."""
    inputs = layers.Input(shape=(IMG_RESIZED, IMG_RESIZED, INPUT_CHANNELS))
    base = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
    base.trainable = trainable_backbone

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inputs, outputs, name="FER_MobileNetV2")
    return model, base


def compile_model(model, lr):
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(model_path, epochs_frozen=10, epochs_finetune=20, batch_size=128,
          lr_frozen=1e-3, lr_finetune=5e-5, unfreeze_at_layer_name="block_13_expand"):
    """
    Two-phase training:
      1) Freeze backbone; train classifier head.
      2) Unfreeze top portion of backbone from `unfreeze_at_layer_name`; fine-tune with low LR.
    """
    set_seeds()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_fer2013_from_folders()
    print(f"[bold]Loaded FER2013[/bold] train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")

    train_ds, val_ds, _ = make_datasets(x_train, y_train, x_val, y_val, x_test, y_test, batch_size)

    # Phase 1: frozen backbone
    model, base = build_transfer_model(trainable_backbone=False, dropout=0.4)
    compile_model(model, lr_frozen)
    model.summary()

    # Create models directory
    models_dir = Path("mobilenetv2_models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / model_path

    callbacks_common = [
        keras.callbacks.ModelCheckpoint(str(model_path), monitor='val_accuracy', save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True),
        keras.callbacks.CSVLogger(str(models_dir / 'training_log_transfer.csv'), append=False)
    ]

    if epochs_frozen > 0:
        print("[cyan]Phase 1: training head with backbone frozen[/cyan]")
        model.fit(train_ds, validation_data=val_ds, epochs=epochs_frozen, callbacks=callbacks_common, verbose=1)

    # Phase 2: fine-tune upper layers of backbone
    # Unfreeze from a chosen layer upwards
    unfreeze = False
    for layer in base.layers:
        if layer.name == unfreeze_at_layer_name:
            unfreeze = True
        layer.trainable = unfreeze
    print(f"[yellow]Fine-tuning from layer '{unfreeze_at_layer_name}' onward: {sum(l.trainable for l in base.layers)} layers trainable[/yellow]")

    compile_model(model, lr_finetune)
    if epochs_finetune > 0:
        print("[cyan]Phase 2: fine-tuning backbone (low LR)[/cyan]")
        model.fit(train_ds, validation_data=val_ds, epochs=epochs_finetune, callbacks=callbacks_common, verbose=1)

    # Final save (best already saved by checkpoint)
    model.save(str(model_path).replace('.h5', '_last.h5'))
    print(f"[green]Training complete.[/green] Best model saved to: {model_path}")


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', save_path=None):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    plt.close()


def make_eval_ds(x, y, batch_size=256):
    """Helper to build a test-only dataset (no aug)."""
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(lambda img, lbl: (to_rgb_and_resize(img), lbl), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def evaluate(model_path, batch_size=256, plot_cm=True):
    set_seeds()
    (_, _), (_, _), (x_test, y_test) = load_fer2013_from_folders()
    test_ds = make_eval_ds(x_test, y_test, batch_size=batch_size)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    compile_model(model, lr=1e-4)

    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"[bold]Test Loss:[/bold] {loss:.4f} | [bold]Test Acc:[/bold] {acc:.4f}")

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(test_ds, verbose=1), axis=1)

    print("\n[bold]Classification Report[/bold]")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    if plot_cm:
        models_dir = Path("mobilenetv2_models")
        models_dir.mkdir(exist_ok=True)
        plot_confusion_matrix(cm, EMOTION_LABELS, normalize=False, title='Confusion Matrix', save_path=str(models_dir / 'cm_counts.png'))
        plot_confusion_matrix(cm, EMOTION_LABELS, normalize=True, title='Confusion Matrix (Normalized)', save_path=str(models_dir / 'cm_normalized.png'))


def live_inference(model_path, camera_index=0, min_confidence=0.45):
    """
    Webcam live detection with MobileNetV2 backbone model.
    Steps: detect face -> grayscale -> to RGB -> resize to 160 -> preprocess -> predict.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("[bold]Loading model...[/bold]")
    model = keras.models.load_model(model_path, compile=False)

    # Haar cascade - try local files first
    face_cascade_path = "haarcascade_files/haarcascade_frontalface_default.xml"
    if not os.path.exists(face_cascade_path):
        face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(face_cascade_path):
            raise FileNotFoundError("OpenCV haarcascade_frontalface_default.xml not found.")
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Failed to load face cascade classifier.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not access the webcam. Try a different --camera_index.")

    print("[green]Webcam opened. Press 'q' to quit.[/green]")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[red]Failed to grab frame from webcam.[/red]")
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        # Add frame counter and face count to display
        cv2.putText(frame, f"Frame: {frame_count}, Faces: {len(faces)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for (x, y, w, h) in faces:
            pad = int(0.1 * w)
            x0 = max(x - pad, 0); y0 = max(y - pad, 0)
            x1 = min(x + w + pad, frame.shape[1]); y1 = min(y + h + pad, frame.shape[0])

            face_roi = gray[y0:y1, x0:x1]                           # (h,w)
            face_resized = cv2.resize(face_roi, (IMG_SIZE_RAW, IMG_SIZE_RAW))
            face_norm = face_resized.astype('float32') / 255.0
            face_gray3 = np.repeat(face_norm[..., None], 3, axis=-1) # (48,48,3)
            face_rgb = cv2.cvtColor((face_gray3*255).astype('uint8'), cv2.COLOR_BGR2RGB)  # still gray but 3ch
            face_rgb = cv2.resize(face_rgb, (IMG_RESIZED, IMG_RESIZED))
            face_rgb = face_rgb.astype('float32')
            face_input = mnv2_preprocess(face_rgb)                   # [-1,1]
            face_input = np.expand_dims(face_input, axis=0)          # (1,H,W,3)

            preds = model.predict(face_input, verbose=0)[0]
            best_idx = int(np.argmax(preds))
            prob = float(preds[best_idx])
            label = EMOTION_LABELS[best_idx]

            # Draw
            color = (0, 255, 0) if prob >= min_confidence else (0, 165, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            text = f"{label}: {prob*100:.1f}%"
            cv2.putText(frame, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.imshow("FER2013 MobileNetV2 Live (press 'q' to quit)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[bold]Live session ended.[/bold]")


def main():
    parser = argparse.ArgumentParser(description="FER2013 Transfer Learning (MobileNetV2)")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'live'])
    parser.add_argument('--model_path', type=str, default='fer_mnv2_best.h5')
    parser.add_argument('--epochs_frozen', type=int, default=10, help='Train head with backbone frozen')
    parser.add_argument('--epochs_finetune', type=int, default=20, help='Fine-tune upper layers')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr_frozen', type=float, default=1e-3)
    parser.add_argument('--lr_finetune', type=float, default=5e-5)
    parser.add_argument('--unfreeze_from', type=str, default='block_13_expand',
                        help="Unfreeze from this layer name (in MobileNetV2) for fine-tuning")
    parser.add_argument('--camera_index', type=int, default=0)
    parser.add_argument('--min_conf', type=float, default=0.45)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.model_path,
              epochs_frozen=args.epochs_frozen, epochs_finetune=args.epochs_finetune,
              batch_size=args.batch_size, lr_frozen=args.lr_frozen, lr_finetune=args.lr_finetune,
              unfreeze_at_layer_name=args.unfreeze_from)
    elif args.mode == 'evaluate':
        evaluate(args.model_path, batch_size=max(64, args.batch_size))
    elif args.mode == 'live':
        live_inference(args.model_path, camera_index=args.camera_index, min_confidence=args.min_conf)
    else:
        raise ValueError("Invalid mode. Choose from: train, evaluate, live.")


if __name__ == '__main__':
    main()
