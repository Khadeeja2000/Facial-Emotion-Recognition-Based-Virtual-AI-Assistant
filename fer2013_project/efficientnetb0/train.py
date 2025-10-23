#!/usr/bin/env python3
"""
FER2013 Facial Emotion Recognition with EfficientNetB0 (ImageNet-pretrained)
- Training + Evaluation
- Saves best and last weights
- Exports metrics (accuracy, macro-F1), confusion matrices, and classification report
"""

import os, json, argparse, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, efficientnet
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# -------------------------
# Config
# -------------------------
IMG_SIZE_RAW = 48
IMG_RESIZED = 224      # EfficientNetB0 default input
NUM_CLASSES = 7
EMOTION_LABELS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# -------------------------
# Helpers
# -------------------------
def parse_pixels(pixels_str: str):
    arr = np.fromstring(pixels_str, dtype=np.uint8, sep=' ')
    arr = arr.reshape(IMG_SIZE_RAW, IMG_SIZE_RAW)
    arr = arr.astype("float32") / 255.0
    return np.expand_dims(arr, -1)  # (48,48,1)

def load_fer2013(csv_path="data/fer2013.csv"):
    df = pd.read_csv(csv_path)
    df['emotion'] = df['emotion'].astype(int)
    def pack(subset):
        sub = df[df['Usage'] == subset]
        x = np.stack(sub['pixels'].apply(parse_pixels))
        y = keras.utils.to_categorical(sub['emotion'], NUM_CLASSES)
        return x, y
    x_train,y_train = pack("Training")
    x_test,y_test   = pack("PrivateTest")
    
    # Split training data into train/validation
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    return (x_train,y_train),(x_val,y_val),(x_test,y_test)

def to_rgb_resize(img):
    img = tf.tile(img, [1,1,3])
    img = tf.image.resize(img, (IMG_RESIZED, IMG_RESIZED))
    img = img*255.0
    img = efficientnet.preprocess_input(img)
    return img

def make_ds(x,y,batch=64, training=False):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((x,y))
    if training:
        def aug(img,lbl):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img,0.2)
            img = tf.image.random_contrast(img,0.8,1.2)
            return img,lbl
        ds = ds.shuffle(8192).map(aug,num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda im,lb:(to_rgb_resize(im),lb), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)

def build_model(trainable_backbone=False, dropout=0.4):
    inputs = layers.Input(shape=(IMG_RESIZED, IMG_RESIZED, 3))
    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
    base.trainable = trainable_backbone
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="FER_EfficientNetB0")
    return model, base

def plot_cm(cm, labels, normalize=False, fname=None):
    if normalize: cm = cm.astype("float") / (cm.sum(1)[:,None]+1e-12)
    plt.figure(figsize=(8,8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),ha="center",
                 color="white" if cm[i,j]>thresh else "black")
    plt.tight_layout(); plt.ylabel("True"); plt.xlabel("Predicted")
    if fname: plt.savefig(fname); print(f"Saved {fname}")
    plt.close()

# -------------------------
# Training + Eval
# -------------------------
def train(csv="data/fer2013.csv", out_dir="efficientnetb0", epochs_frozen=8, epochs_ft=20):
    (xtr,ytr),(xval,yval),(xts,yts) = load_fer2013(csv)
    tr = make_ds(xtr,ytr,64,True)
    va = make_ds(xval,yval,64,False)
    te = make_ds(xts,yts,64,False)

    model, base = build_model(False)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir,"best.h5")
    last_path = os.path.join(out_dir,"last.h5")

    cbs = [
        keras.callbacks.ModelCheckpoint(best_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ]
    print("Phase 1: training head...")
    model.fit(tr, validation_data=va, epochs=epochs_frozen, callbacks=cbs, verbose=1)

    # Fine-tune top layers
    unfreeze=False
    for layer in base.layers:
        if layer.name=="block6a_expand_conv": unfreeze=True
        layer.trainable = unfreeze
    model.compile(optimizer=keras.optimizers.Adam(3e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    print("Phase 2: fine-tuning...")
    model.fit(tr, validation_data=va, epochs=epochs_ft, callbacks=cbs, verbose=1)

    model.save(last_path)
    print("Saved last model at", last_path)

    # Evaluate
    best_model = keras.models.load_model(best_path, compile=False)
    y_true = np.argmax(yts,1)
    y_pred = np.argmax(best_model.predict(te),1)
    acc = accuracy_score(y_true,y_pred)
    f1  = f1_score(y_true,y_pred, average="macro")
    rpt = classification_report(y_true,y_pred,target_names=EMOTION_LABELS,digits=4)
    cm  = confusion_matrix(y_true,y_pred)

    res_dir = os.path.join(out_dir,"results"); os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir,"metrics.json"),"w") as f: json.dump({"accuracy":acc,"macro_f1":f1}, f, indent=2)
    with open(os.path.join(res_dir,"classification_report.txt"),"w") as f: f.write(rpt)
    plot_cm(cm, EMOTION_LABELS, False, os.path.join(res_dir,"cm_counts.png"))
    plot_cm(cm, EMOTION_LABELS, True, os.path.join(res_dir,"cm_normalized.png"))
    print("Evaluation done. Accuracy=%.4f Macro-F1=%.4f"%(acc,f1))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/fer2013.csv")
    ap.add_argument("--out_dir", default="efficientnetb0")
    ap.add_argument("--epochs_frozen", type=int, default=8)
    ap.add_argument("--epochs_ft", type=int, default=20)
    args = ap.parse_args()
    train(args.csv, args.out_dir, args.epochs_frozen, args.epochs_ft)
