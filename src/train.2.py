import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# === Ayarlar ===
DATA_DIR = Path("../Veri/train_set")
MODEL_DIR = Path("../models")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# === Veri yükleyici ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=SEED
)

# === Model tanımları ===
def build_model_1():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_model_2():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(2,2),
        Dropout(0.25),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

# === Eğitim fonksiyonu ===
def train_and_save(model, name):
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'mse'])

    checkpoint_path = MODEL_DIR / f"{name}.h5"
    checkpoint = ModelCheckpoint(str(checkpoint_path), save_best_only=True, monitor='val_accuracy', verbose=1)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint],
        verbose=1
    )

    return history

# === Görselleştirme ===
def plot_history(history, name):
    metrics = ['accuracy', 'loss', 'mse']
    for m in metrics:
        plt.figure()
        plt.plot(history.history[m], label=f'Train {m}')
        plt.plot(history.history[f'val_{m}'], label=f'Val {m}')
        plt.title(f'{name} - {m.upper()}')
        plt.xlabel('Epoch')
        plt.ylabel(m)
        plt.legend()
        plt.grid(True)
        plt.savefig(MODEL_DIR / f"{name}_{m}.png")

# === Eğitim işlemleri ===
if __name__ == "__main__":
    histories = {}

    model1 = build_model_1()
    h1 = train_and_save(model1, "model1")
    plot_history(h1, "model1")
    histories["model1"] = h1.history

    model2 = build_model_2()
    h2 = train_and_save(model2, "model2")
    plot_history(h2, "model2")
    histories["model2"] = h2.history

    # İstatistikleri JSON olarak da kaydet
    with open(MODEL_DIR / "histories.json", "w") as f:
        json.dump({k: {m: list(v) for m, v in h.items()} for k, h in histories.items()}, f, indent=2)