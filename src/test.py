import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Ayarlarım
IMG_SIZE = (128, 128)
CAT_DIR = '/Users/sabitboyoglu/Desktop/trex_digital_Ai/Cats_v_Dogs/Veri/test_set/cats'
DOG_DIR = '/Users/sabitboyoglu/Desktop/trex_digital_Ai/Cats_v_Dogs/Veri/test_set/dogs'
MODEL_PATH = '/Users/sabitboyoglu/Desktop/trex_digital_Ai/Cats_v_Dogs/models/model1.h5' #model1.h5' ,#model2.h5'

# Rastgele görselleri seç
def get_random_images(folder, count=5):
    all_images = os.listdir(folder)
    selected = random.sample(all_images, count)
    return [os.path.join(folder, fname) for fname in selected]

# Görselleri yükle ve tahmin et
def load_and_predict_images(image_paths, model):
    images = []
    labels = []
    preds = []

    for path in image_paths:
        target_size = (128, 128)
        img = image.load_img(path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0
        img_expanded = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_expanded)[0][0]
        images.append(img_array)
        preds.append(pred)
        label = "Cat" if "cats" in path else "Dog"
        labels.append(label)

    return images, preds, labels

def plot_predictions(images, preds, labels):
    plt.figure(figsize=(15, 6))
    for i, (img, pred, true_label) in enumerate(zip(images, preds, labels)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.axis('off')
        pred_label = "Dog" if pred > 0.5 else "Cat"
        color = "green" if pred_label == true_label else "red"
        plt.title(f" Tahmin: {pred_label}\n Gerçek: {true_label}", color=color)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Modeli yükleyelim
    model = load_model(MODEL_PATH)

    # Görselleri toplanıyor
    cat_images = get_random_images(CAT_DIR, 5)
    dog_images = get_random_images(DOG_DIR, 5)
    all_images = cat_images + dog_images

    # Tahmin yapacağız
    images, preds, labels = load_and_predict_images(all_images, model)

    # Sonuçları görselleştir
    plot_predictions(images, preds, labels)