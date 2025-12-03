import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os

# Load trained model
model = keras.models.load_model("C:\\Users\\Elijah\\Desktop\\PyCharm\\CV_Animal_Distinguisher\\ews_animal_distinguisher_v3.keras")

# Load class labels
with open("C:\\Users\\Elijah\\Desktop\\PyCharm\\CV_Animal_Distinguisher\\classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

img_size = (224, 224)

def classify_and_show(path):
    # Load image and convert to array
    img = image.load_img(path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)

    # Predict animal type
    preds = model.predict(img_array)[0]
    index = np.argmax(preds)
    predicted_class = class_names[index]
    confidence = preds[index]

    # Plot image + prediction
    plt.imshow(image.load_img(path))
    plt.axis("off")
    plt.title(f"Prediction: {predicted_class}\n(Confidence: {confidence*100:.2f}%)")
    plt.show()

test_picture = "insert test picture path"
classify_and_show(test_picture)

# Used to add picture to training dataset after classification
shutil.copy(test_picture, "insert path to training dataset for test picture animal")