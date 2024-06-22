from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.preprocessing import image
import numpy as np
import os

model_path = "app/models/best_model.keras"
modelo_clasificacion = load_model(model_path)

print(modelo_clasificacion)
img_width, img_height = 256, 256

def clasificar_celula(imagen_path):
    img_width, img_height = 256, 256

    img = image.load_img(imagen_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)


    prediction = modelo_clasificacion.predict(x)
    class_index = np.argmax(prediction)

    class_names = [
        "Interfase",
        "Profase",
        "Metafase",
        "Anafase",
        "Telofase"
    ]
    return class_names[class_index]