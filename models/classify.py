from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargar el modelo de clasificación (ajusta la ruta si es necesario)
modelo_clasificacion = load_model('./models/best_model.keras')

print(modelo_clasificacion)
img_width, img_height = 256, 256

def clasificar_celula(imagen_path):
    """
    Clasifica una sola imagen de célula.

    Args:
        imagen_path (str): Ruta a la imagen de la célula.

    Returns:
        str: Nombre de la clase predicha.
    """
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