from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os



def clasificar_celula(imagen_path):


    # Verificar el directorio actual
    current_directory = os.getcwd()
    print(f"Directorio actual: {current_directory}")

    # Verificar los archivos en el directorio de modelos
    models_directory = os.path.join(current_directory, 'models')
    print(f"Archivos en el directorio 'models': {os.listdir(models_directory)}")

    # Ruta del modelo
    model_path = os.path.join(models_directory, 'best_model.keras')
    print(f"Ruta del modelo: {model_path}")

    # Verificar si el archivo existe en la ruta especificada
    if not os.path.exists(model_path):
        print(f"Error: el archivo {model_path} no existe.")
    else:
        print(f"El archivo {model_path} existe. Intentando cargar el modelo...")

    # Cargar el modelo
    try:
        modelo_clasificacion = load_model(model_path)
        print(f"Modelo cargado correctamente desde {model_path}")
    except ValueError as e:
        print(f"Error al cargar el modelo: {str(e)}")
    img_width, img_height = 256, 256
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