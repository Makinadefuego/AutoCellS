import os
import numpy as np
from cellpose import models, core, io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image 
from skimage import io
import json
def segment_image(image_path, model):
    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')

    output_base_name = os.path.splitext(os.path.basename(image_path))[0] + "_segmented"
    output_path = os.path.join(os.path.dirname(image_path), output_base_name + ".png")

    image = io.imread(image_path)
    if image is None:
        raise ValueError("The image could not be loaded. Check the image path.")

    print(f"Loaded image shape: {image.shape}")
    print(f"Loaded image dtype: {image.dtype}")

    # Cargar el modelo de CNN 
        # Llama al script de fusión antes de cargar el modelo
    merge_files('best_model_part', 'models/best_model.keras')
    # (adapta la ruta a tu modelo)
    model = load_model('app/models/best_model.keras ')

    print(model)

    # Preprocesar la imagen
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))  # Ajusta las dimensiones si es necesario
    x = image.img_to_array(img)
    x = x / 255.0  

    # Hacer la predicción
    predictions = model.predict(x)[0]

    # Obtener el índice de la clase con mayor probabilidad
    class_predictions = np.argmax(predictions, axis=1)

    # Obtener el conteo de células por clase
    cell_counts = {}
    num_classes = 5  # Ajusta el número de clases
    for class_id in range(num_classes):
        cell_counts[class_id] = np.sum(class_predictions == class_id)
    
    print(f"Segmentation results saved as {output_path}")

    # Guardar las máscaras con io.masks_flows_to_seg
    #io.masks_flows_to_seg(image, masks, flows, output_path)
    #print(f"Segmentation saved as {output_path}")
    
    # Guardar el conteo de células como un archivo JSON
    json_path = os.path.join(os.path.dirname(image_path), output_base_name + "_cell_counts.json")
    with open(json_path, 'w') as f:
        json.dump(cell_counts, f)

    print(f"Cell counts saved as {json_path}")

    return output_path