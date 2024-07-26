import numpy as np
from PIL import Image, ImageEnhance
import os
import cv2

ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff')
TARGET_SIZE = (256, 256)  # Tamaño fijo para las imágenes de salida
control = False
def preprocess_images(images_dir):
    global control
    if control:
        print(f"Preprocesamiento ya realizado para {images_dir}. Saltando...")
        return
    control = True

    # Obtener los nombres de los archivos de imagen y máscara
    image_files = {os.path.splitext(f)[0]: os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(ALLOWED_EXTENSIONS)}
    mask_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('_seg.npy')]

    # Crear el directorio de salida
    output_folder = os.path.join(images_dir, 'preprocessed')
    os.makedirs(output_folder, exist_ok=True)

    # Procesar cada archivo de máscara
    for mask_file in mask_files:
        try:
            mask_data = np.load(mask_file, allow_pickle=True).item()
            print(f"Procesando máscara: {mask_file}")
            mask = mask_data['masks']
            print("El valor máximo de la máscara es: ")
            print(np.max(mask))

            image_base_name = os.path.splitext(os.path.basename(mask_file))[0].replace('_seg', '')
            image_file = image_files.get(image_base_name)

            image = Image.open(image_file)
            for cell_id, cell_mask_value in enumerate(np.unique(mask), start=0):
                if cell_mask_value == 0:
                    continue

                cell_mask = mask == cell_mask_value
                if not np.any(cell_mask):
                    continue

                # Convertir la máscara a uint8 para operaciones de OpenCV
                cell_mask_uint8 = (cell_mask * 255).astype(np.uint8)
                
                # Aplicar operaciones morfológicas para limpiar la máscara
                kernel = np.ones((5, 5), np.uint8)
                cell_mask_cleaned = cv2.morphologyEx(cell_mask_uint8, cv2.MORPH_CLOSE, kernel)

                # Encontrar el cuadro delimitador de la región de la célula enmascarada
                x, y, w, h = cv2.boundingRect(cell_mask_cleaned)

                # Recortar la región de la célula enmascarada
                cell_image = image.crop((x, y, x + w, y + h))

                # Crear una nueva imagen con fondo negro del tamaño objetivo
                cell_image_with_mask = Image.new('RGB', (w, h), (0, 0, 0))

                # Aplicar la máscara a la imagen recortada
                cell_mask_pil = Image.fromarray(cell_mask_cleaned[y:y+h, x:x+w])
                cell_image_with_mask.paste(cell_image, (0, 0), cell_mask_pil)

                # Redimensionar la célula al tamaño objetivo
                cell_image_with_mask = cell_image_with_mask.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

                # Aplicar mejoras de imagen
                enhancer = ImageEnhance.Contrast(cell_image_with_mask)
                cell_image_with_mask = enhancer.enhance(2)  # Ajustar contraste
                enhancer = ImageEnhance.Sharpness(cell_image_with_mask)
                cell_image_with_mask = enhancer.enhance(3)  # Ajustar nitidez
                

                 # Guardar la máscara como imagen
                mask_image = Image.fromarray((cell_mask * 255).astype(np.uint8))
                mask_image_path = os.path.join(output_folder, f'{os.path.basename(mask_file)[:-8]}_{cell_id}_mask.png')
                mask_image.save(mask_image_path)

                # Guardar la imagen de la célula
                cell_image_path = os.path.join(output_folder, f'{os.path.basename(mask_file)[:-8]}_{cell_id}.png')
                cell_image_with_mask.save(cell_image_path)

        except Exception as e:
            print(f"Error procesando la máscara {mask_file}: {e}")

    print("Preprocesamiento y división de datos completados")
    control = False
    return output_folder

if __name__ == '__main__':
    images_dir = "representativas"
    preprocess_images(images_dir)
