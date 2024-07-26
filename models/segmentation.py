import os
import cv2
from cellpose import models, core, io

control = False

def segment_dataset(images_dir):
    global control
    if control:
        print(f"Segmentación ya realizada para {images_dir}. Saltando...")
        return
    
    control = True
    control_file = os.path.join(images_dir, ".segmentation_done")
    if os.path.exists(control_file):
        print(f"Segmentación ya realizada para {images_dir}. Saltando...")
        return

    model_path = "models/vicia_faba_11"
    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')

    model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_path)
    channels = [0, 0]

    for filename in os.listdir(images_dir):
        if filename.endswith(('.tiff', '.jpeg', '.jpg', '.png')):
            print(f"Procesando imagen: {filename}")
            image_path = os.path.join(images_dir, filename)
            output_path = os.path.join(images_dir, filename)  # Asegúrate de que la salida se guarde correctamente
            image = io.imread(image_path)
            if image is None:
                print(f"Error: No se pudo cargar la imagen {image_path}")
                continue

            #Se hace un resize de la imagen paque sea de 900x900 pero con otra biblioteca
            # image = cv2.resize(image, (900, 900))

            masks, flows, styles = model.eval(image, diameter=None, channels=channels)

            # Guardar las máscaras con io.masks_flows_to_seg
            io.masks_flows_to_seg(image, masks, flows, output_path)
            print(f"Segmentación guardada en: {output_path}")

    # Crear el archivo de control al finalizar la segmentación
    open(control_file, 'w').close()
    control = False