import os
from cellpose import models, core, io

def segment_dataset(images_dir):
    model_path = "models/vicia_faba_11" 
    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')

    model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_path)
    channels = [0, 0]  # Ajusta si usas canales adicionales
    
    for filename in os.listdir(images_dir):
        if filename.endswith(('.tiff', '.jpeg', '.jpg', '.png')): 
            print(f"Procesando imagen: {filename}")
            image_path = os.path.join(images_dir, filename)
            output_path = os.path.join(images_dir, filename)
            image = io.imread(image_path)
            if image is None:
                print(f"Error: No se pudo cargar la imagen {image_path}")
                continue

            print(f"Procesando imagen: {filename}")

            masks, flows, styles = model.eval(image, diameter=None, channels=channels)

            # Guardar las máscaras con io.masks_flows_to_seg
            io.masks_flows_to_seg(image, masks, flows, output_path)
            print(f"Segmentación guardada en: {output_path}")