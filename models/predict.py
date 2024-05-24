import os
import numpy as np
from cellpose import models, core, io
import cv2

def segment_image(image_path):
    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')

    model_path = "models/vicia_faba_07"
    output_base_name = os.path.splitext(os.path.basename(image_path))[0] + "_segmented"
    output_path = os.path.join(os.path.dirname(image_path), output_base_name + ".png")

    image = io.imread(image_path)
    if image is None:
        raise ValueError("The image could not be loaded. Check the image path.")

    print(f"Loaded image shape: {image.shape}")
    print(f"Loaded image dtype: {image.dtype}")

    model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_path)
    channels = [0, 0]
    masks, flows, styles = model.eval(image, diameter=None, channels=channels)

    img_scaled = cv2.normalize(masks, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(output_path, img_scaled)

    print(f"Segmentation results saved as {output_path}")
    return output_path
