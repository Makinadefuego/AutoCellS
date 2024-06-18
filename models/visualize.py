#Script que habre una imagen en uint16 y la normaliza a uint8
#Para su visualización

import cv2
import numpy as np
import os

def normalize_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("The image cox uld not be loaded. Check the image path.")
    
    print(f"Loaded image shape: {image.shape}")
    print(f"Loaded image dtype: {image.dtype}")
    
    img_scaled = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    output_path = os.path.splitext(image_path)[0] + "_normalized.png"
    cv2.imwrite(output_path, img_scaled)
    
    
    print(f"Normalized image saved as {output_path}")
    
    return output_path

if __name__ == "__main__":
    image_path = "models/vicia_faba_07_segmented.png"
    normalize_image(image_path)