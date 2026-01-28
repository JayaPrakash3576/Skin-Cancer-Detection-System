import os
from PIL import Image, ImageOps
import numpy as np

def generate_dummy_masks(image_folder, mask_folder, threshold=160):
    os.makedirs(mask_folder, exist_ok=True)
    for fname in os.listdir(image_folder):
        if not fname.endswith('.jpg'):
            continue
        img_path = os.path.join(image_folder, fname)
        image = Image.open(img_path).convert('L')
        inverted = ImageOps.invert(image)
        binary = inverted.point(lambda p: 255 if p > threshold else 0)
        mask_name = fname.replace('.jpg', '_mask.png')
        binary.save(os.path.join(mask_folder, mask_name))
    print(f"Dummy masks generated in '{mask_folder}'")

if __name__ == "__main__":
    generate_dummy_masks("HAM10000_images_part_1", "lesion_masks")
