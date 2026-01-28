import os
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def load_image(image_id, root1="HAM10000_images_part_1", root2="HAM10000_images_part_2"):
    path1 = os.path.join(root1, image_id + ".jpg")
    path2 = os.path.join(root2, image_id + ".jpg")
    path = path1 if os.path.exists(path1) else path2
    image = Image.open(path).convert("RGB")
    return transform(image)
