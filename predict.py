import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
from unet_model import UNet
from sklearn.metrics import classification_report

def map_severity(label):
    return 'High' if label == 'Malignant' else 'Low'


def segment_lesion(image, unet_model, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    mask = unet_model(image_tensor)
    mask = (mask > 0.5).float()
    return image_tensor * mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_model = UNet().to(device)
    unet_model.load_state_dict(torch.load("models/unet.pth", map_location=device))
    unet_model.eval()

    class_names = np.load("label_classes.npy", allow_pickle=True)

    model = models.googlenet(weights=None, aux_logits=True, num_classes=2).to(device)
    model.fc = nn.Linear(1024, 2)
    model.aux1.fc2 = nn.Linear(1024, 2)
    model.aux2.fc2 = nn.Linear(1024, 2)
    model.load_state_dict(torch.load("models/googlenet.pth", map_location=device))
    model.eval()

    metadata = pd.read_csv("HAM10000_metadata.csv")
    test_images = metadata.sample(100, random_state=42)

    y_true, y_pred = [], []

    for _, row in test_images.iterrows():
        img_id = row['image_id']
        img_path = f"HAM10000_images_part_1/{img_id}.jpg"
        if not os.path.exists(img_path):
            img_path = f"HAM10000_images_part_2/{img_id}.jpg"
        image = Image.open(img_path).convert('RGB')

        segmented = segment_lesion(image, unet_model, device)
        with torch.no_grad():
            outputs = model(segmented)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            pred_idx = torch.argmax(logits, dim=1).item()

        pred_label = class_names[pred_idx]
        true_label = 'Malignant' if row['dx'] in ['akiec', 'bcc', 'mel'] else 'Benign'
        severity = map_severity(pred_label)


        y_true.append(true_label)
        y_pred.append(pred_label)

        print(f"{img_id}: True={true_label}, Predicted={pred_label}, Severity={severity}")

    print("\n Classification Report:")
    print(classification_report(y_true, y_pred))
    print("\n Severity Breakdown:")
    severities = [map_severity(label) for label in y_pred]
    print(pd.Series(severities).value_counts())


if __name__ == "__main__":
    main()
