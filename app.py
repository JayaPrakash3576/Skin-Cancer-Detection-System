import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision import models
import torch.nn as nn
import pandas as pd
from unet_model import UNet
from utils import map_severity

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load U-Net model
unet_model = UNet().to(device)
unet_model.load_state_dict(torch.load("models/unet.pth", map_location=device))
unet_model.eval()

# Load GoogLeNet model
model = models.googlenet(weights=None, aux_logits=True, num_classes=2).to(device)
model.fc = nn.Linear(1024, 2)
model.aux1.fc2 = nn.Linear(1024, 2)
model.aux2.fc2 = nn.Linear(1024, 2)
model.load_state_dict(torch.load("models/googlenet.pth", map_location=device))
model.eval()

# Load label classes
class_names = np.load("label_classes.npy", allow_pickle=True)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Lesion segmentation
def segment_lesion(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        mask = unet_model(image_tensor)
        mask = (mask > 0.5).float()
        segmented = image_tensor * mask
    return segmented

# Streamlit UI
st.title("🧪 Skin Cancer Detection App")
st.markdown("Upload a skin lesion image to classify it as **Benign** or **Malignant**, view **severity**, and see if prediction matches ground truth.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    if st.button("🔍 Classify"):
        segmented = segment_lesion(image)

        with torch.no_grad():
            outputs = model(segmented)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_label = class_names[pred_idx]
            severity = map_severity(pred_label)

        # Match actual label from HAM10000 metadata
        metadata = pd.read_csv("HAM10000_metadata.csv")
        img_name = uploaded_file.name.split(".")[0]
        actual_row = metadata[metadata['image_id'] == img_name]

        if not actual_row.empty:
            true_label = 'Malignant' if actual_row.iloc[0]['dx'] in ['akiec', 'bcc', 'mel'] else 'Benign'
            correct = (true_label == pred_label)
        else:
            true_label = "Unknown"
            correct = None

        # Display results
        st.markdown(f"🧾 **Prediction:** `{pred_label}`")
        st.markdown(f"📌 **Actual Diagnosis:** `{true_label}`")
        st.markdown(f"⚠️ **Severity Level:** `{severity}`")

        if correct is not None:
            st.success(f"🎯 **Prediction Correct:** {correct}")
