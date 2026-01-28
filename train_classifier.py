import os
import warnings
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import GoogLeNet_Weights

warnings.filterwarnings("ignore", category=UserWarning, message="auxiliary heads in the pretrained googlenet model")

class HAMDataset(Dataset):
    def __init__(self, dataframe, img_dir1, img_dir2, transform, label_encoder):
        self.data = dataframe
        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self.transform = transform
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id, label = row['image_id'], row['dx_binary']
        img_path1 = os.path.join(self.img_dir1, image_id + ".jpg")
        img_path2 = os.path.join(self.img_dir2, image_id + ".jpg")
        img_path = img_path1 if os.path.exists(img_path1) else img_path2
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.label_encoder.transform([label])[0]
        label = torch.tensor(label, dtype=torch.long)
        return image, label

def train_classifier():
    metadata = pd.read_csv("HAM10000_metadata.csv")
    img_dir1 = "HAM10000_images_part_1"
    img_dir2 = "HAM10000_images_part_2"

    malignant = ['akiec', 'bcc', 'mel']
    metadata['dx_binary'] = metadata['dx'].apply(lambda x: 'Malignant' if x in malignant else 'Benign')

    le = LabelEncoder()
    metadata['label'] = le.fit_transform(metadata['dx_binary'])  # 0: Benign, 1: Malignant
    np.save("label_classes.npy", le.classes_)

    train_df, val_df = train_test_split(metadata, test_size=0.2, stratify=metadata['dx_binary'], random_state=42)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = HAMDataset(train_df, img_dir1, img_dir2, transform, le)
    val_dataset = HAMDataset(val_df, img_dir1, img_dir2, transform, le)

    label_counts = train_df['label'].value_counts().to_dict()
    sample_weights = train_df['label'].map(lambda x: 1.0 / label_counts[x]).to_numpy()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=True)
    model.fc = nn.Linear(1024, 2)
    model.aux1.fc2 = nn.Linear(1024, 2)
    model.aux2.fc2 = nn.Linear(1024, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(25):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            main_output = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(main_output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(main_output, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        print(f"Epoch {epoch+1}/25 | Train Loss: {total_loss:.4f} | Train Accuracy: {correct / total:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/googlenet.pth")
    print(" Model saved to models/googlenet.pth")

if __name__ == "__main__":
    train_classifier()
