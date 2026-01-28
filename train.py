import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
from googlenet_classifier import GoogLeNetClassifier

class HAMDataset(Dataset):
    def __init__(self, dataframe, image_dir1, image_dir2, transform=None):
        self.data = dataframe
        self.image_dir1 = image_dir1
        self.image_dir2 = image_dir2
        self.transform = transform
        self.label_map = {'nv': 0, 'bkl': 1, 'df': 2, 'vasc': 3, 'bcc': 4, 'akiec': 5, 'mel': 6}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id, label = row['image_id'], row['dx']
        path1 = os.path.join(self.image_dir1, img_id + ".jpg")
        path2 = os.path.join(self.image_dir2, img_id + ".jpg")
        img_path = path1 if os.path.exists(path1) else path2
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label_map[label]

def prepare_hybrid_dataset(original_csv, samples_per_class=200):
    df = pd.read_csv(original_csv)
    balanced_df = df.groupby('dx').apply(lambda x: x.sample(n=samples_per_class, replace=True)).reset_index(drop=True)
    combined_df = pd.concat([balanced_df, df.sample(frac=0.4, random_state=42)])
    print(f"Hybrid dataset created with {len(combined_df)} samples.")
    return combined_df

def train_classifier():
    # Step 1: Prepare dataset
    df = prepare_hybrid_dataset("HAM10000_metadata.csv", samples_per_class=200)

    # Step 2: Define strong augmentations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor()
    ])

    # Step 3: Load data
    dataset = HAMDataset(df, "HAM10000_images_part_1", "HAM10000_images_part_2", transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Step 4: Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNetClassifier().to(device)

    # Step 5: Define weighted loss
    weights = torch.tensor([1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0], dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Step 6: Train for more epochs
    for epoch in range(20):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
        torch.save(model.state_dict(), f"models/googlenet_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "models/googlenet.pth")
    print("Final GoogLeNet model saved.")

if __name__ == "__main__":
    train_classifier()
