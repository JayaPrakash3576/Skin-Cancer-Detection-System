import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from unet_model import UNet

class UNetDataset(Dataset):
    def __init__(self, image_ids, image_dir, mask_dir):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path1 = os.path.join(self.image_dir[0], image_id + ".jpg")
        img_path2 = os.path.join(self.image_dir[1], image_id + ".jpg")
        image_path = img_path1 if os.path.exists(img_path1) else img_path2
        mask_path = os.path.join(self.mask_dir, image_id + "_mask.png")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        return image, mask

def train_unet():
    image_dir = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
    mask_dir = "lesion_masks"
    image_ids = [f.split("_mask.png")[0] for f in os.listdir(mask_dir) if f.endswith("_mask.png")]
    if not image_ids:
        raise FileNotFoundError(f"No mask files found in {mask_dir}.")
    
    dataset = UNetDataset(image_ids, image_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/unet.pth")
    print("U-Net model saved to models/unet.pth") 

if __name__ == "__main__":
    train_unet()
