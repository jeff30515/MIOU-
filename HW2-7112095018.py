import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.down5 = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512,
                                      kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256,
                                      kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128,
                                      kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64,
                                      kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        d5 = self.down5(p4)

        u1 = self.up1(d5)
        u1 = self.center_crop(u1, d4.size()[2:])
        u1 = torch.cat([u1, d4], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = self.center_crop(u2, d3.size()[2:])
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = self.center_crop(u3, d2.size()[2:])
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = self.center_crop(u4, d1.size()[2:])
        u4 = torch.cat([u4, d1], dim=1)
        u4 = self.conv4(u4)

        return self.out(u4)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:diff_y + target_size[0], diff_x:diff_x + target_size[1]]


class DRIVE_Dataset(Dataset):
    def __init__(self, images_path, masks_path, fov_masks_path=None, transform=None, mode='train'):
        self.images_path = images_path
        self.masks_path = masks_path
        self.fov_masks_path = fov_masks_path
        self.transform = transform
        self.mode = mode
        self.images = sorted(os.listdir(images_path))
        self.masks = sorted(os.listdir(masks_path))
        if fov_masks_path is not None:
            self.fov_masks = sorted(os.listdir(fov_masks_path))
        else:
            self.fov_masks = None

        self.image_mask_map = {}
        if mode == 'train':
            for img_name in self.images:
                img_number = img_name[:2]
                mask_name = f"{img_number}_manual1.gif"
                fov_mask_name = f"{img_number}_training_mask.gif"
                if mask_name in self.masks and fov_mask_name in self.fov_masks:
                    self.image_mask_map[img_name] = (mask_name, fov_mask_name)
                else:
                    print(f"No matching mask or FOV for image {img_name}")
        elif mode == 'test':
            for img_name in self.images:
                if img_name.endswith('_real_A.png'):
                    mask_name = img_name.replace('_real_A.png', '_real_B.png')
                    if mask_name in self.masks:
                        self.image_mask_map[img_name] = (mask_name, None)
                    else:
                        print(f"No matching mask for image {img_name}")
            print(f"Total matched test images: {len(self.image_mask_map)}")
        else:
            raise ValueError("Mode must be 'train' or 'test'")

    def __len__(self):
        return len(self.image_mask_map)

    def __getitem__(self, idx):
        img_name = list(self.image_mask_map.keys())[idx]
        img_path = os.path.join(self.images_path, img_name)
        mask_name, fov_mask_name = self.image_mask_map[img_name]
        mask_path = os.path.join(self.masks_path, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0.5).float()

        if fov_mask_name is not None:
            fov_mask_path = os.path.join(self.fov_masks_path, fov_mask_name)
            fov_mask = Image.open(fov_mask_path).convert("L")
            if self.transform:
                fov_mask = self.transform(fov_mask)
            fov_mask = (fov_mask > 0.5).float()
        else:
            fov_mask = torch.ones_like(mask)

        return image, mask, fov_mask


transform = transforms.Compose([transforms.Resize((512, 512)),transforms.ToTensor(),])


train_images_path = "C:\\Users\\guanju\\Desktop\\深度學習\\HW2\\archive\\DRIVE\\training\\images"
train_masks_path = "C:\\Users\\guanju\\Desktop\\深度學習\\HW2\\archive\\DRIVE\\training\\1st_manual"
train_fov_masks_path = "C:\\Users\\guanju\\Desktop\\深度學習\\HW2\\archive\\DRIVE\\training\\mask"

test_images_path = "C:\\Users\\guanju\\Desktop\\深度學習\\HW2\\DIRVE_TestingSet"
test_masks_path = "C:\\Users\\guanju\\Desktop\\深度學習\\HW2\\DIRVE_TestingSet"

test_fov_masks_path = None


train_dataset = DRIVE_Dataset(train_images_path,train_masks_path,train_fov_masks_path,transform=transform,mode='train')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = DRIVE_Dataset(test_images_path,test_masks_path,fov_masks_path=test_fov_masks_path,  transform=transform,mode='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


model = Unet().cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 400

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, masks, fov_masks in train_loader:
        images = images.cuda()
        masks = masks.cuda()
        fov_masks = fov_masks.cuda()

        outputs = model(images)

        loss = criterion(outputs * fov_masks, masks * fov_masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def compute_metrics(output, target, fov_mask):
    output = torch.sigmoid(output)
    output = (output > 0.5).float()

    intersection = ((output * target) * fov_mask).sum().item()
    union = ((output + target) * fov_mask).sum().item() - intersection
    iou = intersection / (union + 1e-6)

    tp = (output * target * fov_mask).sum().item()
    precision = tp / ((output * fov_mask).sum().item() + 1e-6)
    recall = tp / ((target * fov_mask).sum().item() + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return iou, f1

iou_scores = []
f1_scores = []

model.eval()
with torch.no_grad():
    fig, axes = plt.subplots(len(test_dataset), 3, figsize=(15, len(test_dataset) * 5))
    for idx, (images, masks, fov_masks) in enumerate(test_loader):
        images = images.cuda()
        masks = masks.cuda()
        fov_masks = fov_masks.cuda()

        outputs = model(images)

        iou, f1 = compute_metrics(outputs, masks, fov_masks)
        iou_scores.append(iou)
        f1_scores.append(f1)

        image_np = images[0].cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        masks_np = masks[0].cpu().numpy().squeeze()
        output_np = torch.sigmoid(outputs[0]).cpu().numpy().squeeze() > 0.5

        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title("Original Image")
        axes[idx, 0].axis('off')
        axes[idx, 1].imshow(output_np, cmap="gray")
        axes[idx, 1].set_title("Segmentation Result")
        axes[idx, 1].axis('off')
        axes[idx, 2].imshow(masks_np, cmap="gray")
        axes[idx, 2].set_title("Ground Truth")
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.show()

data = {"Image Index": list(range(1, len(iou_scores) + 1)), "IoU Score": iou_scores, "F1 Score": f1_scores}
df = pd.DataFrame(data)
mean_iou = np.mean(iou_scores)
print(df)
print(f"Mean IoU: {mean_iou:.4f}")

df.to_csv("segmentation_metrics.csv", index=False)
