
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def convert_to_binary(img, threshold=128):
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Extract the alpha channel
    alpha_channel = img_array[:, :, 3]
    # Create a binary mask for red and blue pixels
    red_mask = (img_array[:, :, 0] > threshold) & (img_array[:, :, 1] < threshold) & (img_array[:, :, 2] < threshold)
    blue_mask = (img_array[:, :, 0] < threshold) & (img_array[:, :, 1] < threshold) & (img_array[:, :, 2] > threshold)
    
    # Initialize binary image with zeros
    binary_image = np.zeros_like(alpha_channel, dtype=np.uint8)
    
    # Assign class labels
    binary_image[red_mask] = 1
    binary_image[blue_mask] = 2

    
    return binary_image

# %%
# Load the images+masks, convert to numpy arrays, and regen masks to fit the training purpose
real_images_dir = 'data/imagery/'
mask_images_dir = 'data/masks/'

x_train = []
y_train = []
x_test = []
y_test = []

SUBIMAGE_SIZE = 576

for item in os.listdir(real_images_dir):
    if item.endswith('.png') and not item.startswith('test'):
        real_img = Image.open(real_images_dir + item)
        mask_img = Image.open(mask_images_dir + item)
        # Divide the image into smaller patches
        real_img_array = np.array(real_img)
        mask_img_array = convert_to_binary(mask_img)
        tiles_real = [real_img_array[x:x+SUBIMAGE_SIZE,y:y+SUBIMAGE_SIZE] for x in range(0,real_img_array.shape[0],SUBIMAGE_SIZE) for y in range(0,real_img_array.shape[1],SUBIMAGE_SIZE)]
        tiles_mask = [mask_img_array[x:x+SUBIMAGE_SIZE,y:y+SUBIMAGE_SIZE] for x in range(0,mask_img_array.shape[0],SUBIMAGE_SIZE) for y in range(0,mask_img_array.shape[1],SUBIMAGE_SIZE)]
        x_train.extend(tiles_real)
        y_train.extend(tiles_mask)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Data shuffling to avoid overfitting
permutation = np.random.permutation(len(x_train))
x_train = x_train[permutation]
y_train = y_train[permutation]

# 80/20 split
split = int(0.8 * len(x_train))
x_test = x_train[split:]
y_test = y_train[split:]
x_train = x_train[:split]
y_train = y_train[:split]

np.save('data/train_images.npy', x_train)
np.save('data/train_masks.npy', y_train)
np.save('data/test_images.npy', x_test)
np.save('data/test_masks.npy', y_test)

del x_train, y_train, x_test, y_test

# %%
x_train = np.load('data/train_images.npy')
y_train = np.load('data/train_masks.npy')
x_test = np.load('data/test_images.npy')
y_test = np.load('data/test_masks.npy')

fig, ax = plt.subplots(1, 2)
ax[0].imshow(x_train[41])
ax[1].imshow(y_train[41])

# %%
import gc
del x_train, y_train, x_test, y_test
gc.collect()

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchmetrics

# Define our custom U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.middle = self.conv_block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        middle = self.middle(self.pool4(enc4))
        dec4 = self.decoder4(torch.cat([self.upconv4(middle), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([self.upconv3(dec4), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.upconv1(dec2), enc1], dim=1))
        return torch.sigmoid(self.final_conv(dec1))

class SegmentationDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images = np.load(images_path)
        self.masks = np.load(masks_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# Preprocessing images
def preprocess_images(images_path, masks_path, batch_size=6, image_size=(SUBIMAGE_SIZE, SUBIMAGE_SIZE)):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    dataset = SegmentationDataset(images_path, masks_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

# Post-processing outputs
def postprocess_outputs(outputs):
    masks = []
    for output in outputs:
        output = output.squeeze(0)                  # Remove batch dimension
        output = output.detach().cpu().numpy()
        masks.append(output)
    return masks

# Function to test the model on a batch of images
def test_model_batch(model, dataloader):
    model.eval()
    all_masks = []
    all_image_names = []
    with torch.no_grad():
        for images, image_names in dataloader:
            images = images.to('cuda')
            outputs = model(images)
            outputs = nn.functional.interpolate(outputs, size=(SUBIMAGE_SIZE, SUBIMAGE_SIZE), mode='bilinear', align_corners=False, recompute_scale_factor=False)
            masks = postprocess_outputs(outputs)
            all_masks.extend(masks)
            all_image_names.extend(image_names)
    return all_masks, all_image_names

# Load saved dataset (CUDA would keep running out of memory)
image_dir = 'data/train_images.npy'
mask_dir = 'data/train_masks.npy'
train_loader = preprocess_images(image_dir, mask_dir, batch_size=5)

model = UNet()
model = model.to('cuda')

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

# Jaccard Loss
class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        jaccard = (intersection + smooth) / (union + smooth)
        return 1 - jaccard

# Initialize the loss function
criterion = JaccardLoss()  # or JaccardLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# %%
# Training loop 
num_epochs = 20
losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(train_loader):
        images = images.to('cuda')
        masks = masks.to('cuda')
        
        optimizer.zero_grad()
        outputs = model(images)
        outputs = nn.functional.interpolate(outputs, size=(SUBIMAGE_SIZE, SUBIMAGE_SIZE), mode='bilinear', align_corners=False)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')

# Plot the training loss
plt.figure()
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# %%
import torch
print(torch.cuda.memory_summary())
torch.cuda.empty_cache()

# %%
# Load trained model
model = UNet()
model.load_state_dict(torch.load('trained_model.pth'))
model = model.to('cuda')  # Move the model to GPU
model.eval()

# Load test dataset
image_dir = 'data/test_images.npy'
mask_dir = 'data/test_masks.npy'

test_loader = preprocess_images(image_dir, mask_dir)
masks, image_names = test_model_batch(model, test_loader)
images, actual_masks = next(iter(test_loader))

fig, ax = plt.subplots(3, 3, figsize=(15, 15))
ax[0, 0].imshow(images[0].cpu().numpy().transpose(1, 2, 0))
ax[0, 1].imshow(masks[0], cmap='gray')
ax[0, 2].imshow(actual_masks[0].cpu().numpy().squeeze(), cmap='gray')
ax[1, 0].imshow(images[1].cpu().numpy().transpose(1, 2, 0))
ax[1, 1].imshow(masks[1], cmap='gray')
ax[1, 2].imshow(actual_masks[1].cpu().numpy().squeeze(), cmap='gray')
ax[2, 0].imshow(images[2].cpu().numpy().transpose(1, 2, 0))
ax[2, 1].imshow(masks[2], cmap='gray')
ax[2, 2].imshow(actual_masks[2].cpu().numpy().squeeze(), cmap='gray')


