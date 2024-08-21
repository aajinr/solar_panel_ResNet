import pandas as pd
from torchvision import datasets, transforms
print ("imported torch vision")

from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Load the Excel file
df = pd.read_excel('annotations.xlsx')

# Custom Dataset Class
class SolarPanelDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create Dataset and DataLoader
dataset = SolarPanelDataset(dataframe=df, image_dir='path/to/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
