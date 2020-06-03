from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
import torchvision
from torch import nn
import torch
import shutil
from zipfile import ZipFile 
import zipfile

class MasterData(Dataset):
    def __init__(self, data_root, transform=None,scale_transform=None,grayTransform=None,f1=Path('bg/'), f2=Path('fg_bg/'), f3=Path('mask_fg_bg/') ,f4=Path('depth/')):
        self.f1_files = list(f1.glob('*.jpeg'))
        self.f2_files = list(f2.glob('*.jpeg'))
        self.f3_files = list(f3.glob('*.jpeg'))
        self.f4_files = list(f4.glob('*.jpeg'))
        self.transform = transform
        self.scale_transform = scale_transform
        self.grayTransform = grayTransform

    def __len__(self):
        return len(self.f1_files)

    def __getitem__(self, index):
        f1_image = Image.open(self.f1_files[index])
        f2_image = Image.open(self.f2_files[index])
        f3_image = Image.open(self.f3_files[index])
        f4_image = Image.open(self.f4_files[index])
        #do transformations here
        if self.transform:
            f1_image = self.transform(f1_image)
            f2_image = self.transform(f2_image)
            f3_image = self.grayTransform(f3_image)
            f4_image = self.grayTransform(f4_image)
        return {'f1': f1_image, 'f2': f2_image, 'f3': f3_image, 'f4': f4_image}

def importDataset():
    shutil.copy('/content/drive/My Drive/bg_fg_bg_mask_2.zip','bg_fg_bg_mask_2.zip')
    shutil.copy('/content/drive/My Drive/Session15/copy_depth_output.zip','copy_depth_output.zip')
    zip = ZipFile('bg_fg_bg_mask_2.zip')
    zip.extractall('./')
    zip = ZipFile('copy_depth_output.zip')
    zip.extractall('./')
    data_root = Path('.')
    f1, f2, f3 ,f4 = data_root/'bg', data_root/'fg_bg', data_root/'mask_fg_bg', data_root/'depth' 
    print(len(list(f1.iterdir())))
    print(len(list(f2.iterdir())))
    print(len(list(f3.iterdir())))
    print(len(list(f4.iterdir())))
    scale_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    #transforms.ColorJitter(brightness=0.2, contrast = 0.2, saturation = 0.2, hue = 0.2),
    transforms.ToTensor(),                                    
    ])
    grayTransform  = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        #transforms.ColorJitter(brightness=0.2, contrast = 0.2, saturation = 0.2, hue = 0.2),
        transforms.ToTensor(),                                    
      ])
    mean, std = torch.tensor([0.485, 0.456, 0.406])*255, torch.tensor([0.229, 0.224, 0.225])*255
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue= 0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_ds = MasterData(data_root, train_transforms, scale_transform)
    train_d1 = DataLoader(train_ds, batch_size=16, shuffle=True, pin_memory=True)
    return train_d1
