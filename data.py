import glob
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np
import cv2 as cv
from config import load_config

train_transforms = A.Compose(
    [
        A.Resize(height=384,width=384),
        # color changed
        A.OneOf([
            A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
            A.GaussianBlur(sigma_limit=(0, 3.0), p=0.5)
        ], p=0.3),
        # complex changed
        A.OneOf([
            A.ColorJitter(contrast=0.2),
            A.ColorJitter(saturation=0.2),
        ], p=0.3),
        ToTensor(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(height=384, width=384),
        ToTensor()
    ]
)

class DeeperDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        #print(img_path)
        img_transformed = self.transform(image = np.array(img))
        label = img_path.split("/")[-1].split(".")[0].split("_")[-1]
        label = 1 if label == "fake" else 0

        return img_transformed, label

def load_data(args):

    train_dir = '/home/liu/deepfake_detection/VIT/data/FF++/val_all/c23/'
    train_list = glob.glob(os.path.join(train_dir,'*6_*_*.png'))
    val_dir = '/home/liu/deepfake_detection/VIT/data/FF++/val_all/c23/'
    val_list = glob.glob(os.path.join(val_dir,'*8_*_*.png'))
    print(len(train_list))
    print(len(val_list))

    train_data = DeeperDataset(train_list, transform=train_transforms)
    valid_data = DeeperDataset(val_list, transform=val_transforms)

    train_loader = DataLoader(dataset = train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset = valid_data, batch_size=args.batch_size, shuffle=True)

    return train_loader, valid_loader

if __name__ == '__main__':
    args = load_config()
    train_data, _ = load_data(args)
    for data, label in tqdm(train_data):
        data = data['image']
        #data = data.to('cuda')
        #label = label.to('cuda')
        print(data[0].shape)
        data = np.uint8(data[0])
        img = np.stack([data[0, :, :], data[1, :, :], data[2, :, :]], axis=2)
        print(img.shape)
        data = Image.fromarray(img)
        data.save('test.png')
        break

