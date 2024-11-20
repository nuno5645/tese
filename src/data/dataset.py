import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from skimage.draw import polygon
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ChromosomeDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train', img_size=256):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.img_size = img_size
        
        # Get all valid image-json pairs
        self.valid_pairs = []
        for root, _, files in os.walk(data_dir):
            json_files = [f for f in files if f.endswith('.json')]
            for json_file in json_files:
                json_path = os.path.join(root, json_file)
                img_path = json_path.replace('.json', '.png')
                
                # Only add if both JSON and image exist
                if os.path.exists(img_path):
                    self.valid_pairs.append((img_path, json_path))
        
        print(f"Found {len(self.valid_pairs)} valid image-annotation pairs in {data_dir}")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        # Load files
        img_path, json_path = self.valid_pairs[idx]
        
        try:
            # Load and resize image
            image = Image.open(img_path)
            image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            image = np.array(image)
            
            # Create mask with reduced size - now with 24 channels (23 chromosomes + background)
            mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)
            
            # Load annotations
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Process each shape in the annotation
            for shape in data['shapes']:
                points = np.array(shape['points'])
                
                # Scale points to match new image size
                points[:, 0] = points[:, 0] * self.img_size / data['imageWidth']
                points[:, 1] = points[:, 1] * self.img_size / data['imageHeight']
                
                # Get chromosome class from label (assuming label format "chromosome_X")
                try:
                    # Extract chromosome number from label
                    if 'chromosome_' in shape['label'].lower():
                        class_idx = int(shape['label'].lower().replace('chromosome_', ''))
                    else:
                        # If label doesn't follow format, use default class 1
                        class_idx = 1
                except ValueError:
                    # If conversion fails, use default class 1
                    class_idx = 1
                
                # Ensure class_idx is within valid range (1-23)
                class_idx = max(1, min(23, class_idx))
                
                # Create polygon mask for this chromosome
                rr, cc = polygon(points[:, 1], points[:, 0], shape=(self.img_size, self.img_size))
                mask[rr, cc] = class_idx
            
            # Apply transformations if any
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            
            return image, mask
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            # Return a zero image and mask in case of error
            if self.transform:
                image = torch.zeros(3, self.img_size, self.img_size)
                mask = torch.zeros(self.img_size, self.img_size)
            else:
                image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)
            return image, mask

def get_transform(mode='train', img_size=256):
    if mode == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
