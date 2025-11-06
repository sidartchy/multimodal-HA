"""
Data utilities for multimodal health assistant project.
Contains reusable components for data loading, preprocessing, and dataset classes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
# import collections
# import collections.abc
# collections.Iterable = collections.abc.Iterable
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import warnings
warnings.filterwarnings('ignore')


class SkinLesionDataset(Dataset):
    """Custom dataset for skin lesion images with metadata"""
    
    def __init__(self, df, image_dirs, transform=None, target_transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dirs = image_dirs
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['img_id']
        
        # Find and load image
        img = None
        for part, dir_path in self.image_dirs.items():
            full_path = os.path.join('archive', dir_path, img_id)
            if os.path.exists(full_path):
                try:
                    img = Image.open(full_path).convert('RGB')
                    break
                except Exception as e:
                    print(f"Error loading {img_id} from {part}: {e}")
                    continue
        
        if img is None:
            # Create a placeholder image if not found
            img = Image.new('RGB', (224, 224), color='black')
            print(f"Warning: Image {img_id} not found, using placeholder")
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Get target (diagnostic category)
        target = row['diagnostic']
        if self.target_transform:
            target = self.target_transform(target)
        
        # Get metadata for multimodal approach
        metadata = {
            'age': row['age'],
            'gender': row['gender'],
            'region': row['region'],
            'symptom_score': row['symptom_score'],
            'risk_score': row['risk_score'],
            'text_description': row['text_description']
        }
        
        return img, target, metadata


def get_image_transforms():
    """Get image transforms for training and validation"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def load_processed_data():
    """Load preprocessed data"""
    
    # Load dataframes
    train_df = pd.read_csv('processed_data/train_df.csv')
    val_df = pd.read_csv('processed_data/val_df.csv')
    test_df = pd.read_csv('processed_data/test_df.csv')
    
    # Load label encoder
    with open('processed_data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load preprocessing config
    with open('processed_data/preprocessing_config.json', 'r') as f:
        config = json.load(f)
    
    return train_df, val_df, test_df, label_encoder, config



def load_processed_data_v2():
    """Load preprocessed data ---   V2"""
    
    # Load dataframes
    train_df = pd.read_csv('processed_data_v2/train_df.csv')
    val_df = pd.read_csv('processed_data_v2/val_df.csv')
    test_df = pd.read_csv('processed_data_v2/test_df.csv')
    
    # Load label encoder
    with open('processed_data_v2/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load preprocessing config
    with open('processed_data_v2/preprocessing_config.json', 'r') as f:
        config = json.load(f)
    
    return train_df, val_df, test_df, label_encoder, config

def custom_collate_fn(batch):
    """Custom collate function to handle metadata dictionaries"""
    images = torch.stack([item[0] for item in batch])
    # Don't convert targets to tensor here - they might be strings
    targets = [item[1] for item in batch]  # Keep as list
    metadatas = [item[2] for item in batch]  # Keep as list of dicts
    
    return images, targets, metadatas


def create_data_loaders(train_df, val_df, test_df, batch_size=32):
    """Create PyTorch DataLoaders for train, validation, and test sets"""
    
    # Image directories
    IMAGE_DIRS = {
        'part1': 'imgs_part_1/imgs_part_1',
        'part2': 'imgs_part_2/imgs_part_2', 
        'part3': 'imgs_part_3/imgs_part_3'
    }
    
    # Get transforms
    train_transform, val_transform = get_image_transforms()
    
    # Create datasets
    train_dataset = SkinLesionDataset(train_df, IMAGE_DIRS, transform=train_transform)
    val_dataset = SkinLesionDataset(val_df, IMAGE_DIRS, transform=val_transform)
    test_dataset = SkinLesionDataset(test_df, IMAGE_DIRS, transform=val_transform)
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=custom_collate_fn  # Add this
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn  # Add this
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn  # Add this
    )
    
    return train_loader, val_loader, test_loader



def plot_training_history(history):
    """Plot training and validation loss/accuracy curves"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy', color='blue')
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    """Plot confusion matrix"""
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return cm

