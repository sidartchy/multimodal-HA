"""
Model utilities for multimodal health assistant project.
Contains CNN architectures, training functions, and evaluation utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torchvision.models as models
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os


class CNNBaseline(nn.Module):
    """CNN baseline model for skin lesion classification"""
    
    def __init__(self, num_classes, model_name='resnet34', pretrained=True):
        super(CNNBaseline, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        
        self.backbone = models.resnet34(pretrained=True)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes) 
    
    def forward(self, x):
        return self.backbone(x)
    

class EfficientNetBaseline(nn.Module):
    def __init__(self, model_name= 'EfficientNet_Base', pretrained=True):
        super(EfficientNetBaseline, self).__init__()

        self.model_name = model_name

        # Load pretrained backbone
        if pretrained:
            self.backbone = models.efficientnet_b2(
                weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.efficientnet_b2(weights=None)

        # Unfreeze last feature blocks
        for name, param in self.backbone.named_parameters():
            if "features.6" in name or "features.7" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] =  nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 128)
        )

    def forward(self, x):
        return self.backbone(x)




class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, model_name= 'EfficientNet_Base', pretrained=True):
        super(EfficientNetClassifier, self).__init__()

        self.model_name = model_name

        # Load pretrained backbone
        if pretrained:
            self.backbone = models.efficientnet_b2(
                weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.efficientnet_b2(weights=None)

        # Unfreeze last feature blocks
        for name, param in self.backbone.named_parameters():
            if "features.6" in name or "features.7" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] =  nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)



class HybridImageDominantFusion(nn.Module):
    def __init__(
        self,
        image_model,
        text_embedding_dim=768,
        image_embedding_dim=128,
        reduced_text_dim=64,
        num_classes=2,
        image_weight=0.75,  
        fusion_type='weighted'    ## this is preferred
    ):
        super(HybridImageDominantFusion, self).__init__()
        self.image_model = image_model
        self.image_weight = image_weight
        self.fusion_type = fusion_type

        # Image Encoder
        for param in self.image_model.parameters():
            param.requires_grad = False  # freeze image model (or unfreeze later for fine-tuning)

    
        # compress text embeddings to smaller dimension ( projecting text embeddings to lower dim)
        self.text_projector = nn.Sequential(
            nn.Linear(text_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, reduced_text_dim),
            nn.ReLU()
        )

        # Fusion logic
        # If concat, we just join both
        if fusion_type == 'concat':
            fusion_input_dim = image_embedding_dim + reduced_text_dim
        elif fusion_type == 'weighted':
            # both projected to same dim to allow weighted sum
            self.image_projection = nn.Linear(image_embedding_dim, reduced_text_dim)
            fusion_input_dim = reduced_text_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, text_embedding):
        # Get image and text representations
        image_embedding = self.image_model(image)         # [batch, 128]
        text_embedding = self.text_projector(text_embedding)  # [batch, reduced_text_dim]

        if self.fusion_type == 'concat':
            # Concatenate but text is already low-dimensional
            fused = torch.cat((image_embedding, text_embedding), dim=1)

        elif self.fusion_type == 'weighted':
            # Project image embedding to same dim as text for weighted fusion
            image_proj = self.image_projection(image_embedding)  # [batch, reduced_text_dim]

            # Weighted hybrid fusion: image dominant
            fused = self.image_weight * image_proj + (1 - self.image_weight) * text_embedding

        output = self.classifier(fused)
        return output


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, 
                device='cuda', patience=10, model_name='cnn_baseline', label_encoder=None):
    """Train the CNN model with early stopping"""
    
    # Move model to device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    
    print(f"Device available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    if  model.model_name:
        print(f"Model: {model.model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, targets, metadata in train_pbar:
            # Encode targets if they are strings
            if label_encoder is not None:
                targets = torch.tensor(label_encoder.transform(targets), dtype=torch.long)
            else:
                targets = torch.tensor(targets, dtype=torch.long)
            
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, targets, metadata in val_pbar:
                # Encode targets if they are strings
                if label_encoder is not None:
                    targets = torch.tensor(label_encoder.transform(targets), dtype=torch.long)
                else:
                    targets = torch.tensor(targets, dtype=torch.long)
                
                images, targets = images.to(device), targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'models/{model_name}_best.pth')
            print(f'New best model saved! Val Acc: {val_acc:.2f}%')
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        
    
    return history


def evaluate_model(model, test_loader, device='cuda', class_names=None, label_encoder=None):
    """Evaluate the trained model on test set"""
    
    # Handle device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    print("Evaluating model on test set...")
    
    with torch.no_grad():
        for images, targets, metadata in tqdm(test_loader, desc='Testing'):
            # Encode targets if they are strings
            if label_encoder is not None:
                targets = torch.tensor(label_encoder.transform(targets), dtype=torch.long)
            else:
                targets = torch.tensor(targets, dtype=torch.long)
            
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    
    print(f"Test Results:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    if class_names:
        print(f"Classification Report:")
        report = classification_report(all_targets, all_predictions, 
                                    target_names=class_names, digits=4)
        print(report)
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'accuracy': accuracy
    }


def plot_training_history(history):
    """Plot training and validation loss/accuracy curves"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy', color='blue', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='red', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return cm
