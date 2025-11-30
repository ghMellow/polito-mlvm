<VSCode.Cell id="#VSC-markdown-1" language="markdown">
# Lab 3 - Transfer Learning with VGG16
## Cats vs Dogs Classification
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-2" language="markdown">
# Import libraries
</VSCode.Cell>

<VSCode.Cell id="#VSC-imports" language="python">
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import copy
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import kagglehub
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-3" language="markdown">
# 1. Transfer Learning Concepts

Transfer learning in Pytorch can be implemented in two steps:
- Create the new model by detaching the old classification head and attaching the new head
- Selectively "freeze" and "unfreeze" layers in order to use them as fixed feature extractors or finetuning them
  - Each parameter has a requires_grad property: when set to False, gradients are not computed
  - To freeze all layers: `for param in model.parameters(): param.requires_grad = False`
  - The summary() method prints whether a layer is trainable or not
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-4" language="markdown">
# 2. Load Pre-trained VGG16 Model
</VSCode.Cell>

<VSCode.Cell id="#VSC-model-load" language="python">
# Load pre-trained VGG16 model
model_ft = models.vgg16(weights='IMAGENET1K_V1')

# Define number of classes for our task (cats vs dogs)
num_classes = 2

# Replace the last layer with a new one for our classification task
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

print("Model loaded successfully!")
print(f"Original final layer input features: {num_ftrs}")
print(f"New final layer output classes: {num_classes}")
</VSCode.Cell>

<VSCode.Cell id="#VSC-model-summary-1" language="python">
# View model architecture before freezing
summary(model_ft, (3, 224, 224))
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-5" language="markdown">
# 3. Freeze Base Layers (Feature Extraction Mode)
</VSCode.Cell>

<VSCode.Cell id="#VSC-freeze-layers" language="python">
# Freeze all layers first
for param in model_ft.parameters():
    param.requires_grad = False

# Unfreeze only the parameters of the last layer
for param in model_ft.classifier[6].parameters():
    param.requires_grad = True

print("✓ All base layers frozen")
print("✓ Only final classifier layer is trainable")
</VSCode.Cell>

<VSCode.Cell id="#VSC-model-summary-2" language="python">
# View model architecture after freezing
summary(model_ft, (3, 224, 224))
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-6" language="markdown">
# 4. Download Dataset from Kaggle
</VSCode.Cell>

<VSCode.Cell id="#VSC-download-dataset" language="python">
# Download latest version of cats and dogs dataset
path = kagglehub.dataset_download("tongpython/cat-and-dog")

print("Path to dataset files:", path)
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-7" language="markdown">
# 5. Prepare Dataset and Create Annotations CSV
</VSCode.Cell>

<VSCode.Cell id="#VSC-create-annotations" language="python">
# Define dataset paths
dataset_path = path  # Use the downloaded path
train_path = os.path.join(dataset_path, 'training_set', 'training_set')
test_path = os.path.join(dataset_path, 'test_set', 'test_set')

def create_annotations_csv(data_dir, output_csv):
    """
    Create CSV file with image filenames and labels
    
    Args:
        data_dir: Directory containing 'cats' and 'dogs' subdirectories
        output_csv: Output CSV filename
    
    Returns:
        DataFrame with annotations
    """
    data = []

    # Process dog images
    dogs_dir = os.path.join(data_dir, 'dogs')
    if os.path.exists(dogs_dir):
        for img_name in os.listdir(dogs_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append({
                    'filename': os.path.join('dogs', img_name),
                    'label': 1  # 1 for dogs
                })

    # Process cat images
    cats_dir = os.path.join(data_dir, 'cats')
    if os.path.exists(cats_dir):
        for img_name in os.listdir(cats_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append({
                    'filename': os.path.join('cats', img_name),
                    'label': 0  # 0 for cats
                })

    # Create DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Created {output_csv} with {len(df)} images")
    print(f"Cats: {len(df[df['label'] == 0])}, Dogs: {len(df[df['label'] == 1])}")
    return df

# Create CSV files for training and test sets
train_annotations = create_annotations_csv(train_path, 'train_annotations.csv')
test_annotations = create_annotations_csv(test_path, 'test_annotations.csv')
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-8" language="markdown">
# 6. Custom Dataset Class
</VSCode.Cell>

<VSCode.Cell id="#VSC-dataset-class" language="python">
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        Custom Dataset for loading images from CSV annotations
        
        Args:
            annotations_file: Path to CSV file with annotations
            img_dir: Directory containing the images
            transform: Transformations to apply to images
            target_transform: Transformations to apply to labels
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        
        # Load image using PIL
        image = Image.open(img_path).convert('RGB')
        
        # Get label
        label = self.img_labels.iloc[idx, 1]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

print("✓ CustomImageDataset class defined")
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-9" language="markdown">
# 7. Define Image Transformations
</VSCode.Cell>

<VSCode.Cell id="#VSC-transforms" language="python">
# Training transformations (with data augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),      # Random crop and resize to 224x224
    transforms.RandomHorizontalFlip(),       # Random horizontal flip with p=0.5
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                        std=[0.229, 0.224, 0.225])
])

# Test transformations (without augmentation)
test_transform = transforms.Compose([
    transforms.Resize(256),                  # Resize to 256
    transforms.CenterCrop(224),              # Center crop to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

print("✓ Transformations defined")
print("  - Training: RandomResizedCrop + RandomHorizontalFlip + Normalize")
print("  - Test: Resize + CenterCrop + Normalize")
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-10" language="markdown">
# 8. Create Datasets and DataLoaders
</VSCode.Cell>

<VSCode.Cell id="#VSC-dataloaders" language="python">
# Create datasets
train_dataset = CustomImageDataset(
    annotations_file='train_annotations.csv',
    img_dir=train_path,
    transform=train_transform
)

test_dataset = CustomImageDataset(
    annotations_file='test_annotations.csv',
    img_dir=test_path,
    transform=test_transform
)

# Define batch size
batch_size = 32

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: {batch_size}")
print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-11" language="markdown">
# 9. Visualize Sample Images
</VSCode.Cell>

<VSCode.Cell id="#VSC-visualize" language="python">
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Get a batch of images
images, labels = next(iter(train_loader))

# Visualize first 8 images
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.ravel()

for idx in range(8):
    # Denormalize image
    img = images[idx].clone()
    img = denormalize(img)
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    
    # Display
    axes[idx].imshow(img)
    axes[idx].set_title(f"{'Cat' if labels[idx] == 0 else 'Dog'}")
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-12" language="markdown">
# 10. Training Setup
</VSCode.Cell>

<VSCode.Cell id="#VSC-training-setup" language="python">
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to device
model_ft = model_ft.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer - only parameters with requires_grad=True
optimizer_ft = optim.Adam(
    filter(lambda p: p.requires_grad, model_ft.parameters()), 
    lr=0.001
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print("✓ Training setup complete")
print(f"  - Loss: CrossEntropyLoss")
print(f"  - Optimizer: Adam (lr=0.001)")
print(f"  - Scheduler: StepLR (step_size=7, gamma=0.1)")
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-13" language="markdown">
# 11. Training Function
</VSCode.Cell>

<VSCode.Cell id="#VSC-train-function" language="python">
def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=10):
    """
    Train the model and return training history
    
    Args:
        model: PyTorch model
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        dataloaders: Dictionary with 'train' and 'val' dataloaders
        num_epochs: Number of training epochs
    
    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Dictionary to store loss and accuracy
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = dataloaders['train']
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = dataloaders['val']
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            pbar = tqdm(dataloader, desc=f'{phase.capitalize()} ')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save to history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # Deep copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'✓ New best model! Validation Accuracy: {best_acc:.4f}')
    
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

print("✓ Training function defined")
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-14" language="markdown">
# 12. Prepare DataLoaders Dictionary
</VSCode.Cell>

<VSCode.Cell id="#VSC-dataloaders-dict" language="python">
# Create dataloaders dictionary for training
dataloaders = {
    'train': train_loader,
    'val': test_loader  # Using test set as validation
}

print("✓ DataLoaders dictionary created")
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-15" language="markdown">
# 13. Train the Model (Feature Extraction)
</VSCode.Cell>

<VSCode.Cell id="#VSC-train-model" language="python">
print("=" * 60)
print("PHASE 1: Feature Extraction (only last layer trainable)")
print("=" * 60)

num_epochs_feature_extraction = 10

model_ft_trained, history_ft = train_model(
    model_ft, 
    criterion, 
    optimizer_ft, 
    scheduler,
    dataloaders,
    num_epochs=num_epochs_feature_extraction
)
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-16" language="markdown">
# 14. Plot Training History
</VSCode.Cell>

<VSCode.Cell id="#VSC-plot-history" language="python">
def plot_training_history(history, title='Training History'):
    """Plot training and validation loss and accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(history['val_acc'], label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_training_history(history_ft, 'Feature Extraction (Frozen Base)')
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-17" language="markdown">
# 15. Evaluation Function
</VSCode.Cell>

<VSCode.Cell id="#VSC-eval-function" language="python">
def evaluate_model(model, dataloader, device):
    """
    Evaluate model and return predictions and labels
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
    
    Returns:
        accuracy: Overall accuracy
        all_preds: All predictions
        all_labels: All true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Cats', 'Dogs']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Cats', 'Dogs'],
                yticklabels=['Cats', 'Dogs'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Calculate accuracy
    accuracy = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    return accuracy, all_preds, all_labels

print("✓ Evaluation function defined")
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-18" language="markdown">
# 16. Final Evaluation
</VSCode.Cell>

<VSCode.Cell id="#VSC-final-eval" language="python">
print("=" * 60)
print("FINAL EVALUATION - Feature Extraction")
print("=" * 60)

final_accuracy, preds, labels = evaluate_model(model_ft_trained, test_loader, device)
print(f"\nFinal Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-19" language="markdown">
# 17. Save the Model
</VSCode.Cell>

<VSCode.Cell id="#VSC-save-model" language="python">
# Save model weights
torch.save(model_ft_trained.state_dict(), 'vgg16_cats_dogs_feature_extraction.pth')
print("✓ Model saved as 'vgg16_cats_dogs_feature_extraction.pth'")

# To load the model later:
# model_ft.load_state_dict(torch.load('vgg16_cats_dogs_feature_extraction.pth'))
# model_ft.eval()
</VSCode.Cell>

<VSCode.Cell id="#VSC-markdown-20" language="markdown">
# Summary

This notebook demonstrates:
1. **Transfer Learning** using pre-trained VGG16
2. **Feature Extraction** by freezing base layers
3. **Fine-tuning** only the final classification layer
4. **Custom Dataset** creation from Kaggle data
5. **Data Augmentation** for better generalization
6. **Training pipeline** with validation
7. **Evaluation metrics** including confusion matrix

The model achieves good accuracy by leveraging features learned from ImageNet!
</VSCode.Cell>
