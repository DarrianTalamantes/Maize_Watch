import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch import nn
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import splitfolders
from tqdm import tqdm
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Adjust this path to your local environment
base_path = '/home/nathan/.cache/kagglehub/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset/versions/1'
split_path = f"{base_path}/splitted_data"
original_data_path = f"{base_path}/data"

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Transformation for visualization 
data_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
])

def train_and_validate_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs, device):
    val_best_accuracy = 0.0
    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    
    logger.info("Training begins...")
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        running_train_loss = 0.0
        running_train_accuracy = 0.0
        train_total = 0
        running_val_accuracy = 0.0
        running_val_loss = 0.0
        val_total = 0
        
        # Set model to training mode
        model.train()
        
        # TRAINING LOOP with tqdm progress bar
        logger.info(f"Epoch {epoch}/{num_epochs} - Training...")
        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for data in train_loop:
            inputs, outputs = data
            inputs, outputs = inputs.to(device), outputs.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            train_loss = loss_fn(predictions, outputs)
            _, train_predicted = torch.max(predictions, 1)
            running_train_accuracy += (train_predicted == outputs).sum().item()
            train_total += outputs.size(0)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
            
            # Update progress bar with current loss
            train_loop.set_postfix(loss=train_loss.item())
            
        train_loss_value = running_train_loss/len(train_dataloader)
        train_loss_history.append(train_loss_value)
        train_accuracy = (100*running_train_accuracy)/train_total
        train_accuracy_history.append(train_accuracy)
        
        # VALIDATION LOOP with tqdm progress bar
        logger.info(f"Epoch {epoch}/{num_epochs} - Validating...")
        with torch.no_grad():
            model.eval()
            val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch}/{num_epochs} [Val]")
            for data in val_loop:
                inputs, outputs = data
                inputs, outputs = inputs.to(device), outputs.to(device)
                predictions = model(inputs)
                val_loss = loss_fn(predictions, outputs)
                
                _, val_predicted = torch.max(predictions, 1)
                running_val_loss += val_loss.item()
                val_total += outputs.size(0)
                running_val_accuracy += (val_predicted == outputs).sum().item()
                
                # Update progress bar with current loss
                val_loop.set_postfix(loss=val_loss.item())
                
        val_loss_value = running_val_loss/len(val_dataloader)
        val_loss_history.append(val_loss_value)
        val_accuracy = (100*running_val_accuracy)/val_total
        val_accuracy_history.append(val_accuracy)
        
        epoch_time = time.time() - epoch_start_time
        
        if val_accuracy > val_best_accuracy:
            torch.save(model.state_dict(), f"resnext_model.pth")
            val_best_accuracy = val_accuracy
            logger.info(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
        
        logger.info(f"Epoch: {epoch}/{num_epochs} - Time: {epoch_time:.1f}s - Train Acc: {train_accuracy:.2f}% - Val Acc: {val_accuracy:.2f}% - Train Loss: {train_loss_value:.4f} - Val Loss: {val_loss_value:.4f}")
    
    return train_accuracy_history, val_accuracy_history, train_loss_history, val_loss_history

def test_model(model, test_dataloader, labels_for_viz, device):
    logger.info("Loading best model for testing...")
    # Load our trained model
    path = f"{base_path}/resnext_model.pth"
    model.load_state_dict(torch.load(path))
    model.eval()
    
    running_accuracy = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    logger.info("Starting evaluation on test set...")
    with torch.no_grad():
        test_loop = tqdm(test_dataloader, desc="Testing")
        for data in test_loop:
            inputs, outputs = data
            inputs, outputs = inputs.to(device), outputs.to(device)
            predictions = model(inputs)
            _, predicted = torch.max(predictions, 1)
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()
            
            # Store predictions and true labels for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(outputs.cpu().numpy())
            
            # Update progress bar
            current_acc = 100 * (predicted == outputs).sum().item() / outputs.size(0)
            test_loop.set_postfix(current_batch_acc=f"{current_acc:.2f}%")
            
    # Calculate overall accuracy
    accuracy = 100 * running_accuracy / total
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    
    # Calculate additional metrics
    class_names = list(labels_for_viz.values())
    
    # Generate classification report
    logger.info("\nClassification Report:")
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    logger.info(f"\n{report}")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig("confuse.png")
    # Calculate per-class metrics
    precision = precision_score(all_labels, all_predictions, average=None)
    recall = recall_score(all_labels, all_predictions, average=None)
    f1 = f1_score(all_labels, all_predictions, average=None)
    
    # Print metrics for each class
    logger.info("\nPer-class Metrics:")
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    logger.info(f"\n{metrics_df}")
    
    return np.array(all_predictions)

if __name__ == '__main__':
    # Create split if it doesn't exist
    if not os.path.exists(split_path):
        logger.info(f"Creating split at {split_path}...")
        splitfolders.ratio(original_data_path,
                        output=split_path,
                        seed=42,
                        ratio=(.7, .2, .1),
                        group_prefix=None,
                        move=False)
        logger.info("Split complete!")
    
    # Create the datasets
    logger.info("Loading datasets...")
    data = datasets.ImageFolder(root=original_data_path, transform=data_transform)
    train = datasets.ImageFolder(root=f"{split_path}/train", transform=train_transform)
    val = datasets.ImageFolder(root=f"{split_path}/val", transform=val_transform)
    test = datasets.ImageFolder(root=f"{split_path}/test", transform=test_transform)
    
    # Check for MPS (Metal Performance Shaders) availability on Mac
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device= torch.device("cuda")
    logger.info(f"Using device: {device}")
    
    # Create labels dictionary
    labels_for_viz = {v: k for k, v in data.class_to_idx.items()}
    logger.info(f"Classes: {list(labels_for_viz.values())}")
    
    # Count images per class
    logger.info("Image count per class:")
    for dataset_name, dataset in [("Train", train), ("Validation", val), ("Test", test)]:
        class_counts = pd.Series([labels_for_viz[y] for y in dataset.targets]).value_counts()
        logger.info(f"{dataset_name} set: {class_counts.to_dict()}")
    
    # Create DataLoaders with num_workers=0 to avoid multiprocessing issues
    logger.info("Creating data loaders...")
    train_dataloader = DataLoader(dataset=train,
                                batch_size=32,
                                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                                shuffle=True)
    
    val_dataloader = DataLoader(dataset=val,
                                batch_size=32,
                                num_workers=0,
                                shuffle=True)
    
    test_dataloader = DataLoader(dataset=test,
                                batch_size=32,
                                num_workers=0,
                                shuffle=False)
    
    # Check a batch
    img, label = next(iter(train_dataloader))
    logger.info(f"Batch and Image Shape: {img.shape} --> [batch_size, color_channels, height, width]")
    logger.info(f"Labels: {label}")
    
    # Use ResNeXt model with pretrained weights
    logger.info("Initializing ResNeXt50 model with pretrained weights...")
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(in_features=2048, out_features=4)  # ResNeXt50 has 2048 features
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=10e-4)
    
    # Train the model
    EPOCHS = 100
    #EPOCHS = 45
    logger.info(f"Starting training for {EPOCHS} epochs...")
    train_accuracy_history, val_accuracy_history, train_loss_history, val_loss_history = train_and_validate_model(
        model, train_dataloader, val_dataloader, loss_fn, optimizer, EPOCHS, device
    )
    logger.info("Training finished...\n")
    
    # Plot training and validation accuracy
    logger.info("Creating performance plots...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS+1), train_accuracy_history, label='Train Accuracy')
    plt.plot(range(1, EPOCHS+1), val_accuracy_history, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.savefig("accuracy.png")

    
    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS+1), train_loss_history, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    plt.savefig("loss.png")
    
    # Initialize a new model for testing
    test_model_instance = models.resnext50_32x4d(weights=None)
    test_model_instance.fc = nn.Linear(in_features=2048, out_features=4)
    test_model_instance = test_model_instance.to(device)
    
    # Test the model and get predictions
    logger.info("Starting model evaluation...")
    all_preds = test_model(test_model_instance, test_dataloader, labels_for_viz, device)
    logger.info("Model evaluation complete!")