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
from attention import ECA_SA_ResNeXtModel
from gradcam import visualize_explanations

import sys

device=torch.device("cpu")

inputpath=sys.argv[1]

model = ECA_SA_ResNeXtModel(num_classes=4)

model.load_state_dict(torch.load("resnext_model.pth"))

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
input = datasets.ImageFolder(root=f"input/", transform=val_transform)
train_dataloader = DataLoader(dataset=input,
                                batch_size=32,
                                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                                shuffle=True)

img, label = next(iter(train_dataloader))

predictions = model(img)
predictions = int(torch.argmax(predictions))
predkey={0:"Common-Rust",1:"Gray Leaf Spot",2:"Blight",3:"Healthy"}
predictions=predkey[predictions]
print(predictions)

labels_for_viz = {0:100,1:100,2:100,3:100}

visualize_explanations(
    model=model, 
    test_dataloader=train_dataloader, 
    labels_for_viz=labels_for_viz, 
    device=device,
    base_path="",
    num_images=5 )

