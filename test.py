import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models

# Check if CUDA is available, otherwise use CPU
device = torch.device("cpu")
print(f"Using device: {device}")

resnet_model = models.resnet50(pretrained=True)

for param in resnet_model.parameters():
    param.requires_grad = True

n_inputs = resnet_model.fc.in_features

resnet_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 4),
                                nn.LogSigmoid())

for name, child in resnet_model.named_children():
    for name2, params in child.named_parameters():
        params.requires_grad = True

resnet_model.to(device)

# Load the model with a CPU fallback
try:
    state_dict = torch.load('models\\bt_resnet50_model.pt', map_location=device)
    resnet_model.load_state_dict(state_dict)
    print("Model loaded successfully.")
except RuntimeError as e:
    print(f"Error loading model: {e}")
    print("Ensure the model was saved in a format compatible with the current environment.")
    exit()

resnet_model.eval()

transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

LABELS = ['None', 'Meningioma', 'Glioma', 'Pitutary']

img_name = input("Enter path to the image: ")

if not os.path.exists(img_name):
    print("File does not exist. Exiting...\n")
    exit()

img = Image.open(img_name)

img = transform(img)

img = img[None, ...]

with torch.no_grad():
    y_hat = resnet_model(img.to(device))
    predicted = torch.argmax(y_hat.data, dim=1)
    print(LABELS[predicted.item()], '\n')