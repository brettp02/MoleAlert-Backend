import torch
from torchvision import transforms
from PIL import Image
import os
import torch
from torch import nn

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import models

class ResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet, self).__init__()
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # Freeze all layers except the final FC layer
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load and preprocess the image
image_path = '../data/test/Benign/6792.jpg'  # Replace with your image file path

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

image = Image.open(image_path).convert('RGB')
image = transform(image)
image = image.unsqueeze(0)

def model_pipeline(image: Image):
    # Load the trained model
    model = ResNet()
    model_path = '../app/best_model.pth'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Perform inference
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Interpret the results
    class_names = ['Benign', 'Malignant']
    predicted_class = class_names[predicted.item()]
    output = (f'Predicted class: {predicted_class}')
    return output

