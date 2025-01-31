import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim

# Load Pre-trained ResNet Model
model = models.resnet18(pretrained=True)

# Modify the last layer to classify 2 categories (cats & dogs)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Change output layer to 2 classes

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy example: Fine-tune on custom dataset (Cats & Dogs)
# train_loader = ... (Your custom dataset here)
train_data = torchvision.datasets.ImageFolder(
	root="data",
	transform=transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor()
	])
)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

# Training Loop (Fine-Tuning)
for epoch in range(5):  # Fine-tune for 5 epochs
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Fine-Tuning Complete! 🐶🐱")
