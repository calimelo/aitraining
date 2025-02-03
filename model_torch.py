import torch

# Define a model
model = torch.nn.Linear(10, 5)
torch.save(model.state_dict(), "model.pth")

# Load model
model.load_state_dict(torch.load("model.pth"))
