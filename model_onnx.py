import torch
import torch.onnx

# Load a PyTorch model
model = torch.load("mnist_model.pth")
model.eval()

# Convert to ONNX format
dummy_input = torch.randn(1, 3, 224, 224)  # Example input shape
torch.onnx.export(model, dummy_input, "mnist_model.onnx")
