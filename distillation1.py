import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ✅ Step 1: Load Dataset (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# ✅ Step 2: Define the Teacher Model (Larger Network)
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)

# ✅ Step 3: Define the Student Model (Smaller Network)
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)

# ✅ Step 4: Train the Teacher Model
def train_teacher(model, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Teacher Loss: {total_loss / len(train_loader):.4f}")

# ✅ Step 5: Define Knowledge Distillation Loss
def distillation_loss(student_output, teacher_output, labels, temperature=2.0, alpha=0.5):
    """
    Computes the distillation loss:
    - Uses teacher's soft labels
    - Blends with standard classification loss
    """
    soft_labels = nn.functional.softmax(teacher_output / temperature, dim=1)
    soft_loss = nn.functional.kl_div(
        nn.functional.log_softmax(student_output / temperature, dim=1),
        soft_labels,
        reduction="batchmean"
    )
    
    hard_loss = nn.functional.cross_entropy(student_output, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss

# ✅ Step 6: Train the Student Model Using Distillation
def train_student(student, teacher, optimizer, epochs=5):
    student.train()
    teacher.eval()  # Teacher model remains fixed

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            teacher_output = teacher(images).detach()  # Get teacher's predictions
            student_output = student(images)
            loss = distillation_loss(student_output, teacher_output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Student Loss: {total_loss / len(train_loader):.4f}")

# ✅ Step 7: Evaluate the Models
def evaluate(model, name="Model"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"{name} Accuracy: {100 * correct / total:.2f}%")

# ✅ Step 8: Run the Full Training Process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher = TeacherModel().to(device)
student = StudentModel().to(device)

teacher_optimizer = optim.Adam(teacher.parameters(), lr=0.001)
student_optimizer = optim.Adam(student.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

import time
print("\n🚀 Training Teacher Model...")
teacherstarttime = time.time()
train_teacher(teacher, teacher_optimizer, criterion, epochs=5)
teacherendtime = time.time()
print(f"Teacher Training Time: {teacherendtime-teacherstarttime:.2f} seconds")

print("\n🏆 Evaluating Teacher Model...")
teacherstarttime = time.time()
evaluate(teacher, name="Teacher")
teacherendtime = time.time()
print(f"Teacher Evaluation Time: {teacherendtime-teacherstarttime:.2f} seconds")

print("\n🔥 Training Student Model with Distillation...")
studentstarttime = time.time()
train_student(student, teacher, student_optimizer, epochs=5)
studentendtime = time.time()
print(f"Student Training Time: {studentendtime-studentstarttime:.2f} seconds")

print("\n📊 Evaluating Student Model...")
studentstarttime = time.time()
evaluate(student, name="Student")
studentendtime = time.time()
print(f"Student Evaluation Time: {studentendtime-studentstarttime:.2f} seconds")

#save the models
torch.save(teacher.state_dict(), 'teacher.pth')
torch.save(student.state_dict(), 'student.pth')

#compare the sizes of the models
import os
teacher_size = os.path.getsize('teacher.pth')
student_size = os.path.getsize('student.pth')

print(f"\n📊 Teacher Model Size: {teacher_size} bytes"
      f"\n📊 Student Model Size: {student_size} bytes")

