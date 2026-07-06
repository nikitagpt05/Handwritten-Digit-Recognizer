import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# --- 1. CNN Architecture ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model():
    # --- 2. Data Preparation ---
    # Reduced rotation to 5 degrees for cleaner learning
    transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Loading dataset...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # --- 3. Setup Model, Optimizer, Loss ---
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 15  # Increased to 15 for better accuracy
    print(f"Starting training for {epochs} epochs...")

    # --- 4. The Training Loop ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # New counters for accuracy
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy for this specific batch
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate the final averages for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")

    # --- 5. Save the weights ---
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("\nTraining complete. New model saved as 'mnist_cnn.pth'")


if __name__ == '__main__':
    train_model()