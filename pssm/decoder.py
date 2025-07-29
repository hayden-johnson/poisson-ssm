import torch
import torch.nn as nn

# MLP decoder going from cumulative spikes -> p(label  | spikes)
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
    def fit(self, train_loader, optimizer, criterion, num_epochs=100):
        self.train()
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                                
                inputs = inputs.float()  # Ensure inputs are float
                labels = labels.long()   # Ensure labels are long
                
                optimizer.zero_grad()
                outputs = self(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
    def evaluate(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy: {100 * correct / total:.2f}%')
    