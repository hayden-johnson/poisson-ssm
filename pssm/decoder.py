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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    
    def train(self, train_loader, num_epochs=100):
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
    def evaluate(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')
    

def train_encoder():
    # Create an instance of the MLP
    mlp = MLP(input_size=input_size, output_size=output_size)
    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Instead of logistic regression, since logistic is for binary classification
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

    # Train the MLP
    mlp.train(spike_trains, num_epochs=100)

    # Evaluate the MLP
    mlp.evaluate(spike_train_loader)