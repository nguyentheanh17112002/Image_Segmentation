import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import f1_score

class SegmentationModel(nn.Module):
    def __init__(self, input_size = 19, num_classes = 7):
        super(SegmentationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)
    
def train(train_dataset, num_epochs, batch_size, lr):
    print("Start training model")
    print("=========================================")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = SegmentationModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    print(f"Training Loss: {running_loss/len(train_loader):.4f}")
    return model

def eval(model, test_dataset, batch_size):
    print("Start evaluating model")
    print("=========================================")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')  

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test F1 Score (Weighted): {f1:.4f}")