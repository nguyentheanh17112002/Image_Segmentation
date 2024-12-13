import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

class SegmentationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def create_pytorch_custom_dataset(train_file_path, test_file_path):
    train_data = pd.read_csv('data/segmentation.data', comment=';', skip_blank_lines=True)
    test_data = pd.read_csv('data/segmentation.test', comment=';', skip_blank_lines=True)

    train_features = train_data.columns[:]
    test_features = test_data.columns[:]

    X_train = train_data[train_features].values
    X_test = test_data[test_features].values 

    y_train = train_data.index.values
    y_test = test_data.index.values

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Encode labels (convert class names to integers)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    train_dataset = SegmentationDataset(X_train, y_train)
    test_dataset = SegmentationDataset(X_test, y_test)  

    return train_dataset, test_dataset
    