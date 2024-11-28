import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_synthetic_data():
    # Load the synthetic data
    genetics = np.loadtxt('synthetic_genetics.csv', delimiter=',')
    environment = np.loadtxt('synthetic_environment.csv', delimiter=',')
    yields = np.loadtxt('synthetic_yields.csv', delimiter=',')
    
    # Split data into training and testing sets
    X_g_train, X_g_test, X_e_train, X_e_test, y_train, y_test = train_test_split(
        genetics, environment, yields, test_size=0.2, random_state=42)
    
    # Reshape yields to match expected dimensions
    y_train = y_train.reshape(-1, 1) # transforms a 1D array into a 2D array
    y_test = y_test.reshape(-1, 1) # transforms a 1D array into a 2D array
    
    return X_g_train, X_g_test, X_e_train, X_e_test, y_train, y_test

"""
CropDataset::
Purpose: The CropDataset class is designed to hold and manage data related to crop genetics, 
environmental factors, and yield outcomes.
It provides a structured way to access this data in a format suitable for training machine learning models.

Integration with PyTorch: By subclassing Dataset, this class can be easily integrated with PyTorch's data loaders,
enabling efficient loading of data in batches, shuffling, and parallel processing.

Flexibility: The design allows for flexibility in handling different types of data 
(genetics, environment, yields) while ensuring that they are stored in a format that is compatible with 
PyTorch's tensor operations.
"""
class CropDataset(Dataset):
    def __init__(self, genetics, environment, yields):
        self.genetics = torch.FloatTensor(genetics)
        self.environment = torch.FloatTensor(environment)
        self.yields = torch.FloatTensor(yields)
        
    def __len__(self):
        return len(self.yields)
    
    def __getitem__(self, idx):
        return self.genetics[idx], self.environment[idx], self.yields[idx]

def create_dataloaders(batch_size=32):
    # Load and split the data
    X_g_train, X_g_test, X_e_train, X_e_test, y_train, y_test = load_synthetic_data()
    
    # Create datasets
    train_dataset = CropDataset(X_g_train, X_e_train, y_train)
    test_dataset = CropDataset(X_g_test, X_e_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Test the data loading
    X_g_train, X_g_test, X_e_train, X_e_test, y_train, y_test = load_synthetic_data()
    train_loader, test_loader = create_dataloaders()
    
    print(f"Data shapes:")
    print(f"Training:")
    print(f"  Genetics: {X_g_train.shape}")
    print(f"  Environment: {X_e_train.shape}")
    print(f"  Yields: {y_train.shape}")
    print(f"\nTesting:")
    print(f"  Genetics: {X_g_test.shape}")
    print(f"  Environment: {X_e_test.shape}")
    print(f"  Yields: {y_test.shape}")
    
    # Test the data loader
    for genetics, environment, yields in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Genetics: {genetics.shape}")
        print(f"  Environment: {environment.shape}")
        print(f"  Yields: {yields.shape}")
        break  # Just print the first batch
