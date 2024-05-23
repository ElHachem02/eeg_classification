import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils import load_data_and_labels
import pandas as pd

# Assuming `dataset` is a class that extends torch.utils.data.Dataset
class dataset(Dataset):
    def __init__(self, X, y, transform=None, fs=256, window_sec=10, device="mps"):
        self.window_size = fs * window_sec  # Total number of samples in 20 seconds
        self.X = [self._extract_window(df) for df in X]  # Extract a 20-second window
        self.y = torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.long)  # Convert y to numpy array if it's a pandas Series
        self.transform = transform
        
        self.device = device  # Device to move tensors to

        # Move tensors to the specified device
        self.X = torch.stack(self.X).to(self.device)
        self.y = self.y.to(self.device)
        
    def _extract_window(self, df):
        # Ensure that we have at least `self.window_size` samples
        if len(df) < self.window_size:
            raise ValueError("Data point length is less than the required window size")
        return torch.tensor(df.iloc[:self.window_size, 1:].values, dtype=torch.float32)
    
    def __len__(self):
        length = len(self.X)
        print(f"Dataset length: {length}")
        return length

    def __getitem__(self, idx):
        sample = {'X': self.X[idx], 'y': self.y[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

def _generate_data(processed_subjects):
    data_dict = {}
    X_all = []
    y_all = []
    
    for subject, (X_list, y_list) in processed_subjects.items():
        # Concatenate the list of DataFrames into a single DataFrame
        X = X_list
        y = y_list
        
        # Split the data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Store the splits in the dictionary
        data_dict[subject] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
        
        # Append to the combined data
        X_all.extend(X)
        y_all.extend(y)
    
    # Create DataFrame with all data points and labels
    X_combined = X_all
    y_combined = y_all
    
    # Split the combined data
    X_train_combined, X_temp_combined, y_train_combined, y_temp_combined = train_test_split(X_combined, y_combined, test_size=0.4, random_state=42)
    X_val_combined, X_test_combined, y_val_combined, y_test_combined = train_test_split(X_temp_combined, y_temp_combined, test_size=0.5, random_state=42)
    
    combined_data = {
        'X_train': X_train_combined,
        'y_train': y_train_combined,
        'X_val': X_val_combined,
        'y_val': y_val_combined,
        'X_test': X_test_combined,
        'y_test': y_test_combined
    }
        
    return data_dict, combined_data


def _get_loaders(train_X, train_y, val_X, val_y, test_X, test_y):
    train_set, val_set, test_set = dataset(train_X, train_y), dataset(val_X, val_y), dataset(test_X, test_y)
    data_loader_train = DataLoader(
        train_set, 
        batch_size=1, 
        num_workers=0,
        pin_memory=True, 
        drop_last=False,
    )
    data_loader_val = DataLoader(
            val_set, 
            batch_size=1, 
            num_workers=0,
            pin_memory=True, 
            drop_last=False,
    )
    data_loader_test = DataLoader(
            test_set, 
            batch_size=1,
            num_workers=0,
            pin_memory=True, 
            drop_last=False,
    )
    dataloaders = {
        'train': data_loader_train,
        'val': data_loader_val,
        'test': data_loader_test
    }
    return dataloaders


def generate_data_with_loaders():
    processed_subjects = load_data_and_labels()
    data_dict, combined_data = _generate_data(processed_subjects)
    
    dataloaders_dict = {}
    for subject, splits in data_dict.items():
        dataloaders = _get_loaders(splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'], splits['X_test'], splits['y_test'])
        dataloaders_dict[subject] = dataloaders
    
    combined_dataloaders = _get_loaders(
        combined_data['X_train'], combined_data['y_train'],
        combined_data['X_val'], combined_data['y_val'],
        combined_data['X_test'], combined_data['y_test']
    )
    
    return data_dict, combined_data, dataloaders_dict, combined_dataloaders
