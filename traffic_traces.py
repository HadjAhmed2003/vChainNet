import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import random
from sklearn.preprocessing import StandardScaler
def min_max_scale(data, stats):
    """
    Apply min-max scaling using global min and max values.
    """
    for feature, min_max in stats.items():
        if feature in data.columns:
            min_val, max_val = min_max['min'], min_max['max']
            if max_val > min_val:  # Avoid division by zero
                data[feature] = (data[feature] - min_val) / (max_val - min_val)
    return data

def standard_scale(data, stats):
    """
    Apply standard scaling using global mean and std values.

    Args:
        data (pd.DataFrame): The data to scale.
        stats (dict): The global mean and std from get_global_mean_std().

    Returns:
        pd.DataFrame: The scaled data.
    """
    for feature, mean_std in stats.items():
        if feature in data.columns:
            mean_val = mean_std['mean']
            std_val = mean_std['std']
            if std_val > 0:
                data[feature] = (data[feature] - mean_val) / std_val
    return data

# Calculate rates
def calc_rate(ser: pd.Series, input_df: pd.DataFrame):
    input_df_roll = input_df.loc[ser.index]
    length_sum = input_df_roll['length_udp'].sum()
    time_diff = input_df_roll['timestamp'].iloc[-1] - input_df_roll['timestamp'].iloc[0]
    return length_sum / time_diff

    

class traffic_traces(Dataset):
    def __init__(self, paths, timesteps, indices, stats, drop = False, std_scale = False, augment = False):
        self.timestep = timesteps
        self.x = []
        self.y = []
        self.indices = indices
        self.stats = stats
        for path in paths:
            data = pd.read_csv(path)
            if "rate" in data.columns:
                data = data.drop(['rate', 'traffic_type'], axis = 1)
        
            if not drop:
                data = data.dropna()
            if 'delay' in data.columns:
                threshold = data['delay'].quantile(1)
                data = data[data['delay'] <= threshold]
            
            data = data.drop(['timestamp', 'drop'], axis = 1)
            if 'payload' in data.columns:
                data = data.drop(['payload'], axis = 1)
            if std_scale:
                data = standard_scale(data, stats)
            else:    
                data = min_max_scale(data, stats)
            y = data[['delay']].to_numpy(dtype=np.float64)
            data = data.drop(['delay'], axis = 1)
            x = data.to_numpy(dtype=np.float64)
            self.x.append(x)
            self.y.append(y)
        # Concatenate all data
        self.x = np.concatenate(self.x, axis=0)
        self.y = np.concatenate(self.y, axis=0)

        # Convert data to PyTorch tensors
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.timestep
        x = self.x[start_idx:end_idx]
        y = self.y[start_idx:end_idx]
        return x, y
