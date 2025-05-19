import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import csv
import random
from deepptm import deepPTM
import scipy.stats as measures
from scipy.stats import wasserstein_distance
from traffic_traces import traffic_traces
import math
import torch
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import time

#global variables
pytorch_seed = 0
torch.manual_seed(pytorch_seed)
traffic_type = 'poisson'
res = '1500'
vnf_type = 'upf'
rate = '20'
parent_folder = 'w1'


#loss function, inspired by Scipy's wasserstein distance's implementation. 
def cdf_distance(p, u_values, v_values, u_weights=None, v_weights=None):
    """
    Compute the CDF-based statistical distance between two one-dimensional distributions.

    Args:
        p (float): Parameter for the distance. p=1 gives Wasserstein distance.
        u_values (torch.Tensor): Observed values in the first distribution.
        v_values (torch.Tensor): Observed values in the second distribution.
        u_weights (torch.Tensor, optional): Weights for the first distribution. Defaults to None.
        v_weights (torch.Tensor, optional): Weights for the second distribution. Defaults to None.

    Returns:
        torch.Tensor: The computed distance between the distributions.
    """
    u_values = u_values.view(-1)
    v_values = v_values.view(-1)
    # Sort the values in the distributions
    u_values, u_indices = torch.sort(u_values)
    v_values, v_indices = torch.sort(v_values)
   
    # Combine and sort all unique values
    all_values = torch.cat((u_values, v_values))
    all_values, _ = torch.sort(all_values)

    # Calculate deltas (differences between consecutive values)
    deltas = all_values[1:] - all_values[:-1]

    # Compute cumulative weights (CDF) for u
    if u_weights is None:
        u_cdf = torch.searchsorted(u_values, all_values[:-1], right=True).float() / len(u_values)
    else:
        u_weights = u_weights[u_indices]
        u_cumsum = torch.cat((torch.tensor([0.0], device=u_weights.device), torch.cumsum(u_weights, dim=0)))
        u_cdf = u_cumsum[torch.searchsorted(u_values, all_values[:-1], right=True)] / u_cumsum[-1]

    # Compute cumulative weights (CDF) for v
    if v_weights is None:
        v_cdf = torch.searchsorted(v_values, all_values[:-1], right=True).float() / len(v_values)
    else:
        v_weights = v_weights[v_indices]
        v_cumsum = torch.cat((torch.tensor([0.0], device=v_weights.device), torch.cumsum(v_weights, dim=0)))
        v_cdf = v_cumsum[torch.searchsorted(v_values, all_values[:-1], right=True)] / v_cumsum[-1]

    # Compute the integral of |U-V|^p weighted by deltas
    diff = torch.abs(u_cdf - v_cdf)
    if p == 1:
        return torch.sum(diff * deltas)
    elif p == 2:
        return torch.sqrt(torch.sum((diff ** 2) * deltas))
    else:
        return torch.sum((diff ** p) * deltas) ** (1 / p)

def train_val(model, train_dl, test_dl, epochs, device, vnf, stats):
    training_loss = []
    testing_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        model.train()
        train_epoch_loss = 0

        # Training Loop
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # Forward pass
            y_pred = model(x_batch)
            loss = cdf_distance(1, y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        avg_train_loss = train_epoch_loss / len(train_dl)
        training_loss.append(avg_train_loss)
        
        # Testing Loop
        model.eval()
        test_epoch_loss = 0
        with torch.inference_mode():
            for x_batch, y_batch in test_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = cdf_distance(1, y_pred, y_batch)
                b1 = torch.zeros_like(y_batch)
                test_epoch_loss += loss.item()
            
        avg_test_loss = test_epoch_loss / len(test_dl)
        testing_loss.append(avg_test_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Testing Loss: {avg_test_loss}")
    # Convert to NumPy
    training_loss = np.array(training_loss)
    testing_loss = np.array(testing_loss)

    # Plot Training and Testing Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(training_loss)+1), training_loss, label='Training Loss', linewidth = 2)
    plt.plot(range(1, len(testing_loss)+1), testing_loss, label='Testing Loss', linewidth = 2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.title(f'Training and Testing Loss Over Epochs')
    plt.savefig(f'./{parent_folder}/{vnf}/loss_plot.png')
    
        
    def plot_pdf_cdf(actuals, predictions, dataset_name, vnf):
        actuals, predictions = np.array(actuals).flatten(), np.array(predictions).flatten()
        bins = np.histogram(np.hstack((actuals, predictions)), bins=150)[1]
        
        # Save actuals vs. predictions to CSV
        with open(f'./{parent_folder}/{vnf}/actuals_vs_predictions_{dataset_name.lower()}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Actual", "Predicted"])
            writer.writerows(zip(actuals, predictions))
        
        # PDF Plot
        plt.figure(figsize=(10, 10))
        plt.hist(actuals, bins, density=True, histtype='step', label='Actual')
        plt.hist(predictions, bins, density=True, histtype='step', label='Predicted')
        plt.xlabel('Delay (ms)')
        plt.ylabel('Density')
        plt.title(f'PDF - {dataset_name} Set')
        plt.legend()
        plt.savefig(f'./{parent_folder}/{vnf}/probability_predicted_vs_actual_{dataset_name.lower()}.png')
        
        # CDF Plot
        plt.figure(figsize=(10,10))
        sorted_actuals = np.sort(actuals)
        sorted_predictions = np.sort(predictions)
        cdf_actuals = np.linspace(0, 1, len(sorted_actuals))
        cdf_predictions = np.linspace(0, 1, len(sorted_predictions))
        plt.step(sorted_actuals, cdf_actuals, label='Actual')
        plt.step(sorted_predictions, cdf_predictions, label='Predicted')
        plt.xlabel('Delay (ms)')
        plt.ylabel('Cumulative Probability')
        plt.title(f'CDF - {dataset_name} Set')
        plt.legend()
        plt.savefig(f'./{parent_folder}/{vnf}/cdf_predicted_vs_actual_{dataset_name.lower()}.png')
        
        # Wasserstein Distance Calculation
        b1 = np.zeros_like(actuals)
        w1_distance = wasserstein_distance(actuals, predictions) / wasserstein_distance(b1, actuals)
        print(f'W1/ground truth ({dataset_name} set): {w1_distance}')
        
        # Save W1 distance to a text file
        with open(f'./{parent_folder}/{vnf}/w1_distance_{dataset_name.lower()}.txt', 'w') as f:
            f.write(f'W1/ground truth ({dataset_name} set): {w1_distance}\n')
    
    # Train Set Predictions and Actuals
    predictions, actuals = [], []
    with torch.inference_mode():
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            predictions.extend(y_pred.squeeze(-1).cpu().numpy())
            actuals.extend(y_batch.squeeze(-1).cpu().numpy())
    plot_pdf_cdf(actuals, predictions, 'Train', vnf)
    
    # Test Set Predictions and Actuals
    predictions, actuals = [], []
    with torch.inference_mode():
        for x_batch, y_batch in test_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            predictions.extend(y_pred.squeeze(-1).cpu().numpy())
            actuals.extend(y_batch.squeeze(-1).cpu().numpy())
    plot_pdf_cdf(actuals, predictions, 'Test', vnf)
    torch.save(model.state_dict(), f"./{parent_folder}/{vnf}/model.pth")



def get_global_mean_std(paths, drop=False):
    """
    Reads multiple CSV files and returns the global mean and std 
    for each feature across all files.

    Args:
        paths (list of str): List of file paths to CSV files.

    Returns:
        dict: A dictionary with mean and std values for each feature.
              Format: {feature: {"mean": x, "std": y}}
    """
    all_data = []
    for path in paths:
        data = pd.read_csv(path)
        if "rate" in data.columns:
            data = data.drop(['rate', 'traffic_type'], axis=1)
        if not drop:
            data = data.dropna()
        if 'delay' in data.columns:
            threshold = data['delay'].quantile(1)
            data = data[data['delay'] <= threshold]
            # data['delay'] = data['delay']*1e6
        all_data.append(data)

    all_data = pd.concat(all_data, ignore_index=True)

    # Compute global mean and std
    stats = all_data.agg(["mean", "std"]).to_dict()

    return stats

def get_global_min_max(paths, drop = False):
    """
    Reads multiple CSV files and returns the global minimum and maximum 
    for each feature across all files.

    Args:
        file_paths (list of str): List of file paths to CSV files.

    Returns:
        dict: A dictionary with min and max values for each feature.
    """
    # Read all files and concatenate them into a single DataFrame
    all_data = []
    for path in paths:
        data = pd.read_csv(path)
        if "rate" in data.columns:
            data = data.drop(['rate', 'traffic_type'], axis = 1)
        if not drop:
            data = data.dropna()
        if 'delay' in data.columns:
            threshold = data['delay'].quantile(1)
            data = data[data['delay'] <= threshold]
        all_data.append(data)
        
    all_data = pd.concat(all_data, ignore_index=True)

    # Compute global min and max for each feature
    stats = all_data.agg(["min", "max"]).to_dict()

    return stats

def get_train_test_indices(paths, test_ratio=0.1, time_steps=42, drop=False):
    train_indices = []
    test_indices = []
    random.seed(42)
    cumulative_indices = 0

    for path in paths:
        df = pd.read_csv(path)
        if not drop:
            df = df.dropna()
        if 'delay' in df.columns:
            threshold = df['delay'].quantile(1)
            df = df[df['delay'] <= threshold]
        valid_indices = list(range(0,df.shape[0] - time_steps + 1, time_steps))
        test_sample_num = int(len(valid_indices) * test_ratio)
        test_samples = random.sample(valid_indices, test_sample_num)
        train_samples = list(set(valid_indices) - set(test_samples))
        train_samples = random.sample(train_samples, len(train_samples))
        train_indices.extend([idx + cumulative_indices for idx in train_samples])
        test_indices.extend([idx + cumulative_indices for idx in test_samples])
        cumulative_indices += df.shape[0]

    return train_indices, test_indices



#Gets the paths to the CSV files, 
traffic_type = 'poisson'
path_vnf = 'ran'
traffic_types = ["poisson"]
cpu_values_upf = [1500, 1400]
cpu_values_ran = [1500, 1400]
bws = [60, 55]
rates = [20, 15, 10]

paths = []

for traffic in traffic_types:
    for upf in cpu_values_upf:
        for ran in cpu_values_ran:
            for bw in bws:
                for rate in rates: #MAKE SURE YOU CHANGE THE PATH HERE
                    path = f"./model_dataset/{path_vnf}_dataset/data_{traffic}_cpu_upf_{upf}_cpu_ran_{ran}_bw_{bw}_rate_{rate}.csv"
                    paths.append(path)



#Implementation for the LSTM Model 
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply dropout
        out = self.dropout(out)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Define paths and parameters
    traffic_type = 'poisson'
    res = '1500'
    vnf_type = 'RAN'
    rate = '20'
    split_ratio = 0.2  # 20% of the data will be used for testing
    random_seed = 42   # For reproducibility
    stats = get_global_min_max(paths)
    # Save the output to a JSON file 
    with open(f'{parent_folder}/{vnf_type}/global_min_max_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    train_indices, test_indices = get_train_test_indices(paths, time_steps=16)

    print("Loading training dataset...")
    train_data = traffic_traces(
        paths,
        timesteps = 16,
        indices = train_indices, 
        stats = stats

    )
    print(f"Training dataset size: {len(train_data)}")

    print("Loading testing dataset...")
    test_data = traffic_traces(
        paths,
        timesteps = 16,
        indices = test_indices,
        stats = stats
    )
    print(f"Testing dataset size: {len(test_data)}")


    # Create DataLoaders
    train_dl = DataLoader(train_data, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=2048, shuffle=False)

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Define the Model and Start Training
    model_instance = deepPTM(in_feat=4, time_steps=16, device=device).to(device)
    # model_instance = LSTMModel(input_size=4, hidden_size=128, num_layers=1, output_size= 1, dropout=0.0).to(device)
    train_val(model_instance, train_dl, test_dl, epochs=1000, device=device, vnf='RAN',stats=stats)