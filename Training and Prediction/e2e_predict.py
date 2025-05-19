# Import necessary libraries
import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from deepptm import deepPTM  # Custom deep learning model
from tqdm import tqdm  # Progress bar
from torch import nn
from scipy.stats import wasserstein_distance  # Metric to compare distributions
import matplotlib.pyplot as plt 
import csv

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global constants
TIMESTEPS = 16  # Number of timesteps for sequence input
BATCH_SIZE = 2048  # Batch size for DataLoader
ALPHA = 0.95  # Smoothing factor for EMA

# Configuration for each VNF model, including paths to trained weights and selected features
model_config = {
    'upf': {
        'model_path': './w1/UPF/model.pth',
        'stats_path': './w1/UPF/global_min_max_stats.json',
        'features': ['interarrival_time', 'length_udp', 'workload', 'cpu_upf']
    },
    'ran': {
        'model_path': './w1/RAN/model.pth',
        'stats_path': './w1//RAN/global_min_max_stats.json',
        'features': ['interarrival_time', 'length_udp', 'workload', 'cpu_ran']
    },
    'ovs': {
        'model_path': './w1/OvS/model.pth',
        'stats_path': './w1/OvS/global_min_max_stats.json',
        'features': ['interarrival_time', 'length_udp', 'workload', 'bw']
    }
}

# Optional LSTM-based model definition (commented out by default)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Load trained models and their corresponding normalization statistics
models = {}
stats = {}
for vnf, config in model_config.items():
    model = deepPTM(in_feat=len(config['features']), time_steps=TIMESTEPS, device=device).to(device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device, weights_only=True))
    model.eval()
    models[vnf] = model

    with open(config['stats_path']) as f:
        stats[vnf] = json.load(f)

# Calculate Exponential Moving Average (EMA)
def calculate_ema(values, alpha=ALPHA):
    ema = np.zeros_like(values, dtype=float)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema

# Undo normalization for delay values
def denormalize_delay(predicted_delays, vnf_type):
    delay_stats = stats[vnf_type]['delay']
    max_val = delay_stats['max']
    min_val = delay_stats['min']
    if isinstance(predicted_delays, torch.Tensor):
        predicted_delays = predicted_delays.detach().cpu().numpy()
    return (predicted_delays * (max_val - min_val)) + min_val

# Apply min-max normalization to features
def min_max_scale(data, stats):
    for feature, min_max in stats.items():
        if feature in data.columns:
            min_val, max_val = min_max['min'], min_max['max']
            if max_val > min_val:
                data[feature] = (data[feature] - min_val) / (max_val - min_val)
    return data

# Load, clean, and preprocess multiple datasets
def load_and_preprocess_data(paths):
    all_data = []
    for path in paths:
        df = pd.read_csv(path).dropna()
        if 'delay' in df.columns:
            threshold = df['delay'].quantile(1)
            df = df[df['delay'] <= threshold]
        all_data.append(df)
    full_df = pd.concat(all_data).sort_values('timestamp').reset_index(drop=True)
    full_df['interarrival_time'] = full_df['timestamp'].diff().fillna(0)
    full_df['workload'] = calculate_ema(full_df['length_udp'].values)
    return full_df

# Dataset class for sequence modeling
class SequenceDataset(Dataset):
    def __init__(self, df, timesteps):
        self.df = df
        self.timesteps = timesteps
        self.feature_cols = ['interarrival_time', 'length_udp', 'workload', 'cpu_upf', 'cpu_ran', 'bw']
        self.indices = range(0, len(self.df)-self.timesteps-1, self.timesteps)
    def __len__(self):
        return len(self.df) - self.timesteps + 1
    def __getitem__(self, idx):
        seq = self.df.iloc[idx:idx+self.timesteps]
        features = seq[self.feature_cols].values
        actual_delay = seq['delay'].values[-1]
        payload = seq['payload'].to_list()[-1]
        return features, actual_delay, payload, idx

# Prepare input data for a specific VNF model
def prepare_vnf_input(batch_features, vnf_type):
    feature_cols = model_config[vnf_type]['features']
    all_features = ['interarrival_time', 'length_udp', 'workload', 'cpu_upf', 'cpu_ran', 'bw']
    col_indices = [all_features.index(f) for f in feature_cols]
    features = batch_features[:, :, col_indices]
    features_flat = features.reshape(-1, len(feature_cols))
    df = pd.DataFrame(features_flat, columns=feature_cols)
    df_scaled = min_max_scale(df, stats[vnf_type])
    scaled_features = df_scaled.values.reshape(features.shape)
    return torch.FloatTensor(scaled_features).to(device)

# Run inference for a VNF stage
def process_vnf_stage(df, vnf_type):
    dataset = SequenceDataset(df, TIMESTEPS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=2)
    n_sequences = len(dataset)
    all_delays = np.zeros((n_sequences, TIMESTEPS))  
    all_actual = np.zeros(n_sequences)
    all_payloads = ["" for _ in range(n_sequences)]
    for batch_features, batch_actual, batch_payloads, batch_indices in tqdm(dataloader, desc=f"Processing {vnf_type}"):
        inputs = prepare_vnf_input(batch_features.numpy(), vnf_type)
        with torch.no_grad():
            with torch.amp.autocast('cuda:0'):
                delays_scaled = models[vnf_type](inputs)
        delays = denormalize_delay(delays_scaled.squeeze(-1), vnf_type)
        batch_indices = batch_indices.numpy()
        all_delays[batch_indices] = delays.cpu().numpy() if isinstance(delays, torch.Tensor) else delays
        all_actual[batch_indices] = batch_actual.numpy()
        for i, idx in enumerate(batch_indices):
            all_payloads[idx] = batch_payloads[i]
    return all_delays, all_actual, all_payloads

# Update timestamps based on predicted delays
def update_timestamps(df, delays):
    updated_df = df.copy()
    for seq_idx in range(delays.shape[0]):
        start_idx = seq_idx
        end_idx = seq_idx + TIMESTEPS
        if end_idx <= len(updated_df):
            updated_df.iloc[start_idx:end_idx, updated_df.columns.get_loc('timestamp')] += delays[seq_idx]
    updated_df = updated_df.sort_values('timestamp').reset_index(drop=True)
    updated_df['interarrival_time'] = updated_df['timestamp'].diff().fillna(0)
    updated_df['workload'] = calculate_ema(updated_df['length_udp'].values)
    return updated_df

# Main inference pipeline across all VNFs
def run_e2e_prediction():
    slice_paths = [
        f"./model_dataset/slice_dataset/data_on_off_cpu_upf_{upf}_cpu_ran_{ran}_bw_{bw}_rate_{rate}.csv"
        for upf in [1500, 1400] 
        for ran in [1500, 1400] 
        for bw in [60, 55]
        for rate in [20, 15, 10]
    ]
    full_df = load_and_preprocess_data(slice_paths)
    upf_delays, actual_delays, payloads = process_vnf_stage(full_df, 'upf')
    full_df = update_timestamps(full_df, upf_delays)
    ovs_delays, _, _ = process_vnf_stage(full_df, 'ovs')
    full_df = update_timestamps(full_df, ovs_delays)
    ran_delays, _, _ = process_vnf_stage(full_df, 'ran')
    results = []
    for seq_idx in range(len(payloads)):
        seq_df = pd.DataFrame({
            'actual_delay': actual_delays[seq_idx],
            'timestep': range(TIMESTEPS),
            'total_delay': upf_delays[seq_idx] + ran_delays[seq_idx] + ovs_delays[seq_idx]
        })
        results.append(seq_df)
    final_results = pd.concat(results, ignore_index=True)
    return final_results

# Plot and save PDF and CDF of actual vs predicted delays
def plot_pdf_cdf(actuals, predictions):
    actuals, predictions = np.array(actuals).flatten(), np.array(predictions).flatten()
    bins = np.histogram(np.hstack((actuals, predictions)), bins=150)[1]
    parent_folder = 'w1'
    vnf = 'slice'
    dataset_name = 'e2e'
    with open(f'./{parent_folder}/{vnf}/actuals_vs_predictions_{dataset_name.lower()}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Actual", "Predicted"])
        writer.writerows(zip(actuals, predictions))

    # PDF Plot
    plt.figure(figsize=(10, 10))
    plt.hist(actuals, bins, density=True, histtype='step', label='Actual', linewidth=3)
    plt.hist(predictions, bins, density=True, histtype='step', label='Predicted', linewidth=3)
    plt.xlabel('Delay (ms)')
    plt.ylabel('Density')
    plt.title(f'PDF - {dataset_name} Set')
    plt.legend()
    plt.savefig(f'./{parent_folder}/{vnf}/probability_predicted_vs_actual_{dataset_name.lower()}.png')

    # CDF Plot
    plt.figure(figsize=(10, 10))
    sorted_actuals = np.sort(actuals)
    sorted_predictions = np.sort(predictions)
    cdf_actuals = np.linspace(0, 1, len(sorted_actuals))
    cdf_predictions = np.linspace(0, 1, len(sorted_predictions))
    plt.step(sorted_actuals, cdf_actuals, label='Actual', linewidth=3)
    plt.step(sorted_predictions, cdf_predictions, label='Predicted', linewidth=3)
    plt.xlabel('Delay (ms)')
    plt.ylabel('Cumulative Probability')
    plt.title(f'CDF - {dataset_name} Set')
    plt.legend()
    plt.savefig(f'./{parent_folder}/{vnf}/cdf_predicted_vs_actual_{dataset_name.lower()}.png')

# Entry point for end-to-end inference and evaluation
if __name__ == "__main__":
    results_df = run_e2e_prediction()
    actual = results_df['actual_delay']
    predicted = results_df['total_delay']
    print(f"W1: {wasserstein_distance(actual, predicted):.4f}")
    plot_pdf_cdf(actual, predicted)
