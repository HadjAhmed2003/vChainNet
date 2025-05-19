import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import StandardScaler

def calculate_ema(byte_counts, alpha=0.95):
    """
    Calculate the exponential moving average (EMA) of byte counts.

    Parameters:
        byte_counts (array-like): Array of byte counts to smooth.
        alpha (float): Smoothing factor between 0 and 1. Higher values give more weight to recent values.

    Returns:
        np.ndarray: Smoothed array using EMA.
    """
    ema = np.zeros_like(byte_counts, dtype=float)
    ema[0] = byte_counts[0]  # Initialize EMA with the first value
    for i in range(1, len(byte_counts)):
        ema[i] = alpha * byte_counts[i] + (1 - alpha) * ema[i - 1]
    return ema

def process_csv_files(input_file, output_file, traffic_type, rate, cpu=None, bw=None):
    """
    Process input and output CSV files to extract features for a VNF dataset.

    Reads the input and output packet capture CSV files, computes various
    time-series features such as interarrival time, rolling transmission rate,
    workload (EMA), delay, and drop indicators, and attaches metadata.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        traffic_type (str): Type of traffic (e.g., 'poisson', 'on_off').
        rate (int): Traffic generation rate.
        cpu (int, optional): CPU limit (used for RAN/UPF).
        bw (int, optional): Bandwidth (used for OVS).

    Returns:
        pd.DataFrame: Processed DataFrame with extracted features.
    """
    print(f"Processing input file: {input_file}")
    print(f"Processing output file: {output_file}")

    # Load input and output packet data
    input_df = pd.read_csv(input_file, header=None, names=[
        'timestamp', 'src', 'dst', 'id', 'length_ip', 'length_frame',
        'protocols', 'srcport', 'dstport', 'length_udp', 'payload'
    ])
    output_df = pd.read_csv(output_file, header=None, names=[
        'timestamp', 'src', 'dst', 'id', 'length_ip', 'length_frame',
        'protocols', 'srcport', 'dstport', 'length_udp', 'payload'
    ])
    
    # Compute interarrival time
    input_df['timestamp'] = input_df['timestamp'].astype(float)
    input_df['interarrival_time'] = input_df['timestamp'].diff().fillna(0)
    
    # Normalize 'length_udp' field (sometimes stored as comma-separated string)
    input_df['length_udp'] = input_df['length_udp'].astype(str).apply(lambda x: x.split(',')[-1]).astype(float)
    
    # Remove invalid or oversized packets
    input_df = input_df.dropna(subset=['length_udp'])
    input_df = input_df[input_df['length_udp'] <= 1500]
    
    # Define helper function to compute rolling rate
    def calc_rate(ser: pd.Series, input_df: pd.DataFrame):
        input_df_roll = input_df.loc[ser.index]
        length_sum = input_df_roll['length_udp'].sum()
        time_diff = input_df_roll['timestamp'].iloc[-1] - input_df_roll['timestamp'].iloc[0]
        return length_sum / time_diff if time_diff > 0 else 0

    # Compute rolling rate over various windows
    for w in [16, 32, 64, 128, 256, 512]:
        input_df[f'rate_{w}'] = input_df['timestamp'].rolling(window=w).apply(calc_rate, args=(input_df,))

    # Compute workload using EMA of byte counts
    byte_counts = input_df['length_udp'].values
    input_df['workload'] = calculate_ema(byte_counts, alpha=0.95)

    # Normalize payload fields for matching
    input_df['payload'] = input_df['payload'].str.split(',').str[-1]
    output_df['payload'] = output_df['payload'].str.split(',').str[-1]

    # Compute delay and drop metrics by merging on payload
    input_df['delay'] = np.nan
    input_df['drop'] = np.nan
    input_df = pd.merge(input_df, output_df, how='left', on='payload', suffixes=("", "_output"))
    input_df['delay'] = (input_df['timestamp'] - input_df['timestamp_output']).abs()
    input_df['drop'] = input_df['delay'].isna().astype(int)
    
    # Keep only the first match per payload
    input_df = input_df.drop_duplicates(subset=['payload'])

    # Clean up columns (remove _output suffix duplicates)
    input_df = input_df.drop(columns=[col for col in input_df.columns if col.endswith('_output')])

    # Add metadata columns
    if cpu is not None:
        input_df['cpu'] = cpu
    if bw is not None:
        input_df['bw'] = bw

    # Drop unused columns
    ip_columns = ["payload", "protocols", "id", 'length_frame', 'src', 'dst', 'length_ip', 'srcport', 'dstport']
    input_df = input_df.drop(ip_columns, axis=1)

    # Drop rows with NaNs in any relevant column except 'delay'
    input_df = input_df.dropna(subset=[col for col in input_df.columns if col != 'delay'])

    print(input_df.isna().sum())  # Debug: show remaining NaNs
    return input_df


def create_dataset(pcap_folder, vnf_type):
    """
    Recursively traverses a pcap folder structure to generate a feature dataset for a given VNF.

    Parameters:
        pcap_folder (str): Path to the auto-profile directory containing traffic data.
        vnf_type (str): Type of VNF (e.g., 'RAN', 'UPF', 'OVS').

    Output:
        Saves processed train/test CSV files in a structured dataset folder.
    """
    output_dir = f"{vnf_type}_dataset_new"
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(pcap_folder):
        if "rate_" in root:
            # Extract traffic type, dataset type, and VNF parameters from folder names
            path_parts = root.split(os.sep)
            dataset_type = path_parts[-4]
            traffic_type = path_parts[-3]

            if vnf_type.lower() == "ovs":
                rate = int(path_parts[-1].split('_')[1])
                bw = int(path_parts[-2].split('_')[1])
                cpu = None
            else:
                rate = int(path_parts[-1].split('_')[1])
                cpu = int(path_parts[-2].split('_')[1])
                bw = None

            # Detect input/output CSV files
            csv_files = [f for f in os.listdir(root) if f.endswith('.csv')]
            input_files = [f for f in csv_files if 'input' in f]
            output_files = [f for f in csv_files if 'output' in f]

            if not input_files or not output_files:
                print(f"Skipping directory {root}: Missing input or output CSV files.")
                continue

            input_file = os.path.join(root, input_files[0])
            output_file = os.path.join(root, output_files[0])

            # Process and extract features
            processed_data = process_csv_files(input_file, output_file, traffic_type, rate, cpu, bw)

            # Save to appropriate subfolder
            if vnf_type.lower() == "ovs":
                output_subdir = os.path.join(output_dir, traffic_type, f"bw_{bw}", f"rate_{rate}")
            else:
                output_subdir = os.path.join(output_dir, traffic_type, f"cpu_{cpu}", f"rate_{rate}")

            os.makedirs(output_subdir, exist_ok=True)
            output_file_path = os.path.join(output_subdir, f"{dataset_type}.csv")
            processed_data.to_csv(output_file_path, index=False)
            print(f"Saved processed data to {output_file_path}")


def get_args():
    """
    Parses command-line arguments for dataset generation.

    Returns:
        argparse.Namespace: Parsed arguments with 'pcap_path' and 'vnf_type'.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('pcap_path', help='Path to the auto-profile directory')
    parser.add_argument('--vnf_type', required=True, help='Type of VNF (RAN, OvS, or UPF)')
    return parser.parse_args()


def main():
    """
    Main execution point: parses arguments and triggers dataset creation.
    """
    args = get_args()
    create_dataset(args.pcap_path, args.vnf_type)


if __name__ == '__main__':
    main()
