"""
Feature Extraction Script for VNF Profiling

This script processes packet-level input and output CSV files captured from different
VNFs (Virtual Network Functions) such as UPF, RAN, OvS, or complete network slices.

It extracts features like:
- Interarrival time
- Delay and drop indication
- Exponential moving average of workload
- Metadata (CPU, bandwidth, rate, traffic type)

Output is structured and saved into separate CSV files for training deep learning models.

Author: Hadj Ahmed Chikh Dahmane
"""

import pandas as pd
import numpy as np
import os
import argparse

def calculate_ema(byte_counts, alpha=0.95):
    """
    Calculate the exponential moving average (EMA) of byte counts.

    Parameters:
        byte_counts (np.ndarray): Array of UDP packet lengths (byte counts).
        alpha (float): Smoothing factor for EMA.

    Returns:
        np.ndarray: Smoothed EMA series.
    """
    ema = np.zeros_like(byte_counts, dtype=float)
    ema[0] = byte_counts[0]
    for i in range(1, len(byte_counts)):
        ema[i] = alpha * byte_counts[i] + (1 - alpha) * ema[i - 1]
    return ema


def process_csv_files(input_file, output_file, traffic_type, rate, vnf_type, cpu_upf=None, cpu_ran=None, bw=None):
    """
    Process a pair of input/output packet CSVs to compute delay, drop, and statistical features.

    Parameters:
        input_file (str): Path to the input packet CSV.
        output_file (str): Path to the output packet CSV.
        traffic_type (str): Type of traffic (e.g., 'poisson', 'on_off').
        rate (int): Transmission rate.
        vnf_type (str): VNF type (e.g., 'upf', 'ran', 'ovs', 'slice').
        cpu_upf (int): CPU config for UPF.
        cpu_ran (int): CPU config for RAN.
        bw (int): Bandwidth for OvS.

    Returns:
        pd.DataFrame: Feature-enriched packet-level dataset.
    """
    print(f"Processing input file: {input_file}")
    print(f"Processing output file: {output_file}")
    
    # Read CSVs with defined headers
    cols = ['timestamp', 'src', 'dst', 'id', 'length_ip', 'length_frame',
            'protocols', 'srcport', 'dstport', 'length_udp', 'payload']
    input_df = pd.read_csv(input_file, header=None, names=cols)
    output_df = pd.read_csv(output_file, header=None, names=cols)

    # Convert timestamps and compute interarrival time
    input_df['timestamp'] = input_df['timestamp'].astype(float)
    input_df['interarrival_time'] = input_df['timestamp'].diff().fillna(0)

    # Clean UDP length field (remove anomalies from merging)
    input_df['length_udp'] = input_df['length_udp'].astype(str).apply(lambda x: x.split(',')[-1]).astype(float)

    # Filter bad rows
    input_df = input_df.dropna(subset=['length_udp'])
    input_df = input_df[input_df['length_udp'] <= 1500]

    # Compute workload using EMA
    byte_counts = input_df['length_udp'].values
    input_df['workload'] = calculate_ema(byte_counts)

    # Normalize payloads to enable matching
    input_df['payload'] = input_df['payload'].str.split(',').str[-1]
    output_df['payload'] = output_df['payload'].str.split(',').str[-1]

    # Merge input with output based on payload to compute delay
    input_df = pd.merge(input_df, output_df, how='left', on='payload', suffixes=("", "_output"))
    input_df['delay'] = (input_df['timestamp'] - input_df['timestamp_output']).abs()
    input_df['drop'] = input_df['delay'].isna().astype(int)

    # Drop duplicate payloads and unnecessary columns
    input_df = input_df.drop_duplicates(subset=['payload'])
    input_df = input_df.drop(columns=[col for col in input_df.columns if col.endswith('_output')])

    # Add metadata
    input_df['traffic_type'] = traffic_type
    input_df['rate'] = rate
    if vnf_type == "upf":
        input_df['cpu_upf'] = cpu_upf
    elif vnf_type == "ran":
        input_df['cpu_ran'] = cpu_ran
    elif vnf_type == "ovs":
        input_df['bw'] = bw
    elif vnf_type == "slice":
        input_df['cpu_upf'] = cpu_upf
        input_df['cpu_ran'] = cpu_ran
        input_df['bw'] = bw

    # Drop raw packet fields
    drop_cols = ["protocols", "id", 'length_frame', 'src', 'dst', 'length_ip', 'srcport', 'dstport']
    input_df = input_df.drop(columns=drop_cols)
    input_df = input_df.dropna(subset=[col for col in input_df.columns if col != 'delay'])

    print(input_df.isna().sum())  # Debug: count remaining NaNs
    return input_df


def parse_config_from_filename(filename, vnf_type):
    """
    Parse traffic config (type, CPU, bandwidth, rate) from structured filename.

    Parameters:
        filename (str): Filename of input/output CSV.
        vnf_type (str): VNF type used to adjust parsing.

    Returns:
        Tuple[str, int, int, int, int]: traffic_type, cpu_upf, cpu_ran, bw, rate
    """
    parts = filename.split('_')

    if vnf_type == "slice" and parts[1] != 'on':
        traffic_type = parts[1]
        cpu_upf = int(parts[2])
        cpu_ran = int(parts[3])
        bw = int(parts[4])
        rate = int(parts[5].split('.')[0])
    elif vnf_type == 'slice':
        traffic_type = 'on_off'
        cpu_upf = int(parts[3])
        cpu_ran = int(parts[4])
        bw = int(parts[5])
        rate = int(parts[6].split('.')[0])
    elif parts[2] != 'on':
        traffic_type = parts[2]
        cpu_upf = int(parts[3])
        cpu_ran = int(parts[4])
        bw = int(parts[5])
        rate = int(parts[6].split('.')[0])
    else:
        traffic_type = 'on_off'
        cpu_upf = int(parts[4])
        cpu_ran = int(parts[5])
        bw = int(parts[6])
        rate = int(parts[7].split('.')[0])

    return traffic_type, cpu_upf, cpu_ran, bw, rate


def create_dataset(pcap_folder):
    """
    Traverse the dataset folder structure and process all VNF logs into a clean dataset.

    Parameters:
        pcap_folder (str): Path to top-level folder containing UPF/RAN/OVS/slice logs.
    """
    output_dir = "model_dataset"
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all VNF subfolders
    for vnf_folder in os.listdir(pcap_folder):
        folder_path = os.path.join(pcap_folder, vnf_folder)
        if not os.path.isdir(folder_path):
            continue

        # Identify VNF type
        if "upf" in vnf_folder.lower():
            vnf_type = "upf"
        elif "ran" in vnf_folder.lower():
            vnf_type = "ran"
        elif "ovs" in vnf_folder.lower():
            vnf_type = "ovs"
        elif "slice" in vnf_folder.lower():
            vnf_type = "slice"
        else:
            print(f"Skipping unknown folder: {vnf_folder}")
            continue

        output_subdir = os.path.join(output_dir, f"{vnf_type}_dataset")
        os.makedirs(output_subdir, exist_ok=True)

        # Walk all subdirectories looking for packet files
        for root, _, files in os.walk(folder_path):
            csv_files = [f for f in files if f.endswith('.csv')]
            input_files = [f for f in csv_files if 'input' in f]
            output_files = [f for f in csv_files if 'output' in f]

            if not input_files or not output_files:
                print(f"Skipping directory {root}: Missing input or output files.")
                continue

            # Match input-output pairs by configuration key
            config_map = {}

            for input_file in input_files:
                key = parse_config_from_filename(input_file, vnf_type)
                config_map.setdefault(key, {})["input"] = input_file

            for output_file in output_files:
                key = parse_config_from_filename(output_file, vnf_type)
                config_map.setdefault(key, {})["output"] = output_file

            # Process and save each configuration
            for config_key, file_pair in config_map.items():
                traffic_type, cpu_upf, cpu_ran, bw, rate = config_key
                input_file = file_pair.get("input")
                output_file = file_pair.get("output")

                if not input_file or not output_file:
                    print(f"Skipping configuration {config_key}: Incomplete file pair.")
                    continue

                input_path = os.path.join(root, input_file)
                output_path = os.path.join(root, output_file)

                processed_df = process_csv_files(
                    input_path, output_path, traffic_type, rate,
                    vnf_type, cpu_upf, cpu_ran, bw
                )

                filename = f"data_{traffic_type}_cpu_upf_{cpu_upf}_cpu_ran_{cpu_ran}_bw_{bw}_rate_{rate}.csv"
                processed_df.to_csv(os.path.join(output_subdir, filename), index=False)
                print(f"Saved processed data to {os.path.join(output_subdir, filename)}")


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('pcap_path', help='Path to dataset directory')
    return parser.parse_args()


def main():
    args = get_args()
    create_dataset(args.pcap_path)


if __name__ == '__main__':
    main()
