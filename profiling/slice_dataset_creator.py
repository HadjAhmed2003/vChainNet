import pandas as pd
import numpy as np
import os
import argparse

def calculate_ema(byte_counts, alpha=0.95):
    """
    Calculate the exponential moving average (EMA) of byte counts.
    """
    ema = np.zeros_like(byte_counts, dtype=float)
    ema[0] = byte_counts[0]  # Initialize EMA with the first value
    for i in range(1, len(byte_counts)):
        ema[i] = alpha * byte_counts[i] + (1 - alpha) * ema[i - 1]
    return ema

def process_csv_files(input_file, output_file, traffic_type, rate, vnf_type, cpu_upf=None, cpu_ran=None, bw=None):
    """
    Process input and output CSV files to calculate features like delay, drop, and rates.
    """
    print(f"Processing input file: {input_file}")
    print(f"Processing output file: {output_file}")
    
    # Read input and output CSV files
    input_df = pd.read_csv(input_file, header=None, names=['timestamp', 'src', 'dst', 'id', 'length_ip', 'length_frame', 'protocols', 'srcport', 'dstport', 'length_udp', 'payload'])
    output_df = pd.read_csv(output_file, header=None, names=['timestamp', 'src', 'dst', 'id', 'length_ip', 'length_frame', 'protocols', 'srcport', 'dstport', 'length_udp', 'payload'])
    
    # Preprocess input data
    input_df['timestamp'] = input_df['timestamp'].astype(float)
    input_df['interarrival_time'] = input_df['timestamp'].diff().fillna(0)
    input_df['length_udp'] = input_df['length_udp'].astype(str).apply(lambda x: x.split(',')[-1]).astype(float)
    input_df = input_df.dropna(subset=['length_udp'])  # Drop rows where 'length_udp' is NaN
    input_df = input_df[input_df['length_udp'] <= 1500]  # Keep rows where 'length_udp' <= 1500
    
    # Calculate rates
    def calc_rate(ser: pd.Series, input_df: pd.DataFrame):
        input_df_roll = input_df.loc[ser.index]
        length_sum = input_df_roll['length_udp'].sum()
        time_diff = input_df_roll['timestamp'].iloc[-1] - input_df_roll['timestamp'].iloc[0]
        return length_sum / time_diff
    
    #input_df['rate_16'] = input_df['timestamp'].rolling(window=16).apply(calc_rate, args=(input_df,))
    #input_df['rate_32'] = input_df['timestamp'].rolling(window=32).apply(calc_rate, args=(input_df,))
    #input_df['rate_64'] = input_df['timestamp'].rolling(window=64).apply(calc_rate, args=(input_df,))
    #input_df['rate_128'] = input_df['timestamp'].rolling(window=128).apply(calc_rate, args=(input_df,))
    #input_df['rate_256'] = input_df['timestamp'].rolling(window=256).apply(calc_rate, args=(input_df,))
    #input_df['rate_512'] = input_df['timestamp'].rolling(window=512).apply(calc_rate, args=(input_df,))
    
    # Calculate workload (EMA of bytes arriving at the ingress stream)
    byte_counts = input_df['length_udp'].values
    input_df['workload'] = calculate_ema(byte_counts, alpha=0.95)
    input_df['payload'] = input_df['payload'].str.split(',').str[-1]
    output_df['payload'] = output_df['payload'].str.split(',').str[-1]
    
    # Calculate delays (matching input and output packets)
    input_df = pd.merge(input_df, output_df, how='left', on='payload', suffixes=("", "_output"))
    input_df['delay'] = (input_df['timestamp'] - input_df['timestamp_output']).abs()
    input_df['drop'] = input_df['delay'].isna().astype(int)
    input_df = input_df.drop_duplicates(subset=['payload'])
    input_df = input_df.drop(columns=[col for col in input_df.columns if col.endswith('_output')])
    
    # Add metadata columns based on VNF type
    input_df['traffic_type'] = traffic_type
    input_df['rate'] = rate

    if vnf_type == "upf":
        input_df['cpu_upf'] = cpu_upf  # Only include CPU for UPF
    elif vnf_type == "ran":
        input_df['cpu_ran'] = cpu_ran  # Only include CPU for RAN
    elif vnf_type == "ovs":
        input_df['bw'] = bw  # Only include bandwidth for OvS
    elif vnf_type == "slice":
        input_df['cpu_upf'] = cpu_upf  # Include all configurations for slice
        input_df['cpu_ran'] = cpu_ran
        input_df['bw'] = bw

    # Drop unnecessary columns (after all calculations are complete)
    if vnf_type == "slice":
        ip_columns = ["protocols", "id", 'length_frame', 'src', 'dst', 'length_ip', 'srcport', 'dstport']
    else:
        ip_columns = ["protocols", "id", 'length_frame', 'src', 'dst', 'length_ip', 'srcport', 'dstport'] 
    input_df = input_df.drop(ip_columns, axis=1)
    input_df = input_df.dropna(subset=[col for col in input_df.columns if col not in ['delay']])
    
    print(input_df.isna().sum())
    return input_df

def parse_config_from_filename(filename, vnf_type):
    """
    Parse configuration details (traffic type, CPU, bandwidth, rate) from the filename.
    """
    # Example filename: input_poisson_1500_1400_100_10.csv
    parts = filename.split('_')
    if vnf_type == "slice" and parts[1] != 'on':
        traffic_type = parts[1]  # e.g., poisson
        cpu_upf = int(parts[2])  # e.g., 1500
        cpu_ran = int(parts[3])  # e.g., 1400
        bw = int(parts[4])       # e.g., 100
        rate = int(parts[5].split('.')[0])  # e.g., 10
    elif vnf_type == 'slice' and parts[1] == 'on':
        traffic_type = 'on_off'
        cpu_upf = int(parts[3])  # e.g., 1500
        cpu_ran = int(parts[4])  # e.g., 1400
        bw = int(parts[5])       # e.g., 100
        rate = int(parts[6].split('.')[0])  # e.g., 10
    elif vnf_type != 'slice' and parts[2] != 'on':
        traffic_type = parts[2]  # e.g., poisson
       	cpu_upf = int(parts[3])  # e.g., 1500
        cpu_ran = int(parts[4])  # e.g., 1400
        bw = int(parts[5])       # e.g., 100
        rate = int(parts[6].split('.')[0])  # e.g., 10
    elif vnf_type != 'slice' and parts[2] == 'on':
        traffic_type = 'on_off'
        cpu_upf = int(parts[4])  # e.g., 1500
        cpu_ran = int(parts[5])  # e.g., 1400
        bw = int(parts[6])       # e.g., 100
        rate = int(parts[7].split('.')[0])  # e.g., 10
    return traffic_type, cpu_upf, cpu_ran, bw, rate

def create_dataset(pcap_folder):
    """
    Create a dataset from the auto-profile folder structure.
    """
    # Create the output directory structure
    output_dir = "model_dataset"
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all VNF folders in the dataset directory
    for vnf_folder in os.listdir(pcap_folder):
        if not os.path.isdir(os.path.join(pcap_folder, vnf_folder)):
            continue  # Skip non-directory files

        # Determine VNF type from folder name
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

        # Create output subdirectory for this VNF type
        output_subdir = os.path.join(output_dir, f"{vnf_type}_dataset")
        os.makedirs(output_subdir, exist_ok=True)

        # Process all files in the VNF folder
        vnf_folder_path = os.path.join(pcap_folder, vnf_folder)
        for root, dirs, files in os.walk(vnf_folder_path):
            # Find input and output CSV files (ignore non-CSV files)
            csv_files = [f for f in files if f.endswith('.csv')]
            input_files = [f for f in csv_files if 'input' in f]
            output_files = [f for f in csv_files if 'output' in f]

            if not input_files or not output_files:
                print(f"Skipping directory {root}: Missing input or output CSV files.")
                continue

            # Create a dictionary to map configurations to input/output files
            config_map = {}

            # Process input files
            for input_file in input_files:
                traffic_type, cpu_upf, cpu_ran, bw, rate = parse_config_from_filename(input_file, vnf_type)
                config_key = (traffic_type, cpu_upf, cpu_ran, bw, rate)
                if config_key not in config_map:
                    config_map[config_key] = {"input": None, "output": None}
                config_map[config_key]["input"] = input_file

            # Process output files
            for output_file in output_files:
                traffic_type, cpu_upf, cpu_ran, bw, rate = parse_config_from_filename(output_file, vnf_type)
                config_key = (traffic_type, cpu_upf, cpu_ran, bw, rate)
                if config_key not in config_map:
                    config_map[config_key] = {"input": None, "output": None}
                config_map[config_key]["output"] = output_file

            # Process each configuration pair
            for config_key, files in config_map.items():
                traffic_type, cpu_upf, cpu_ran, bw, rate = config_key
                input_file = files["input"]
                output_file = files["output"]

                if input_file is None or output_file is None:
                    print(f"Skipping configuration {config_key}: Missing input or output file.")
                    continue

                # Process the CSV files
                input_path = os.path.join(root, input_file)
                output_path = os.path.join(root, output_file)
                processed_data = process_csv_files(input_path, output_path, traffic_type, rate, vnf_type, cpu_upf, cpu_ran, bw)

                # Save the processed data to the output directory
                output_filename = f"data_{traffic_type}_cpu_upf_{cpu_upf}_cpu_ran_{cpu_ran}_bw_{bw}_rate_{rate}.csv"
                output_file_path = os.path.join(output_subdir, output_filename)
                processed_data.to_csv(output_file_path, index=False)
                print(f"Saved processed data to {output_file_path}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pcap_path', help='Path to the dataset directory')
    return parser.parse_args()

def main():
    args = get_args()
    create_dataset(args.pcap_path)

if __name__ == '__main__':
    main()
