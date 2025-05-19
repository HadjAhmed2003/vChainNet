import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import StandardScaler

def calculate_ema(byte_counts, alpha=0.95):
    """
    Calculate the exponential moving average (EMA) of byte counts.
    """
    ema = np.zeros_like(byte_counts, dtype=float)
    ema[0] = byte_counts[0]  # Initialize EMA with the first value
    for i in range(1, len(byte_counts)):
        ema[i] = alpha * byte_counts[i] + (1 - alpha) * ema[i - 1]
    return ema

def process_csv_files(input_file, output_file, traffic_type, rate, cpu=None, bw=None):
    # Read input and output CSV files
    print(f"Processing input file: {input_file}")
    print(f"Processing output file: {output_file}")
    input_df = pd.read_csv(input_file, header=None, names=['timestamp', 'src', 'dst', 'id', 'length_ip', 'length_frame', 'protocols', 'srcport', 'dstport', 'length_udp', 'payload'])
    output_df = pd.read_csv(output_file, header=None, names=['timestamp', 'src', 'dst', 'id', 'length_ip', 'length_frame', 'protocols', 'srcport', 'dstport', 'length_udp', 'payload'])
    
    # Drop rows with missing values (only if absolutely necessary, e.g., for critical columns)
    # input_df = input_df.dropna(subset=['timestamp', 'length_udp', 'payload'])  # Example: Drop rows with missing timestamps or lengths
    # output_df = output_df.dropna(subset=['timestamp', 'length_udp', 'payload'])

    # Calculate interarrival time and other features before dropping any rows
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
    
    input_df['rate_16'] = input_df['timestamp'].rolling(window=16).apply(calc_rate, args=(input_df,))
    input_df['rate_32'] = input_df['timestamp'].rolling(window=32).apply(calc_rate, args=(input_df,))
    input_df['rate_64'] = input_df['timestamp'].rolling(window=64).apply(calc_rate, args=(input_df,))
    input_df['rate_128'] = input_df['timestamp'].rolling(window=128).apply(calc_rate, args=(input_df,))
    input_df['rate_256'] = input_df['timestamp'].rolling(window=256).apply(calc_rate, args=(input_df,))
    input_df['rate_512'] = input_df['timestamp'].rolling(window=512).apply(calc_rate, args=(input_df,))
    
    # Calculate workload (EMA of bytes arriving at the ingress stream)
    byte_counts = input_df['length_udp'].values
    
    input_df['workload'] = calculate_ema(byte_counts, alpha=0.95)
    input_df['payload'] = input_df['payload'].str.split(',').str[-1]
    output_df['payload'] = output_df['payload'].str.split(',').str[-1]
    # Calculate delays (matching input and output packets)
    input_df['delay'] = np.nan
    input_df['drop'] = np.nan

    # total = input_df.shape[0]
    # for i, row in input_df.iterrows():
    #     print(f"Processing ... {i + 1}/{total}")
    #     hex_string = row['payload'].split(',')[-1]
    #     matching_row = output_df[output_df["payload"] == hex_string]
    #     if not matching_row.empty:
    #         # Use the first matching timestamp (if there are multiple matches)
    #         matching_timestamp = matching_row['timestamp'].iloc[0]
    #         input_df.at[i, "delay"] = abs(row["timestamp"] - matching_timestamp)
    #         output_df = output_df.drop(matching_row.index)
    #         input_df.at[i, 'drop'] = 0
    #     else:
    #         input_df.at[i, 'drop'] = 1 
    input_df = pd.merge(input_df, output_df, how='left', on='payload', suffixes=("", "_output"))
    input_df['delay'] = (input_df['timestamp'] - input_df['timestamp_output']).abs()
    input_df['drop'] = input_df['delay'].isna().astype(int)
    input_df = input_df.drop_duplicates(subset=['payload'])
    input_df = input_df.drop(columns=[col for col in input_df.columns if col.endswith('_output')])
    # Add metadata columns
    if cpu is not None:
        input_df['cpu'] = cpu  # Add CPU column for RAN and UPF
    if bw is not None:
        input_df['bw'] = bw  # Add bandwidth column for OVS

    # Drop unnecessary columns (after all calculations are complete)
    ip_columns = ["payload", "protocols", "id", 'length_frame', 'src', 'dst', 'length_ip', 'srcport', 'dstport']
    input_df = input_df.drop(ip_columns, axis=1)
    input_df = input_df.dropna(subset=[col for col in input_df.columns if col not in ['delay']])
    
    print(input_df.isna().sum())
    return input_df


def create_dataset(pcap_folder, vnf_type):
    # Create the output directory structure
    output_dir = f"{vnf_type}_dataset_new"
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the auto-profile directory structure
    for root, dirs, files in os.walk(pcap_folder):
        if "rate_" in root:
            # Extract metadata from the directory path
            path_parts = root.split(os.sep)
            dataset_type = path_parts[-4]  # train or test
            traffic_type = path_parts[-3]  # poisson or on_off

            # Handle OVS directory structure
            if vnf_type.lower() == "ovs":
                rate = int(path_parts[-1].split('_')[1])  # Extract rate from directory name
                bw = int(path_parts[-2].split('_')[1])    # Extract bandwidth from directory name
                cpu = None  # OVS does not have CPU
            else:
                # Handle RAN and UPF directory structure
                rate = int(path_parts[-1].split('_')[1])  # Extract rate from directory name
                cpu = int(path_parts[-2].split('_')[1])   # Extract CPU from directory name
                bw = None  # RAN and UPF do not have bandwidth

            # Find input and output CSV files (ignore non-CSV files)
            csv_files = [f for f in os.listdir(root) if f.endswith('.csv')]
            input_files = [f for f in csv_files if 'input' in f]
            output_files = [f for f in csv_files if 'output' in f]

            if not input_files or not output_files:
                print(f"Skipping directory {root}: Missing input or output CSV files.")
                continue

            input_file = os.path.join(root, input_files[0])
            output_file = os.path.join(root, output_files[0])

            # Process the CSV files
            processed_data = process_csv_files(input_file, output_file, traffic_type, rate, cpu, bw)

            # Save the processed data to train.csv or test.csv in the output directory
            if vnf_type.lower() == "ovs":
                output_subdir = os.path.join(output_dir, traffic_type, f"bw_{bw}", f"rate_{rate}")
            else:
                output_subdir = os.path.join(output_dir, traffic_type, f"cpu_{cpu}", f"rate_{rate}")
            os.makedirs(output_subdir, exist_ok=True)
            output_file_path = os.path.join(output_subdir, f"{dataset_type}.csv")
            processed_data.to_csv(output_file_path, index=False)
            print(f"Saved processed data to {output_file_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pcap_path', help='Path to the auto-profile directory')
    parser.add_argument('--vnf_type', required=True, help='Type of VNF (RAN, OvS, or UPF)')
    return parser.parse_args()


def main():
    args = get_args()
    create_dataset(args.pcap_path, args.vnf_type)


if __name__ == '__main__':
    main()



