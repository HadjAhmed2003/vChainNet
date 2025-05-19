# vChainNet: Accurate and Scalable End-to-End Slice Modeling for 5G and Beyond Networks

vChainNet is a deep learning framework for predicting end-to-end performance in 5G network slices. It uses a lightweight shallow LSTM architecture to accurately model packet delays across virtualized network functions (VNFs) in a service chain.

## Overview

Modern 5G networks rely on network slicing to provide customized services with specific performance guarantees. vChainNet addresses the challenge of accurately predicting end-to-end performance across complex service chains composed of multiple virtualized network functions (VNFs).

The framework uses deep learning to:
1. Profile individual VNF performance under various resource constraints
2. Create datasets capturing packet-level delays through each VNF
3. Train specialized models for each VNF in the service chain
4. Predict end-to-end performance by chaining these models together

## Dataset

The vChainNet dataset is available for download at:
[https://drive.google.com/file/d/1fI2zy-wKQmaAO7c1DYOujrmbHmDcTD3F/view?usp=sharing](https://drive.google.com/file/d/1fI2zy-wKQmaAO7c1DYOujrmbHmDcTD3F/view?usp=sharing)

The dataset contains packet-level measurements from three key VNFs in a 5G service chain:
- User Plane Function (UPF)
- Radio Access Network (RAN)
- Open vSwitch (OvS)

Each VNF dataset includes:
- Packet timestamps
- Interarrival times
- Packet sizes
- Resource constraints (CPU limits, bandwidth)
- Measured delays

## Testbed Deployment

For testbed deployment, refer to the testbed-automator repository:
[https://github.com/niloysh/testbed-automator](https://github.com/niloysh/testbed-automator)

The testbed-automator provides scripts and configurations for deploying a Kubernetes-based 5G testbed with Open5GS and srsRAN components.

## Repository Structure

```
vChainNet/
├── Profiling Scripts
│   ├── upf_profiler.sh         # Profiles UPF performance
│   ├── ran_profiler.sh         # Profiles RAN performance
│   ├── ovs_profiler.sh         # Profiles OvS performance
│   ├── e2e_profiler.sh         # End-to-end profiling
│   ├── set_cgroup_cpu_upf.sh   # Sets CPU limits for UPF
│   ├── set_cgroup_cpu_ran.sh   # Sets CPU limits for RAN
│   └── pcap_to_csv.sh          # Converts pcap to CSV
│
├── Traffic Generation
│   └── traffic_generator_udp.py # Generates UDP traffic patterns
│
├── Dataset Creation
│   ├── dataset_creator.py      # Creates per-VNF datasets from profiling data
│   └── slice_dataset_creator.py # Creates end-to-end slice datasets
│   
│
├── Model Implementation
│   └── deepptm.py              # Deep Packet-Time Model implementation (reference only)
│
└── Training and Prediction
    ├── traffic_traces.py       # Pytorch Dataset Class for the traffic traces
    ├── train_val.py            # Model training and validation (includes shallow LSTM)
    └── e2e_predict.py          # End-to-end prediction (includes shallow LSTM)
```

## Installation and Setup

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Pandas, NumPy, Matplotlib
- tcpdump, tshark
- Kubernetes cluster (for testbed deployment)

### Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vChainNet.git
   cd vChainNet
   ```

2. Install Python dependencies:
   ```bash
   pip install torch pandas numpy matplotlib scikit-learn tqdm
   ```

3. Download the dataset from the provided Google Drive link and extract it:
   ```bash
   # After downloading, extract to the project directory
   unzip vChainNet_dataset.zip -d ./model_dataset
   ```

## Usage

### Data Collection

To collect new profiling data from a deployed testbed:

1. Set up the testbed using the testbed-automator repository.

2. Run the profiling scripts for each VNF:
   ```bash
   # Profile UPF
   bash upf_profiler.sh
   
   # Profile RAN
   bash ran_profiler.sh
   
   # Profile OvS
   bash ovs_profiler.sh
   ```

   **Important**: The traffic generation script must be deployed within the pod of the VNF preceding the target VNF being profiled. For example, when profiling the UPF, the traffic generator should be in the preceding network element.

3. Convert the captured pcap files to CSV:
   ```bash
   bash pcap_to_csv.sh /path/to/pcap/file.pcap
   ```

### Dataset Creation

vChainNet uses two distinct dataset creation processes:

1. **Per-VNF Dataset Creation** (`dataset_creator.py`):
   This script processes the raw profiling data for individual VNFs, extracting features and calculating delays.

   ```bash
   # Create UPF dataset
   python dataset_creator.py /path/to/upf_auto_profile --vnf_type UPF

   # Create RAN dataset
   python dataset_creator.py /path/to/ran_auto_profile --vnf_type RAN

   # Create OvS dataset
   python dataset_creator.py /path/to/ovs_auto_profile --vnf_type OvS
   ```

2. **End-to-End Dataset Creation** (`slice_dataset_creator.py`):
   This script combines the individual VNF datasets to create end-to-end slice datasets for training and evaluating the complete service chain.

   ```bash
   # Create end-to-end slice dataset
   python slice_dataset_creator.py
   ```

### Model Training

To train models for each VNF:

```bash
# Edit the paths in train_val.py to point to your dataset
# Then run:
python train_val.py
```

The script will:
1. Load and preprocess the dataset
2. Split it into training and testing sets
3. Train the shallow LSTM model
4. Evaluate performance using Wasserstein distance
5. Save the trained model and performance plots

### End-to-End Prediction

To predict end-to-end performance across the service chain:

```bash
# Edit the paths in e2e_predict.py to point to your trained models
# Then run:
python e2e_predict.py
```

This will:
1. Load the trained models for each VNF
2. Chain them together to predict end-to-end performance
3. Generate performance comparison plots and metrics

## Model Architecture

vChainNet primarily uses a lightweight shallow LSTM architecture that outperforms more complex models while reducing computational complexity by 95%. This model is implemented in both `train_val.py` and `e2e_predict.py`.

The shallow LSTM model:
1. Efficiently captures temporal dependencies in packet sequences
2. Uses a custom loss function based on Wasserstein distance to accurately model delay distributions
3. Achieves superior performance with significantly reduced complexity

The repository also includes a reference implementation of deepPTM (Deep Packet-Time Model) that combines:
1. Bidirectional LSTM layers
2. Multi-head attention mechanisms
3. More complex feature processing

Both models take as input:
- Packet interarrival times
- Packet sizes
- Resource constraints (CPU, bandwidth)
- Workload metrics

And predict:
- Packet delay distributions through each VNF
- End-to-end delay across the service chain

## Resource Management

The framework includes scripts for controlling resource allocation:

```bash
# Set CPU limits for UPF
sudo bash set_cgroup_cpu_upf.sh <cpu_limit>

# Set CPU limits for RAN
sudo bash set_cgroup_cpu_ran.sh <cpu_limit>
```

These scripts use Linux cgroups to limit CPU resources available to each VNF, allowing for controlled experiments with different resource constraints.

## Traffic Generation

The `traffic_generator_udp.py` script can generate different traffic patterns:

```bash
python traffic_generator_udp.py --client_ip <ip> --interface <interface> --traffic_type <type> --rate_mbps <rate> --duration <seconds>
```

Supported traffic types:
- `poisson`: Poisson arrival process
- `on_off`: ON-OFF traffic pattern
- `map`: Markovian Arrival Process

**Important**: The traffic generator must be deployed within the pod of the VNF preceding the target VNF being profiled to ensure accurate measurement of packet processing delays.

## Contributing

Contributions to vChainNet are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use vChainNet in your research, please cite:

```
@article{vChainNet,
  title={vChainNet: vChainNet: Accurate and Scalable End-to-End Slice Modeling for 5G and Beyond Networks}
}
```

The deepPTM code is based on the implementation from:
[https://github.com/ostjul/COS561_final_project](https://github.com/ostjul/COS561_final_project)

## Acknowledgments

- The testbed deployment uses components from the [Open5GS](https://open5gs.org/) and [srsRAN](https://www.srslte.com/) projects.
- The testbed automation is based on the [testbed-automator](https://github.com/niloysh/testbed-automator) repository.
- The deepPTM implementation is adapted from the [COS561_final_project](https://github.com/ostjul/COS561_final_project) repository.
