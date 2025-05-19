# Variables
k8s-cpy.sh upf1 open5gs traffic_generator_udp.py

injection_interface="ogstun"
input_port="47627"
output_port="37257"
ran_name=$(kubectl get pods -n open5gs | grep "gnb" | awk '{print $1}')
ue_name=$(kubectl get pods -n open5gs | grep "ue" | awk '{print $1}')
train_duration=200
test_duration=60

# Arrays
rates=(5 10 15 20 25 30 35 40 45 50)  # Array of rates
bws=(100 150 200)    # Array of bandwidths (must be greater than the corresponding rate)
traffic_types=("poisson")  # Array of traffic types

# Remote server details
remote_server="transitvm"
remote_log_dir="./ovs_auto_profile"

# Ensure the base directory exists on the remote server
ssh $remote_server "mkdir -p '$remote_log_dir/train' '$remote_log_dir/test'"

# Loop through traffic types, rates, and bandwidths for both training and testing
for dataset_type in "train"; do
    duration=$([ "$dataset_type" == "train" ] && echo $train_duration || echo $test_duration)

    for traffic_type in "${traffic_types[@]}"; do
        for bw in "${bws[@]}"; do
            for rate in "${rates[@]}"; do
                # Ensure bandwidth is greater than rate
                if [ "$bw" -le "$rate" ]; then
                    echo "Skipping combination: rate=$rate Mbps, bw=$bw Mbps (bandwidth must be greater than rate)"
                    continue
                fi

                # Create subdirectories for rate and bw
                sub_dir="$remote_log_dir/$dataset_type/$traffic_type/bw_${bw}/rate_${rate}"

                # Create subdirectories on the remote server
                ssh $remote_server "mkdir -p '$sub_dir'"

                # Get client IP
                client_ip=$(k8s-exec.sh ue1 open5gs "ip netns exec ue1 ifconfig" | grep "10.41.0" | awk '{print $2}')
                echo "Client IP: $client_ip"

                # Update bandwidth
                echo "Updating bandwidth to $bw Mbps for traffic type $traffic_type"
                ssh $remote_server "sudo bash ./k8s_srsran_open5gs/configs/onos/scripts/update_queue.sh $bw"
                sleep 10

                # Start capturing traffic on the remote server
                echo "Starting tcpdump for traffic capture on the remote server..."
                ssh $remote_server "sudo tcpdump -i eno1 src port $input_port -w $sub_dir/input_${traffic_type}_${rate}_${bw}.pcap" &
                ssh $remote_server "sudo tcpdump -i eno1 src port $output_port -w $sub_dir/output_${traffic_type}_${rate}_${bw}.pcap" &
                sleep 10

                # Run traffic generator
                echo "Running traffic generator for traffic type $traffic_type, rate $rate Mbps, and bandwidth $bw Mbps..."
                k8s-exec.sh upf1 open5gs "python3 /tmp/traffic_generator_udp.py --client_ip $client_ip --interface $injection_interface --traffic_type $traffic_type --rate_mbps $rate --duration $duration"

                # Stop capturing traffic
                ssh $remote_server "sudo pkill -f tcpdump"
                sleep 60

                # Convert PCAP to CSV on the server
                echo "Converting PCAP files to CSV on the remote server..."
                ssh $remote_server "bash pcap_to_csv.sh $sub_dir/input_${traffic_type}_${rate}_${bw}.pcap"
                ssh $remote_server "bash pcap_to_csv.sh $sub_dir/output_${traffic_type}_${rate}_${bw}.pcap"
                sleep 10

                echo "Completed test for $dataset_type: traffic type $traffic_type, rate $rate Mbps, bandwidth $bw Mbps."
            done
        done
    done
done
scp -r transitvm:$remote_log_dir /mnt/hdd/profiling
echo "Finished all tests."
