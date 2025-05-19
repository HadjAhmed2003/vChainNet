# Variables
injection_interface="n3"
input_interface="n3"
output_interface="tun_srsue"
iperf_name=$(kubectl get pods -n open5gs | grep "iperf2-server" | awk '{print $1}')
upf_name=$(kubectl get pods -n open5gs | grep "upf1" | awk '{print $1}')
ran_name=$(kubectl get pods -n open5gs | grep "gnb" | awk '{print $1}')
ue_name=$(kubectl get pods -n open5gs | grep "ue" | awk '{print $1}')
train_duration=20
test_duration=10

# Arrays
traffic_types=("on_off")
cpu_values_upf=(1500 1400)  # CPU allocations for UPF
cpu_values_ran=(1500 1400)    # CPU allocations for RAN
bws=(60 55)                # Bandwidth values for OvS
rates=(20 15 10) # Traffic rates
remote_server="transitvm"

# Log directory
log_dir="/mnt/hdd/profiling/dataset"
mkdir -p "$log_dir/slice_auto_profile" "$log_dir/upf_auto_profile" "$log_dir/ovs_auto_profile" "$log_dir/ran_auto_profile"

# Loop through dataset types (train and test)
for dataset_type in "train"; do
    duration=$([ "$dataset_type" == "train" ] && echo $train_duration || echo $test_duration)
    # Loop through traffic types, CPU values for UPF, CPU values for RAN, and bandwidths
    for t_type in "${traffic_types[@]}"; do
        for c_upf in "${cpu_values_upf[@]}"; do
            for c_ran in "${cpu_values_ran[@]}"; do
                for bw in "${bws[@]}"; do
                    # Set bandwidth for OvS
                    echo "Updating bandwidth to $bw Mbps"
                    ssh $remote_server "sudo bash ./k8s_srsran_open5gs/configs/onos/scripts/update_queue.sh $bw"
                    sleep 1

                    # Loop through rates
                    for r in "${rates[@]}"; do
                        # Skip if bandwidth is less than or equal to the rate
                        if [ "$bw" -le "$r" ]; then
                            echo "Skipping combination: rate=$r Mbps, bw=$bw Mbps (bandwidth must be greater than rate)"
                            continue
                        fi

                        # Create subdirectories for each test
                        slice_sub_dir="$log_dir/slice_auto_profile"
                        upf_sub_dir="$log_dir/upf_auto_profile"
                        ovs_sub_dir="$log_dir/ovs_auto_profile"
                        ran_sub_dir="$log_dir/ran_auto_profile"
                        mkdir -p "$slice_sub_dir" "$upf_sub_dir" "$ovs_sub_dir" "$ran_sub_dir"

                        # Set CPU for UPF
                        ssh corevm "sudo bash set_cgroup_cpu_upf.sh $c_upf"
                        echo "UPF CPU set to: $c_upf"

                        # Set CPU for RAN
                        sudo bash set_cgroup_cpu_ran.sh $c_ran
                        echo "RAN CPU set to: $c_ran"

                        # Get client IP
                        client_ip=$(k8s-exec.sh ue1 open5gs "ip netns exec ue1 ifconfig" | grep "10.41.0" | awk '{print $2}')
                        echo "Client IP: $client_ip"

                        # Start tcpdump for E2E (slice)
                        k8s-exec.sh iperf open5gs "tcpdump -i $input_interface -w /mnt/input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap" &
                        k8s-exec.sh ue1 open5gs "ip netns exec ue1 tcpdump -i $output_interface -w /mnt/output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap" &

                        # Start tcpdump for UPF
                        k8s-exec.sh iperf open5gs "tcpdump -i $input_interface -w /mnt/upf_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap" &
                        k8s-exec.sh upf1 open5gs "tcpdump -i ogstun -w /mnt/upf_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap" &

                        # Start tcpdump for OvS (on remote server)
                        ssh $remote_server "sudo tcpdump -i eno1 src port 47627 -w input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap" &
                        ssh $remote_server "sudo tcpdump -i eno1 src port 37257 -w output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap" &

                        # Start tcpdump for RAN
                        kubectl exec -it $ran_name -n open5gs -c gnb -- bash -c "tcpdump -i n3 src 10.10.3.1 -w /mnt/ran_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap" &
                        k8s-exec.sh ue1 open5gs "ip netns exec ue1 tcpdump -i $output_interface -w /mnt/ran_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap" &

                        # Run the UDP generator
                        k8s-exec.sh iperf open5gs "python3 /tmp/traffic_generator_udp.py --client_ip $client_ip --interface $injection_interface --traffic_type $t_type --rate_mbps $r --duration $duration"
                        echo "Traffic injection done for type: $t_type, UPF CPU: $c_upf, RAN CPU: $c_ran, Bandwidth: $bw Mbps, Rate: $r Mbps"

                        # Stop tcpdump for E2E (slice)
                        k8s-exec.sh iperf open5gs 'pkill -SIGINT -f tcpdump' &
                        k8s-exec.sh ue1 open5gs 'pkill -SIGINT -f tcpdump' &

                        # Stop tcpdump for UPF
                        k8s-exec.sh upf1 open5gs 'pkill -SIGINT -f tcpdump' &

                        # Stop tcpdump for OvS (on remote server)
                        ssh $remote_server "sudo pkill -f tcpdump" &

                        # Stop tcpdump for RAN
                        kubectl exec -it $ran_name -n open5gs -c gnb -- bash -c 'pkill -SIGINT -f tcpdump' &

                        sleep 5  # Wait for tcpdump to finish writing

                        # Copy and convert PCAP files for E2E (slice)
                        kubectl cp open5gs/$iperf_name:/mnt/input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap "$slice_sub_dir/input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        kubectl cp open5gs/$ue_name:/mnt/output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap "$slice_sub_dir/output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        bash pcap_to_csv.sh "$slice_sub_dir/input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap";
                        bash pcap_to_csv.sh "$slice_sub_dir/output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap";

                        # Copy and convert PCAP files for UPF
                        kubectl cp open5gs/$iperf_name:/mnt/upf_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap "$upf_sub_dir/upf_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        kubectl cp open5gs/$upf_name:/mnt/upf_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap "$upf_sub_dir/upf_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        bash pcap_to_csv.sh "$upf_sub_dir/upf_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap";
                        bash pcap_to_csv.sh "$upf_sub_dir/upf_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap";

                        # Copy PCAP files for OvS from remote server to local machine
                        scp $remote_server:input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap "$ovs_sub_dir/ovs_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        scp $remote_server:output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap "$ovs_sub_dir/ovs_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"

                        # Convert PCAP files for OvS locally
                        bash pcap_to_csv.sh "$ovs_sub_dir/ovs_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap";
                        bash pcap_to_csv.sh "$ovs_sub_dir/ovs_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap";

                        # Copy and convert PCAP files for RAN
                        kubectl cp open5gs/$ran_name:/mnt/ran_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap "$ran_sub_dir/ran_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        kubectl cp open5gs/$ue_name:/mnt/ran_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap "$ran_sub_dir/ran_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        bash pcap_to_csv.sh "$ran_sub_dir/ran_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap";
                        bash pcap_to_csv.sh "$ran_sub_dir/ran_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap";

                        # Clean up remote PCAP files
                        ssh $remote_server "rm input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        kubectl exec -it $ran_name -n open5gs -c gnb -- bash -c "rm /mnt/ran_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        k8s-exec.sh ue1 open5gs "rm /mnt/output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap /mnt/ran_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        k8s-exec.sh iperf open5gs "rm /mnt/input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap /mnt/upf_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        k8s-exec.sh upf1 open5gs "rm /mnt/upf_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        #Clean up local PCAP files
                        rm "$slice_sub_dir/input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        rm "$slice_sub_dir/output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        
                        rm "$upf_sub_dir/upf_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        rm "$upf_sub_dir/upf_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        
                        rm "$ovs_sub_dir/ovs_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        rm "$ovs_sub_dir/ovs_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        
                        rm "$ran_sub_dir/ran_input_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                        rm "$ran_sub_dir/ran_output_${t_type}_${c_upf}_${c_ran}_${bw}_${r}.pcap"
                    done
                done
            done
        done
    done
done

echo "Finished all tests."
