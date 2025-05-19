k8s-cpy.sh upf1 open5gs traffic_generator_udp.py
# Variables
injection_interface="ogstun"
input_interface="n3"
output_interface="tun_srsue"
ran_name=$(kubectl get pods -n open5gs | grep "gnb" | awk '{print $1}')
ue_name=$(kubectl get pods -n open5gs | grep "ue" | awk '{print $1}')
train_duration=300
test_duration=10

# Arrays
traffic_types=("poisson")
cpu_values=(1500 1400 1300)
rates=(5 10 15 20 25 30 35 40 45 50)

# Log directory
log_dir="./ran_auto_profile"
mkdir -p "$log_dir"   # Create the directory if it doesn't exist

#Set bw of the transport network
ssh transitvm "sudo bash ./k8s_srsran_open5gs/configs/onos/scripts/update_queue.sh 2000000"
sleep 10

# Loop through dataset types (train and test)
for dataset_type in "train"; do
    duration=$([ "$dataset_type" == "train" ] && echo $train_duration || echo $test_duration)

    # Loop through traffic types, CPU values, and rates
    for t_type in "${traffic_types[@]}"; do
        for c in "${cpu_values[@]}"; do
            for r in "${rates[@]}"; do
                # Create subdirectory for the current test
                sub_dir="$log_dir/$dataset_type/$t_type/cpu_$c/rate_$r"
                mkdir -p "$sub_dir"

                # Set CPU using sudo mode
                sudo bash set_cgroup_cpu.sh $c
                echo "CPU set to: $c"

                # Get client IP
                client_ip=$(k8s-exec.sh ue1 open5gs "ip netns exec ue1 ifconfig" | grep "10.41.0" | awk '{print $2}')
                echo "Client IP: $client_ip"

                # Start tcpdump
                k8s-exec.sh ue1 open5gs "ip netns exec ue1 tcpdump -i tun_srsue -w /mnt/output_${t_type}_${c}_${r}.pcap" &
                kubectl exec -it $ran_name -n open5gs -c gnb -- bash -c "tcpdump -i n3 src 10.10.3.1 -w /mnt/input_${t_type}_${c}_${r}.pcap" &

                # Run the UDP generator
                k8s-exec.sh upf1 open5gs "python3 /tmp/traffic_generator_udp.py --client_ip $client_ip --interface $injection_interface --traffic_type $t_type --rate_mbps $r --duration $duration"
                echo "Traffic injection done for type: $t_type, CPU: $c, Rate: $r"

                # Stop tcpdump
                k8s-exec.sh ue1 open5gs 'pkill -SIGINT -f tcpdump' &
                kubectl exec -it $ran_name -n open5gs -c gnb -- bash -c 'pkill -SIGINT -f tcpdump' &
                sleep 60

                # Copy the pcap files to the local machine
                kubectl cp open5gs/$ran_name:/mnt/input_${t_type}_${c}_${r}.pcap "$sub_dir/input_${t_type}_${c}_${r}.pcap"
                kubectl cp open5gs/$ue_name:/mnt/output_${t_type}_${c}_${r}.pcap "$sub_dir/output_${t_type}_${c}_${r}.pcap"
		
		
                # Convert PCAP to CSV
                bash pcap_to_csv.sh "$sub_dir/input_${t_type}_${c}_${r}.pcap"
                bash pcap_to_csv.sh "$sub_dir/output_${t_type}_${c}_${r}.pcap"
                sleep 10;
		kubectl exec -it $ran_name -n open5gs -c gnb -- bash -c "rm /mnt/input_${t_type}_${c}_${r}.pcap"
		k8s-exec.sh ue1 open5gs "rm /mnt/output_${t_type}_${c}_${r}.pcap"
            done
        done
    done
done

echo "Finished all tests."
