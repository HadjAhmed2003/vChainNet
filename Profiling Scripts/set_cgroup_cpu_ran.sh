#!/bin/bash

# Find the PID of the /srsran/gnb process
pid=$(ps ax | grep '[/]srsran/gnb' | awk '{print $1}')

# Check if the PID was found
if [[ -z "$pid" ]]; then
    echo "Process /srsran/gnb not found."
    exit 1
fi

# Display the PID
echo "PID of /srsran/gnb: $pid"

# Get the cgroup information for the process and remove the "0::" prefix
cgroup_info=$(cat /proc/$pid/cgroup | sed 's/^0:://')

# Check if we successfully got the cgroup info
if [[ -z "$cgroup_info" ]]; then
    echo "Failed to get cgroup information for PID $pid."
    exit 1
fi

CPU=$1
# echo /sys/fs/cgroup$cgroup_info
echo "${CPU}00 100000" | sudo tee /sys/fs/cgroup$cgroup_info/cpu.max > /dev/null
cat /sys/fs/cgroup/$cgroup_info/cpu.max

# echo $CPU >> /mnt/cpu_log.txt