#!/bin/bash

# Find the PID of the /srsran/gnb process
pid=$(ps ax | grep '[u]pf' | awk '{print $1}' | head -n 1)

# Check if the PID was found
if [[ -z "$pid" ]]; then
    echo "Process /srsran/gnb not found."
    exit 1
fi

# Display the PID
echo "PID of upf1: $pid"

# Get the cgroup information for the process and remove the "0::" prefix
cgroup_info=$(cat /proc/$pid/cgroup | grep 'cpu,cpuacct' | sed 's/^[^:]*:[^:]*://' | awk '{print "/sys/fs/cgroup/cpu" $0}')

# Check if we successfully got the cgroup info
if [[ -z "$cgroup_info" ]]; then
    echo "Failed to get cgroup information for PID $pid."
    exit 1
fi

# Set CPU limit in millicores (input argument)
MILLICORES=$1
if [[ -z "$MILLICORES" ]]; then
    echo "Usage: $0 <CPU limit in millicores (e.g., 500 for 500m)>"
    exit 1
fi

# Get the current cpu.cfs_period_us value
CFS_PERIOD_US=$(cat "$cgroup_info/cpu.cfs_period_us")

# Calculate cpu.cfs_quota_us based on millicores
CFS_QUOTA_US=$(echo "($MILLICORES * $CFS_PERIOD_US) / 1000" | bc)

# Set the CPU limit
echo "$CFS_QUOTA_US" | sudo tee "$cgroup_info/cpu.cfs_quota_us" > /dev/null
if [[ $? -ne 0 ]]; then
    echo "Failed to set CPU limit. Check permissions."
    exit 1
fi

# Verify the CPU limit
echo "CPU limit set for upf (PID $pid):"
echo "cpu.cfs_quota_us: $(cat $cgroup_info/cpu.cfs_quota_us)"
echo "cpu.cfs_period_us: $(cat $cgroup_info/cpu.cfs_period_us)"
echo "Millicores: $MILLICORES"
