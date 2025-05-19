import socket
import random
import argparse
import netifaces as ni
import threading
import time, os


def get_interface_details(interface_name):
    addrs = ni.ifaddresses(interface_name)
    ip_info = addrs[ni.AF_INET][0]['addr']
    mac_info = None
    try:
        mac_info = addrs[ni.AF_LINK][0]['addr']
    except KeyError:
        mac_info = None
    return ip_info, mac_info


def send_packet(client_socket, packet, destination_ip, destination_port):
    """Send packet data over the UDP socket."""
    client_socket.sendto(packet, (destination_ip, destination_port))

def poisson_traffic(client_socket, destination_ip, destination_port, rate_mbps, duration, packet_size):
    """Generate and send Poisson traffic."""
    rate = (rate_mbps * 1e6) / (packet_size * 8)  # Convert Mbps to packets per second
    start_time = time.time()
    packets_sent = 0  # Track the number of packets sent
    sleep_factor = 1
    increase_factor = 1.1
    decrease_factor = 0.99
    st = time.time()
    start_time = time.time()
    total_packet_size = 0
    while time.time() - st < duration:
        packet = f"{time.time()}"
        packet_size = int(random.uniform(30, 1500))
        packet = packet + "X" * (packet_size - len(packet))
        send_packet(client_socket, packet.encode(), destination_ip, destination_port)
        packets_sent += 1  # Increment the packet count
        interarrival_time = max(0, random.expovariate(rate)) / sleep_factor
        time.sleep(interarrival_time)
        total_packet_size += packet_size

        # Print the traffic rate every second
        if time.time() - start_time >= 1:
            actual_rate = (total_packet_size * 8) / (time.time() - start_time) / 1e6  # Convert to Mbps
            print(f"Actual traffic rate: {actual_rate:.2f} Mbps")
            packets_sent = 0  # Reset the packet count
            start_time = time.time()  # Reset the start time for the next second
            if actual_rate < rate_mbps:
                sleep_factor = sleep_factor * increase_factor
            else:
                sleep_factor = sleep_factor * decrease_factor
                if increase_factor > 1.01:
                    st = time.time()
                increase_factor = 1.01
            total_packet_size = 0

def on_off_traffic(client_socket, destination_ip, destination_port, on_rate_mbps, off_rate_mbps, duration, packet_size):
    """Generate and send ON-OFF traffic."""
    on_rate = (on_rate_mbps * 1e6) / (packet_size * 8)  # Convert Mbps to packets per second
    off_rate = (off_rate_mbps * 1e6) / (packet_size * 8)  # Convert Mbps to packets per second
    start_time = time.time()
    while time.time() - start_time < duration:
        on_period = random.expovariate(on_rate)
        end_on_period = time.time() + on_period
        print(f'On Period {on_period}')
        while time.time() < end_on_period:
            packet = f"{time.time()}"
            packet_size = int(random.uniform(30, 1500))
            packet = packet + "X" * (packet_size - len(packet))
            #packet = f"X" * packet_size
            send_packet(client_socket, packet.encode(), destination_ip, destination_port)
            time.sleep(0.01)  # Small delay to simulate packet transmission
        off_period = random.expovariate(off_rate)
        print(f'Off Period {off_period}')
        time.sleep(off_period)

def map_traffic(client_socket, destination_ip, destination_port, arrival_rate_mbps, duration, packet_size):
    """Generate and send Markovian Arrival Process (MAP) traffic."""
    arrival_rate = (arrival_rate_mbps * 1e6) / (packet_size * 8)  # Convert Mbps to packets per second
    start_time = time.time()
    while time.time() - start_time < duration:
        packet = f"{time.time()}"
        packet_size = int(random.uniform(30, 1500))
        packet = packet + "X" * (packet_size - len(packet))
        #packet = f"X" * packet_size
        send_packet(client_socket, packet.encode(), destination_ip, destination_port)
        time.sleep(random.expovariate(arrival_rate))

def start_server(client_ip, port, traffic_type, rate_mbps, duration, packet_size, server_ip):
    """Start the server and send packets to the client."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Use UDP
    server_socket.bind((server_ip, port))  # Bind to the specific interface
    server_socket.setblocking(0)  # Set the socket to non-blocking mode
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)  # 1 MB buffer

    print(f"Server is sending {traffic_type} traffic to {client_ip}:{port} on interface with IP {server_ip}...")

    if traffic_type == "poisson":
        poisson_traffic(server_socket, client_ip, port, rate_mbps, duration, packet_size)
    elif traffic_type == "on_off":
        on_off_traffic(server_socket, client_ip, port, rate_mbps, rate_mbps, duration, packet_size)
    elif traffic_type == "map":
        map_traffic(server_socket, client_ip, port, rate_mbps, duration, packet_size)

    server_socket.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_ip", type=str, required=True, help="Client IP address to send traffic to")
    parser.add_argument("--port", type=int, default=9999, help="Client port")
    parser.add_argument("--traffic_type", choices=["poisson", "on_off", "map"], default="poisson", help="Type of traffic to generate")
    parser.add_argument("--rate_mbps", type=int, default=20, help="Rate in Mbps")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--packet_size", type=int, default=1500, help="Packet size in bytes")
    parser.add_argument("--interface", type=str, default="n3", help="Network interface to use")  # New argument

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    CLIENT_IP = args.client_ip
    PORT = args.port
    TRAFFIC_TYPE = args.traffic_type
    RATE_MBPS = args.rate_mbps
    DURATION = args.duration
    PACKET_SIZE = args.packet_size
    INTERFACE = args.interface  # New variable

    SERVER_IP, _ = get_interface_details(INTERFACE)

    start_server(CLIENT_IP, PORT, TRAFFIC_TYPE, RATE_MBPS, DURATION, PACKET_SIZE, SERVER_IP)  # Pass the interface
