import socket
import random
import argparse
import netifaces as ni
import time


def get_interface_details(interface_name: str) -> tuple[str, str | None]:
    """
    Retrieve IP and MAC address of the given network interface.

    Parameters:
        interface_name (str): Name of the network interface.

    Returns:
        tuple: (IP address, MAC address)
    """
    addrs = ni.ifaddresses(interface_name)
    ip_info = addrs[ni.AF_INET][0]['addr']
    mac_info = addrs.get(ni.AF_LINK, [{}])[0].get('addr')
    return ip_info, mac_info


def send_packet(sock: socket.socket, packet: str, destination_ip: str, destination_port: int) -> None:
    """
    Send a UDP packet over the socket.

    Parameters:
        sock (socket.socket): The UDP socket.
        packet (str): Packet payload.
        destination_ip (str): Destination IP.
        destination_port (int): Destination port.
    """
    sock.sendto(packet.encode(), (destination_ip, destination_port))


def poisson_traffic(sock: socket.socket, dst_ip: str, dst_port: int, rate_mbps: float, duration: float, max_packet_size: int) -> None:
    """
    Generate Poisson traffic.

    Parameters:
        sock (socket.socket): UDP socket.
        dst_ip (str): Destination IP.
        dst_port (int): Destination port.
        rate_mbps (float): Target rate in Mbps.
        duration (float): Duration of traffic in seconds.
        max_packet_size (int): Max size of a packet in bytes.
    """
    rate_pps = (rate_mbps * 1e6) / (max_packet_size * 8)
    sleep_factor = 1.0
    increase_factor = 1.1
    decrease_factor = 0.99

    start_time = time.time()
    total_packet_size = 0
    second_timer = time.time()

    while time.time() - start_time < duration:
        packet_size = random.randint(30, max_packet_size)
        payload = f"{time.time()}" + "X" * max(0, packet_size - len(str(time.time())))
        send_packet(sock, payload, dst_ip, dst_port)

        total_packet_size += packet_size
        time.sleep(max(0, random.expovariate(rate_pps)) / sleep_factor)

        # Log actual transmission rate every second
        if time.time() - second_timer >= 1.0:
            elapsed = time.time() - second_timer
            actual_rate = (total_packet_size * 8) / elapsed / 1e6  # in Mbps
            print(f"[Poisson] Actual rate: {actual_rate:.2f} Mbps")

            # Adjust sleep_factor
            if actual_rate < rate_mbps:
                sleep_factor *= increase_factor
            else:
                sleep_factor *= decrease_factor

            total_packet_size = 0
            second_timer = time.time()


def on_off_traffic(sock: socket.socket, dst_ip: str, dst_port: int, rate_mbps: float, duration: float, max_packet_size: int) -> None:
    """
    Generate ON-OFF traffic.

    Parameters:
        sock (socket.socket): UDP socket.
        dst_ip (str): Destination IP.
        dst_port (int): Destination port.
        rate_mbps (float): Rate during ON period (same used for OFF period distribution).
        duration (float): Total time to generate traffic.
        max_packet_size (int): Max packet size in bytes.
    """
    start_time = time.time()

    while time.time() - start_time < duration:
        on_duration = random.expovariate(rate_mbps)
        off_duration = random.expovariate(rate_mbps)

        print(f"[ON-OFF] On period: {on_duration:.2f}s, Off period: {off_duration:.2f}s")

        # ON period: Send packets
        end_on = time.time() + on_duration
        while time.time() < end_on:
            packet_size = random.randint(30, max_packet_size)
            payload = f"{time.time()}" + "X" * max(0, packet_size - len(str(time.time())))
            send_packet(sock, payload, dst_ip, dst_port)
            time.sleep(0.01)

        # OFF period: No packets
        time.sleep(off_duration)



def start_server(client_ip: str, port: int, traffic_type: str, rate_mbps: float, duration: float,
                 packet_size: int, server_ip: str) -> None:
    """
    Start UDP traffic generation based on the selected traffic type.

    Parameters:
        client_ip (str): IP address of the client.
        port (int): Destination port.
        traffic_type (str): One of ['poisson', 'on_off', 'map'].
        rate_mbps (float): Traffic rate in Mbps.
        duration (float): Duration of traffic generation.
        packet_size (int): Max packet size in bytes.
        server_ip (str): IP to bind the server socket.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((server_ip, port))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)

    print(f"[INFO] Sending {traffic_type} traffic to {client_ip}:{port} via {server_ip}")

    if traffic_type == "poisson":
        poisson_traffic(sock, client_ip, port, rate_mbps, duration, packet_size)
    elif traffic_type == "on_off":
        on_off_traffic(sock, client_ip, port, rate_mbps, duration, packet_size)

    sock.close()


def get_args():
    parser = argparse.ArgumentParser(description="UDP Traffic Generator")
    parser.add_argument("--client_ip", required=True, help="Client IP to send traffic to")
    parser.add_argument("--port", type=int, default=9999, help="Client UDP port")
    parser.add_argument("--traffic_type", choices=["poisson", "on_off"], default="poisson", help="Traffic pattern")
    parser.add_argument("--rate_mbps", type=float, default=20, help="Target rate in Mbps")
    parser.add_argument("--duration", type=int, default=60, help="Traffic generation duration in seconds")
    parser.add_argument("--packet_size", type=int, default=1500, help="Maximum packet size in bytes")
    parser.add_argument("--interface", type=str, default="n3", help="Network interface to use")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    client_ip = args.client_ip
    port = args.port
    traffic_type = args.traffic_type
    rate_mbps = args.rate_mbps
    duration = args.duration
    packet_size = args.packet_size
    interface = args.interface

    server_ip, _ = get_interface_details(interface)
    start_server(client_ip, port, traffic_type, rate_mbps, duration, packet_size, server_ip)
