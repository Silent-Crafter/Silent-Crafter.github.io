def ip_to_binary(ip):
    return [f"{int(x):08b}" for x in ip.split('.')]

def binary_to_ip(binary):
    print(binary)
    return '.'.join(str(int(oct, 2)) for oct in binary)

def int_to_ip(addr):
    return ".".join(str((addr >> (8 * i)) & 0xFF) for i in reversed(range(4)))

def divide_network(ip, mask, n):
    ip_binary = ip_to_binary(ip)
    network_binary = ''.join(ip_binary)[:mask] + '0' * (32 - mask)
    network_binary = [network_binary[i:i+8] for i in range(0, 32, 8)]
    network_ip = int(''.join(network_binary), 2)

    bits_needed = 0
    while (1 << bits_needed) < n:
        bits_needed += 1

    new_mask = mask + bits_needed
    print(f"{ip_binary=}\n{network_binary=}\n{network_ip=}\n{mask=}\n{new_mask=}")

    subnets = []
    try:
        subnet_size = 1 << (32 - new_mask)
    except ValueError:
        print("Subnetting not possible")
        return []

    for i in range(n):
        subnet_start = network_ip + (i * subnet_size)
        subnet_start = int_to_ip(subnet_start)
        subnets.append((subnet_start, new_mask))

    return subnets

def cidr_to_netmask(cidr):
    mask = '1' * cidr + '0' * (32 - cidr)
    return binary_to_ip(mask)

def calculate_network_info(ip_cidr) -> dict:
    info = {}
    ip, cidr = ip_cidr.split('/')
    cidr = int(cidr)

    ip_binary = ip_to_binary(ip)
    print(ip_binary)

    network_binary = ip_binary[:cidr] + '0' * (32 - cidr)
    broadcast_binary = ip_binary[:cidr] + '1' * (32 - cidr)

    network_address = binary_to_ip(network_binary)
    broadcast_address = binary_to_ip(broadcast_binary)
    netmask = cidr_to_netmask(cidr)

    num_hosts = (2 ** (32 - cidr)) - 2

    info["Network Address"] = network_address
    info["Broadcast Address"] = broadcast_address
    info["Number of Hosts"] = num_hosts
    info["Subnet Mask"] = netmask

    return info


if __name__ == "__main__":
    user_input = input("Enter an IP address with CIDR (e.g., 192.168.1.0/24): ")
    n = int(input("Enter number of subnets: "))
    ip, mask = user_input.split('/')
    mask = int(mask)

    subnets = divide_network(ip, mask, n)
    print(subnets)
    for subnet in subnets:
        print(f"Subnet: {subnet[0]}, Mask: /{subnet[1]}")
