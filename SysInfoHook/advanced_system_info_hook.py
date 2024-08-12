import os
import platform
import psutil
import socket
from datetime import timedelta

def get_os_info():
    print("Operating System Information")
    print("----------------------------")
    print(f"System: {platform.system()}")
    print(f"Node Name: {platform.node()}")
    print(f"Release: {platform.release()}")
    print(f"Version: {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print()

def get_cpu_info():
    print("CPU Information")
    print("---------------")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Total cores: {psutil.cpu_count(logical=True)}")
    print(f"CPU usage: {psutil.cpu_percent(interval=1)}%")
    print()

def get_memory_info():
    print("Memory Information")
    print("------------------")
    mem = psutil.virtual_memory()
    print(f"Total: {get_size(mem.total)}")
    print(f"Available: {get_size(mem.available)}")
    print(f"Used: {get_size(mem.used)}")
    print(f"Percentage: {mem.percent}%")
    print()

def get_disk_info():
    print("Disk Information")
    print("----------------")
    partition_info = psutil.disk_partitions()
    for partition in partition_info:
        print(f"Device: {partition.device}")
        print(f"  Mountpoint: {partition.mountpoint}")
        print(f"  File system type: {partition.fstype}")
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
            print(f"  Total Size: {get_size(partition_usage.total)}")
            print(f"  Used: {get_size(partition_usage.used)}")
            print(f"  Free: {get_size(partition_usage.free)}")
            print(f"  Percentage: {partition_usage.percent}%")
        except PermissionError:
            print("  Access Denied")
        print()

def get_network_info():
    print("Network Information")
    print("-------------------")
    print(f"Hostname: {socket.gethostname()}")
    try:
        print(f"IP Address: {socket.gethostbyname(socket.gethostname())}")
    except socket.error:
        print("IP Address: Unable to retrieve")
    
    net_io = psutil.net_io_counters()
    print(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    print(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")
    
    interfaces = psutil.net_if_addrs()
    for interface_name, interface_addresses in interfaces.items():
        for address in interface_addresses:
            print(f"Interface: {interface_name}")
            if str(address.family) == 'AddressFamily.AF_INET':
                print(f"  IP Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == 'AddressFamily.AF_PACKET':
                print(f"  MAC Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast MAC: {address.broadcast}")
    print()

def get_process_info():
    print("Top 5 Processes by CPU and Memory Usage")
    print("---------------------------------------")
    
    # Get all processes sorted by CPU and memory usage
    processes = sorted(psutil.process_iter(attrs=['pid', 'name', 'cpu_percent', 'memory_info']),
                       key=lambda p: (p.info['cpu_percent'], p.info['memory_info'].rss), reverse=True)
    
    for proc in processes[:5]:
        try:
            print(f"PID: {proc.info['pid']}, Name: {proc.info['name']}")
            print(f"  CPU Usage: {proc.info['cpu_percent']}%")
            print(f"  Memory Usage: {get_size(proc.info['memory_info'].rss)}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    print()

def get_battery_info():
    print("Battery Information")
    print("-------------------")
    if psutil.sensors_battery():
        battery = psutil.sensors_battery()
        print(f"Battery Percentage: {battery.percent}%")
        print(f"Power Plugged In: {battery.power_plugged}")
        print(f"Time Left: {str(timedelta(seconds=battery.secsleft)) if not battery.power_plugged else 'N/A'}")
    else:
        print("No battery information available.")
    print()

def get_uptime_info():
    print("Uptime Information")
    print("------------------")
    uptime_seconds = timedelta(seconds=int(psutil.boot_time()))
    uptime = timedelta(seconds=int((psutil.time.time() - psutil.boot_time())))
    print(f"System Uptime: {uptime}")
    print()

def get_temperature_info():
    print("Temperature Sensors")
    print("-------------------")
    if hasattr(psutil, "sensors_temperatures"):
        temps = psutil.sensors_temperatures()
        if not temps:
            print("No temperature sensors available.")
        for name, entries in temps.items():
            print(f"{name}:")
            for entry in entries:
                print(f"  {entry.label or name} - Current: {entry.current}°C, High: {entry.high}°C, Critical: {entry.critical}°C")
    else:
        print("Temperature sensors not supported on this platform.")
    print()

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format, e.g., 1253656 => '1.20MB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def main():
    print("Advanced System Information Hook")
    print("===============================\n")
    
    get_os_info()
    get_cpu_info()
    get_memory_info()
    get_disk_info()
    get_network_info()
    get_process_info()
    get_battery_info()
    get_uptime_info()
    get_temperature_info()

if __name__ == "__main__":
    main()
