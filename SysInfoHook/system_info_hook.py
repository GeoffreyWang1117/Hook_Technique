import os
import platform
import psutil

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
    net_io = psutil.net_io_counters()
    print(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    print(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")
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
    print("System Information Hook")
    print("=======================\n")
    
    get_os_info()
    get_cpu_info()
    get_memory_info()
    get_disk_info()
    get_network_info()

if __name__ == "__main__":
    main()
