import os
import platform
import psutil
import socket
import tkinter as tk
from tkinter import ttk
from datetime import timedelta

def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def get_os_info():
    os_info = (
        f"System: {platform.system()}\n"
        f"Node Name: {platform.node()}\n"
        f"Release: {platform.release()}\n"
        f"Version: {platform.version()}\n"
        f"Machine: {platform.machine()}\n"
        f"Processor: {platform.processor()}\n"
    )
    return os_info

def get_cpu_info():
    cpu_info = (
        f"Physical cores: {psutil.cpu_count(logical=False)}\n"
        f"Total cores: {psutil.cpu_count(logical=True)}\n"
        f"CPU usage: {psutil.cpu_percent(interval=1)}%\n"
    )
    return cpu_info

def get_memory_info():
    mem = psutil.virtual_memory()
    memory_info = (
        f"Total: {get_size(mem.total)}\n"
        f"Available: {get_size(mem.available)}\n"
        f"Used: {get_size(mem.used)}\n"
        f"Percentage: {mem.percent}%\n"
    )
    return memory_info

def get_disk_info():
    disk_info = ""
    partition_info = psutil.disk_partitions()
    for partition in partition_info:
        disk_info += (
            f"Device: {partition.device}\n"
            f"  Mountpoint: {partition.mountpoint}\n"
            f"  File system type: {partition.fstype}\n"
        )
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
            disk_info += (
                f"  Total Size: {get_size(partition_usage.total)}\n"
                f"  Used: {get_size(partition_usage.used)}\n"
                f"  Free: {get_size(partition_usage.free)}\n"
                f"  Percentage: {partition_usage.percent}%\n\n"
            )
        except PermissionError:
            disk_info += "  Access Denied\n\n"
    return disk_info

def get_network_info():
    network_info = ""
    network_info += f"Hostname: {socket.gethostname()}\n"
    try:
        network_info += f"IP Address: {socket.gethostbyname(socket.gethostname())}\n"
    except socket.error:
        network_info += "IP Address: Unable to retrieve\n"
    
    net_io = psutil.net_io_counters()
    network_info += (
        f"Total Bytes Sent: {get_size(net_io.bytes_sent)}\n"
        f"Total Bytes Received: {get_size(net_io.bytes_recv)}\n"
    )
    
    interfaces = psutil.net_if_addrs()
    for interface_name, interface_addresses in interfaces.items():
        for address in interface_addresses:
            network_info += f"Interface: {interface_name}\n"
            if str(address.family) == 'AddressFamily.AF_INET':
                network_info += (
                    f"  IP Address: {address.address}\n"
                    f"  Netmask: {address.netmask}\n"
                    f"  Broadcast IP: {address.broadcast}\n"
                )
            elif str(address.family) == 'AddressFamily.AF_PACKET':
                network_info += (
                    f"  MAC Address: {address.address}\n"
                    f"  Netmask: {address.netmask}\n"
                    f"  Broadcast MAC: {address.broadcast}\n"
                )
        network_info += "\n"
    return network_info

def get_process_info():
    process_info = "Top 5 Processes by CPU and Memory Usage:\n\n"
    
    processes = sorted(psutil.process_iter(attrs=['pid', 'name', 'cpu_percent', 'memory_info']),
                       key=lambda p: (p.info['cpu_percent'], p.info['memory_info'].rss), reverse=True)
    
    for proc in processes[:5]:
        try:
            process_info += (
                f"PID: {proc.info['pid']}, Name: {proc.info['name']}\n"
                f"  CPU Usage: {proc.info['cpu_percent']}%\n"
                f"  Memory Usage: {get_size(proc.info['memory_info'].rss)}\n\n"
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return process_info

def get_battery_info():
    battery_info = "Battery Information:\n"
    if psutil.sensors_battery():
        battery = psutil.sensors_battery()
        battery_info += (
            f"Battery Percentage: {battery.percent}%\n"
            f"Power Plugged In: {battery.power_plugged}\n"
            f"Time Left: {str(timedelta(seconds=battery.secsleft)) if not battery.power_plugged else 'N/A'}\n"
        )
    else:
        battery_info += "No battery information available.\n"
    return battery_info

def get_uptime_info():
    uptime_seconds = timedelta(seconds=int(psutil.boot_time()))
    uptime = timedelta(seconds=int((psutil.time.time() - psutil.boot_time())))
    uptime_info = f"System Uptime: {uptime}\n"
    return uptime_info

def get_temperature_info():
    temperature_info = "Temperature Sensors:\n"
    if hasattr(psutil, "sensors_temperatures"):
        temps = psutil.sensors_temperatures()
        if not temps:
            temperature_info += "No temperature sensors available.\n"
        for name, entries in temps.items():
            temperature_info += f"{name}:\n"
            for entry in entries:
                temperature_info += (
                    f"  {entry.label or name} - "
                    f"Current: {entry.current}°C, "
                    f"High: {entry.high}°C, "
                    f"Critical: {entry.critical}°C\n"
                )
    else:
        temperature_info += "Temperature sensors not supported on this platform.\n"
    return temperature_info

def refresh_info():
    os_label.config(text=get_os_info())
    cpu_label.config(text=get_cpu_info())
    memory_label.config(text=get_memory_info())
    disk_label.config(text=get_disk_info())
    network_label.config(text=get_network_info())
    process_label.config(text=get_process_info())
    battery_label.config(text=get_battery_info())
    uptime_label.config(text=get_uptime_info())
    temperature_label.config(text=get_temperature_info())

# Setting up the GUI
root = tk.Tk()
root.title("System Information")

# Adding a notebook (tabbed interface)
notebook = ttk.Notebook(root)
notebook.pack(pady=10, expand=True)

# Creating frames for each tab
os_frame = ttk.Frame(notebook, width=700, height=400)
cpu_frame = ttk.Frame(notebook, width=700, height=400)
memory_frame = ttk.Frame(notebook, width=700, height=400)
disk_frame = ttk.Frame(notebook, width=700, height=400)
network_frame = ttk.Frame(notebook, width=700, height=400)
process_frame = ttk.Frame(notebook, width=700, height=400)
battery_frame = ttk.Frame(notebook, width=700, height=400)
uptime_frame = ttk.Frame(notebook, width=700, height=400)
temperature_frame = ttk.Frame(notebook, width=700, height=400)

# Adding frames to the notebook
notebook.add(os_frame, text="OS Info")
notebook.add(cpu_frame, text="CPU Info")
notebook.add(memory_frame, text="Memory Info")
notebook.add(disk_frame, text="Disk Info")
notebook.add(network_frame, text="Network Info")
notebook.add(process_frame, text="Processes")
notebook.add(battery_frame, text="Battery Info")
notebook.add(uptime_frame, text="Uptime Info")
notebook.add(temperature_frame, text="Temperature")

# Adding content to each frame
os_label = ttk.Label(os_frame, text=get_os_info(), anchor='w', justify='left')
os_label.pack(pady=10, padx=10)

cpu_label = ttk.Label(cpu_frame, text=get_cpu_info(), anchor='w', justify='left')
cpu_label.pack(pady=10, padx=10)

memory_label = ttk.Label(memory_frame, text=get_memory_info(), anchor='w', justify='left')
memory_label.pack(pady=10, padx=10)

disk_label = ttk.Label(disk_frame, text=get_disk_info(), anchor='w', justify='left')
disk_label.pack(pady=10, padx=10)

network_label = ttk.Label(network_frame, text=get_network_info(), anchor='w', justify='left')
network_label.pack(pady=10, padx=10)

process_label = ttk.Label(process_frame, text=get_process_info(), anchor='w', justify='left')
process_label.pack(pady=10, padx=10)

battery_label = ttk.Label(battery_frame, text=get_battery_info(), anchor='w', justify='left')
battery_label.pack(pady=10, padx=10)

uptime_label = ttk.Label(uptime_frame, text=get_uptime_info(), anchor='w', justify='left')
uptime_label.pack(pady=10, padx=10)

temperature_label = ttk.Label(temperature_frame, text=get_temperature_info(), anchor='w', justify='left')
temperature_label.pack(pady=10, padx=10)

# Adding a refresh button
refresh_button = ttk.Button(root, text="Refresh", command=refresh_info)
refresh_button.pack(pady=20)

# Run the GUI
refresh_info()  # Initial load of information
root.mainloop()
