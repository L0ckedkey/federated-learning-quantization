import subprocess
import csv
import re
from datetime import datetime

csv_file = "tegrastats_log.csv"
header = ["timestamp", "RAM_used_MB", "RAM_total_MB", "SWAP_used_MB", "CPU_percentages", "GPU_percent", "GPU_clock_MHz", "VDD_IN_mW", "VDD_CPU_GPU_CV_mW", "VDD_SOC_mW"]

def parse_tegrastats_line(line):
    data = {}

    ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
    swap_match = re.search(r'SWAP (\d+)/(\d+)MB', line)
    if ram_match:
        data["RAM_used_MB"] = int(ram_match.group(1))
        data["RAM_total_MB"] = int(ram_match.group(2))
    else:
        data["RAM_used_MB"] = data["RAM_total_MB"] = None
    if swap_match:
        data["SWAP_used_MB"] = int(swap_match.group(1))
    else:
        data["SWAP_used_MB"] = None

    cpu_match = re.search(r'CPU \[([^\]]+)\]', line)
    if cpu_match:
        cpu_vals = cpu_match.group(1).split(",")
        cpu_percent = [int(v.split("%")[0]) for v in cpu_vals]
        data["CPU_percentages"] = cpu_percent
    else:
        data["CPU_percentages"] = []

    gpu_match = re.search(r'GR3D_FREQ (\d+)%@\[(\d+)\]', line)
    if gpu_match:
        data["GPU_percent"] = int(gpu_match.group(1))
        data["GPU_clock_MHz"] = int(gpu_match.group(2))
    else:
        data["GPU_percent"] = data["GPU_clock_MHz"] = None
        
    vdd_in_match = re.search(r'VDD_IN (\d+)mW', line)
    vdd_cpu_gpu_match = re.search(r'VDD_CPU_GPU_CV (\d+)mW', line)
    vdd_soc_match = re.search(r'VDD_SOC (\d+)mW', line)

    data["VDD_IN_mW"] = int(vdd_in_match.group(1)) if vdd_in_match else None
    data["VDD_CPU_GPU_CV_mW"] = int(vdd_cpu_gpu_match.group(1)) if vdd_cpu_gpu_match else None
    data["VDD_SOC_mW"] = int(vdd_soc_match.group(1)) if vdd_soc_match else None

    return data

with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()

with subprocess.Popen(["tegrastats"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) as proc:
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            parsed = parse_tegrastats_line(line)
            parsed["timestamp"] = datetime.now().isoformat()
            
            with open(csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writerow(parsed)

            print(parsed)

    except KeyboardInterrupt:
        proc.terminate()
        print("\nStopped logging.")
