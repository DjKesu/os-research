import random
import time
import csv
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

@dataclass
class Process:
    pid: int
    process_type: str
    cpu_bursts: List[float]
    memory_usage: float
    io_operations: int

def generate_process(pid: int) -> Process:
    process_types = ["CPU-bound", "I/O-bound", "Mixed"]
    return Process(
        pid=pid,
        process_type=random.choice(process_types),
        cpu_bursts=[],
        memory_usage=random.uniform(1, 1000),  # MB
        io_operations=random.randint(0, 100)
    )

def simulate_cpu_burst(process: Process, system_load: float) -> float:
    base_time = {
        "CPU-bound": random.uniform(10, 100),
        "I/O-bound": random.uniform(1, 10),
        "Mixed": random.uniform(5, 50)
    }[process.process_type]
    
    load_factor = 1 + (system_load / 100)
    burst_time = base_time * load_factor
    
    if process.cpu_bursts:
        burst_time = 0.7 * burst_time + 0.3 * process.cpu_bursts[-1]
    
    process.cpu_bursts.append(burst_time)
    return burst_time

def collect_data(num_processes: int, num_bursts: int, csv_file: str, update_interval: int = 1000):
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ["pid", "process_type", "burst_number", "cpu_burst_time", "avg_previous_bursts", 
                      "memory_usage", "io_operations", "system_load", "time_of_day"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        system_load = random.uniform(0, 100)
        data_buffer = []
        
        progress_bar = tqdm(total=num_processes * num_bursts, desc="Generating data")
        
        for i in range(num_processes):
            process = generate_process(i)
            for j in range(num_bursts):
                system_load = max(0, min(100, system_load + random.uniform(-5, 5)))
                
                burst_time = simulate_cpu_burst(process, system_load)
                data_point = {
                    "pid": process.pid,
                    "process_type": process.process_type,
                    "burst_number": j + 1,
                    "cpu_burst_time": burst_time,
                    "avg_previous_bursts": sum(process.cpu_bursts) / len(process.cpu_bursts),
                    "memory_usage": process.memory_usage,
                    "io_operations": process.io_operations,
                    "system_load": system_load,
                    "time_of_day": (time.time() % 86400) / 3600
                }
                data_buffer.append(data_point)
                
                if len(data_buffer) >= update_interval:
                    writer.writerows(data_buffer)
                    data_buffer = []
                    csvfile.flush()
                    update_visualizations(csv_file)
                
                progress_bar.update(1)
        
        if data_buffer:
            writer.writerows(data_buffer)
        
        progress_bar.close()

def update_visualizations(csv_file: str):
    df = pd.read_csv(csv_file)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    sns.histplot(data=df, x='cpu_burst_time', hue='process_type', kde=True)
    plt.title('Distribution of CPU Burst Times')
    
    plt.subplot(132)
    sns.scatterplot(data=df, x='system_load', y='cpu_burst_time', hue='process_type', alpha=0.1)
    plt.title('CPU Burst Time vs System Load')
    
    plt.subplot(133)
    sns.barplot(data=df, x='process_type', y='cpu_burst_time')
    plt.title('Average CPU Burst Time by Process Type')
    
    plt.tight_layout()
    plt.savefig('real_time_visualizations.png')
    plt.close()

def main():
    num_processes = 1000
    num_bursts_per_process = 50
    csv_file = 'process_data.csv'
    
    collect_data(num_processes, num_bursts_per_process, csv_file)
    update_visualizations(csv_file)
    
    print(f"Data collection complete. Total data points: {num_processes * num_bursts_per_process}")

if __name__ == "__main__":
    main()