import random
import time
import csv
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from tqdm import tqdm

@dataclass
class Process:
    pid: int
    process_type: str
    cpu_bursts: List[float] = field(default_factory=list)
    memory_usage: float = 0.0
    io_operations: int = 0
    priority: int = 0

@dataclass
class ReadyQueue:
    processes: List[Process] = field(default_factory=list)

    def add_process(self, process: Process):
        self.processes.append(process)

    def remove_process(self):
        if self.processes:
            return self.processes.pop(0)
        return None

def generate_process(pid: int) -> Process:
    process_types = ["CPU-bound", "I/O-bound", "Mixed"]
    return Process(
        pid=pid,
        process_type=random.choice(process_types),
        memory_usage=random.uniform(1, 1000),  # MB
        io_operations=random.randint(0, 100),
        priority=random.randint(1, 10)
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
                      "memory_usage", "io_operations", "system_load", "time_of_day", "priority",
                      "ready_queue_size", "ready_queue_cpu_bound", "ready_queue_io_bound", "ready_queue_mixed",
                      "last_100_decisions"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        system_load = random.uniform(0, 100)
        data_buffer = []
        ready_queue = ReadyQueue()
        last_100_decisions = []
        
        progress_bar = tqdm(total=num_processes * num_bursts, desc="Generating data")
        
        for i in range(num_processes):
            process = generate_process(i)
            ready_queue.add_process(process)

        for _ in range(num_processes * num_bursts):
            system_load = max(0, min(100, system_load + random.uniform(-5, 5)))
            
            process = ready_queue.remove_process()
            if process is None:
                continue

            burst_time = simulate_cpu_burst(process, system_load)
            
            ready_queue_stats = {
                "size": len(ready_queue.processes),
                "cpu_bound": sum(1 for p in ready_queue.processes if p.process_type == "CPU-bound"),
                "io_bound": sum(1 for p in ready_queue.processes if p.process_type == "I/O-bound"),
                "mixed": sum(1 for p in ready_queue.processes if p.process_type == "Mixed")
            }

            last_100_decisions.append(process.process_type)
            if len(last_100_decisions) > 100:
                last_100_decisions.pop(0)

            data_point = {
                "pid": process.pid,
                "process_type": process.process_type,
                "burst_number": len(process.cpu_bursts),
                "cpu_burst_time": burst_time,
                "avg_previous_bursts": np.mean(process.cpu_bursts),
                "memory_usage": process.memory_usage,
                "io_operations": process.io_operations,
                "system_load": system_load,
                "time_of_day": (time.time() % 86400) / 3600,
                "priority": process.priority,
                "ready_queue_size": ready_queue_stats["size"],
                "ready_queue_cpu_bound": ready_queue_stats["cpu_bound"],
                "ready_queue_io_bound": ready_queue_stats["io_bound"],
                "ready_queue_mixed": ready_queue_stats["mixed"],
                "last_100_decisions": ",".join(last_100_decisions)
            }
            data_buffer.append(data_point)
            
            if len(data_buffer) >= update_interval:
                writer.writerows(data_buffer)
                data_buffer = []
                csvfile.flush()
            
            ready_queue.add_process(process)
            progress_bar.update(1)
        
        if data_buffer:
            writer.writerows(data_buffer)
        
        progress_bar.close()

def main():
    num_processes = 1000
    num_bursts_per_process = 50
    csv_file = 'enhanced_process_data.csv'
    
    collect_data(num_processes, num_bursts_per_process, csv_file)
    
    print(f"Data collection complete. Total data points: {num_processes * num_bursts_per_process}")

if __name__ == "__main__":
    main()