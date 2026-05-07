# compute_report.py

import os
import json
import time
import platform
import subprocess
import psutil
from contextlib import contextmanager
from datetime import datetime


def get_gpu_info():
    """Return GPU info if nvidia-smi is available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            name, mem_total, mem_used = [x.strip() for x in line.split(",")]
            gpus.append(
                {
                    "name": name,
                    "memory_total_MB": float(mem_total),
                    "memory_used_MB": float(mem_used),
                }
            )

        return gpus

    except Exception:
        return []


def get_system_info():
    """Return basic machine information."""
    vm = psutil.virtual_memory()

    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_GB": round(vm.total / (1024 ** 3), 2),
        "gpu_info": get_gpu_info(),
    }


@contextmanager
def compute_tracker(experiment_name, output_path="compute_report.jsonl", extra=None):
    """
    Tracks wall-clock time, CPU time, and memory usage for a code block.

    Example:
        with compute_tracker("save_npy_file"):
            np.save("embeddings.npy", embeddings)
    """

    process = psutil.Process(os.getpid())

    start_time = time.time()
    start_cpu = process.cpu_times()
    start_mem = process.memory_info().rss

    system_info = get_system_info()

    try:
        yield

    finally:
        end_time = time.time()
        end_cpu = process.cpu_times()
        end_mem = process.memory_info().rss

        record = {
            "experiment_name": experiment_name,
            "wall_time_seconds": round(end_time - start_time, 3),
            "wall_time_minutes": round((end_time - start_time) / 60, 3),
            "cpu_user_seconds": round(end_cpu.user - start_cpu.user, 3),
            "cpu_system_seconds": round(end_cpu.system - start_cpu.system, 3),
            "rss_memory_start_GB": round(start_mem / (1024 ** 3), 3),
            "rss_memory_end_GB": round(end_mem / (1024 ** 3), 3),
            "rss_memory_change_GB": round((end_mem - start_mem) / (1024 ** 3), 3),
            "system_info": system_info,
            "extra": extra or {},
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"\n[COMPUTE] Saved compute report for: {experiment_name}")
        print(json.dumps(record, indent=2))