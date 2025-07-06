#!/usr/bin/env python3
"""
benchmark.py

Measure and compare CPU/GPU/RAM/Disk performance on this machine.
Prints out a labeled table of results.

Last tried with python 3.13.4 on Windows 11
"""




import os
import time
import platform
import tempfile
import subprocess
import multiprocessing as mp
import datetime
import csv


import numpy as np
import psutil
import cpuinfo
import GPUtil
import torch

# Do we have a CUDA-capable GPU?
HAS_CUDA = torch.cuda.is_available()

# CSV output configuration
OUTPUT_DIR = "benchmark_results"
CSV_FILE = os.path.join(OUTPUT_DIR, "performance_benchmarks.csv")


def get_system_info():
    """Detect CPU, GPU, and Motherboard info."""
    info = {}
    # CPU
    cpu = cpuinfo.get_cpu_info()
    info['CPU'] = cpu.get('brand_raw', platform.processor())
    info['Cores (logical)'] = psutil.cpu_count(logical=True)
    info['Cores (physical)'] = psutil.cpu_count(logical=False)
    # GPU
    gpus = GPUtil.getGPUs()
    if gpus:
        info['GPU'] = ", ".join(f"{gpu.name} ({gpu.memoryTotal} MB)" for gpu in gpus)
    else:
        info['GPU'] = "None detected"
    # Motherboard
    if platform.system() == 'Windows':
        try:
            import wmi
            c = wmi.WMI()
            mb = c.Win32_BaseBoard()[0]
            info['Motherboard'] = f"{mb.Manufacturer} {mb.Product}"
        except Exception:
            info['Motherboard'] = "Unknown (WMI failed)"
    else:
        # Linux / Mac: use dmidecode if available
        try:
            out = subprocess.check_output(['dmidecode', '-t', 'baseboard'],
                                          stderr=subprocess.DEVNULL,
                                          universal_newlines=True)
            # crude parse: Manufacturer and Product Name lines
            lines = out.splitlines()
            mfg = next((l.split(":",1)[1].strip() for l in lines if "Manufacturer:" in l), "")
            prod= next((l.split(":",1)[1].strip() for l in lines if "Product Name:" in l), "")
            info['Motherboard'] = f"{mfg} {prod}".strip()
        except Exception:
            info['Motherboard'] = "Unknown (dmidecode failed/not root)"
    return info


def benchmark_cpu_single(n=2000, iters=5):
    """Single‐threaded matrix‐multiply FLOPS."""
    a = np.random.rand(n, n).astype(np.float64)
    b = np.random.rand(n, n).astype(np.float64)
    # warm
    _ = a.dot(b)
    start = time.time()
    for _ in range(iters):
        a.dot(b)
    elapsed = time.time() - start
    # flops ≈ 2 * n^3 per multiply
    total_flops = 2 * (n**3) * iters
    return total_flops / elapsed / 1e9  # GFLOPS


def _cpu_worker(n, iters, queue):
    """Helper for multi‐proc."""
    gflops = benchmark_cpu_single(n, iters)
    queue.put(gflops)


def benchmark_cpu_multi(n=2000, iters=5):
    """Multi‐process CPU FLOPS across all logical cores."""
    procs = []
    queue = mp.Queue()
    cores = mp.cpu_count()
    for _ in range(cores):
        p = mp.Process(target=_cpu_worker, args=(n, iters, queue))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    # sum GFLOPS from each process
    return sum(queue.get() for _ in procs)


def benchmark_gpu_single(n=4096, iters=20):
    """Single CUDA stream matrix FLOPS via PyTorch."""
    if not HAS_CUDA:
        return None
    torch.cuda.synchronize()
    a = torch.randn(n, n, device='cuda', dtype=torch.float32)
    b = torch.randn(n, n, device='cuda', dtype=torch.float32)
    # warm
    _ = a @ b
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    elapsed = time.time() - start
    total_flops = 2 * (n**3) * iters
    return total_flops / elapsed / 1e9  # GFLOPS


def benchmark_gpu_multi(n=4096, iters=10, streams=4):
    """Multi‐stream GPU FLOPS."""
    if not HAS_CUDA:
        return None
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream() for _ in range(streams)]
    a = torch.randn(n, n, device='cuda', dtype=torch.float32)
    b = torch.randn(n, n, device='cuda', dtype=torch.float32)
    # warm
    with torch.cuda.stream(streams[0]):
        _ = a @ b
    torch.cuda.synchronize()
    start = time.time()
    for s in streams:
        with torch.cuda.stream(s):
            _ = a @ b
    torch.cuda.synchronize()
    elapsed = time.time() - start
    total_flops = 2 * (n**3) * iters * len(streams)
    return total_flops / elapsed / 1e9  # GFLOPS


def benchmark_ram_sequential(size_mb=500):
    """Sequential write + read bandwidth (MB/s)."""
    size = size_mb * 1024 * 1024 // 8
    arr = np.empty(size, dtype=np.float64)
    # write
    start = time.time()
    arr.fill(1.23)
    write_time = time.time() - start
    # read
    start = time.time()
    s = arr.sum()
    read_time = time.time() - start
    return size_mb / write_time, size_mb / read_time


def benchmark_ram_random(size_mb=500, accesses=10_000_000):
    """Random write + read bandwidth (MB/s)."""
    size = size_mb * 1024 * 1024 // 8
    arr = np.zeros(size, dtype=np.float64)
    idx = np.random.randint(0, size, size=accesses)
    # write
    start = time.time()
    arr[idx] = 3.21
    write_time = time.time() - start
    # read
    start = time.time()
    tmp = arr[idx]
    read_time = time.time() - start
    mb_moved = accesses * arr.itemsize / (1024*1024)
    return mb_moved / write_time, mb_moved / read_time


def benchmark_disk_io(file_size_mb=500):
    """Disk write + read throughput (MB/s)."""
    path = os.path.join(tempfile.gettempdir(), 'bench_io.tmp')
    data = b'\0' * (1024*1024)
    # write
    with open(path, 'wb') as f:
        start = time.time()
        for _ in range(file_size_mb):
            f.write(data)
        write_time = time.time() - start
    # read
    with open(path, 'rb') as f:
        start = time.time()
        while f.read(1024*1024):
            pass
        read_time = time.time() - start
    os.remove(path)
    return file_size_mb / write_time, file_size_mb / read_time


def write_csv_row(data: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(data.keys()))
        if write_header: writer.writeheader()
        writer.writerow(data)


def report(results, key, label, value, fmt=None, skip_if_none=False, unit=None):
    """
    Print a label and value, and record into results dict.
    If skip_if_none is True and value is None, do nothing.
    fmt should be a Python format string, e.g. '{:.1f}'.
    unit should be a string like 'GFLOPS', 'MB/s', 'ms', etc.
    """
    if skip_if_none and value is None:
        return
    if fmt:
        val_str = fmt.format(value)
    else:
        val_str = str(value)
    
    # Add unit to the printed output if provided
    if unit:
        print(f"{label}: {val_str} {unit}")
    else:
        print(f"{label}: {val_str}")
    
    results[key] = val_str


def main():
    sys = get_system_info()
    results = {}
    results['timestamp'] = datetime.datetime.now().isoformat()
    results['hostname'] = platform.node()
    for k, v in sys.items(): results[k] = v

    # CPU
    print("Running CPU benchmarks...")
    start_time = time.time()
    cpu_s = benchmark_cpu_single()
    cpu_s_time = time.time() - start_time
    
    start_time = time.time()
    cpu_m = benchmark_cpu_multi()
    cpu_m_time = time.time() - start_time
    
    report(results, 'cpu_single_gflops', 'CPU single-thread GFLOPS', cpu_s, fmt='{:.1f}', unit='GFLOPS')
    report(results, 'cpu_single_time', 'CPU single-thread runtime', cpu_s_time, fmt='{:.2f}', unit='seconds')
    report(results, 'cpu_multi_gflops',  'CPU multi-thread GFLOPS',  cpu_m, fmt='{:.1f}', unit='GFLOPS')
    report(results, 'cpu_multi_time', 'CPU multi-thread runtime', cpu_m_time, fmt='{:.2f}', unit='seconds')

    # GPU
    print("Running GPU benchmarks...")
    start_time = time.time()
    gpu_s = benchmark_gpu_single()
    gpu_s_time = time.time() - start_time
    
    start_time = time.time()
    gpu_m = benchmark_gpu_multi()
    gpu_m_time = time.time() - start_time
    
    if HAS_CUDA:
        report(results, 'gpu_single_gflops', 'GPU single-stream GFLOPS', gpu_s, fmt='{:.1f}', skip_if_none=True, unit='GFLOPS')
        report(results, 'gpu_single_time', 'GPU single-stream runtime', gpu_s_time, fmt='{:.2f}', unit='seconds')
        report(results, 'gpu_multi_gflops',  'GPU multi-stream GFLOPS',  gpu_m, fmt='{:.1f}', skip_if_none=True, unit='GFLOPS')
        report(results, 'gpu_multi_time', 'GPU multi-stream runtime', gpu_m_time, fmt='{:.2f}', unit='seconds')
    else:
        print('GPU benchmarks skipped (no CUDA)')
        results['gpu_single_gflops'] = ''
        results['gpu_single_time'] = ''
        results['gpu_multi_gflops']  = ''
        results['gpu_multi_time'] = ''

    # RAM
    print("Running RAM benchmarks...")
    start_time = time.time()
    seq_w, seq_r = benchmark_ram_sequential()
    ram_seq_time = time.time() - start_time
    
    start_time = time.time()
    rnd_w, rnd_r = benchmark_ram_random()
    ram_rnd_time = time.time() - start_time
    
    report(results, 'ram_seq_write', 'RAM sequential write', seq_w, fmt='{:.0f}', unit='MB/s')
    report(results, 'ram_seq_read',  'RAM sequential read',  seq_r, fmt='{:.0f}', unit='MB/s')
    report(results, 'ram_seq_time', 'RAM sequential runtime', ram_seq_time, fmt='{:.2f}', unit='seconds')
    report(results, 'ram_rnd_write','RAM random write',     rnd_w, fmt='{:.0f}', unit='MB/s')
    report(results, 'ram_rnd_read', 'RAM random read',      rnd_r, fmt='{:.0f}', unit='MB/s')
    report(results, 'ram_rnd_time', 'RAM random runtime', ram_rnd_time, fmt='{:.2f}', unit='seconds')

    # Disk
    print("Running Disk I/O benchmarks...")
    start_time = time.time()
    io_w, io_r = benchmark_disk_io()
    disk_time = time.time() - start_time
    
    report(results, 'disk_write', 'Disk write', io_w, fmt='{:.0f}', unit='MB/s')
    report(results, 'disk_read',  'Disk read',  io_r, fmt='{:.0f}', unit='MB/s')
    report(results, 'disk_time', 'Disk I/O runtime', disk_time, fmt='{:.2f}', unit='seconds')

    write_csv_row(results)
    print(f"\nResults saved to {CSV_FILE}\n")


if __name__ == '__main__':
    main()
