#!/usr/bin/env python3
"""
Autotune script for 3D Maxwell FDTD kernels.

This script tests different block configurations and finds the optimal
settings for a given grid size.

Usage:
    python autotune_maxwell3d.py --nz 256 --ny 256 --nx 256
"""

import argparse
import os
import subprocess
import sys
from typing import List, Tuple


def run_benchmark(nz: int, ny: int, nx: int, nt: int, stencil: int,
                  block_x: int, block_y: int) -> Tuple[float, float]:
    """Run benchmark with given configuration and return (time_per_step_ms, cell_updates_per_s)."""
    env = os.environ.copy()
    env["TIDE_BLOCK_X"] = str(block_x)
    env["TIDE_BLOCK_Y"] = str(block_y)

    cmd = [
        sys.executable,
        "examples/benchmark_maxwell3d.py",
        f"--nz={nz}", f"--ny={ny}", f"--nx={nx}",
        f"--nt={nt}", f"--stencil={stencil}",
        "--warmup=2", "--iters=3"
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, env=env
        )
        output = result.stdout

        time_per_step = None
        cell_updates = None

        for line in output.split('\n'):
            if 'Time per step:' in line:
                time_per_step = float(line.split()[-2])
            if 'Cell updates/s:' in line:
                cell_updates = float(line.split()[-1].replace('e+', 'e'))

        if time_per_step and cell_updates:
            return time_per_step, cell_updates
    except Exception as e:
        print(f"Error: {e}")

    return float('inf'), 0.0


def autotune(nz: int, ny: int, nx: int, nt: int, stencil: int) -> None:
    """Find optimal block configuration for given grid size."""
    print("=" * 70)
    print("3D Maxwell FDTD Autotuning")
    print("=" * 70)
    print(f"Grid: {nz} x {ny} x {nx}")
    print(f"Time steps: {nt}")
    print(f"Stencil order: {stencil}")
    print("=" * 70)
    print()

    # Block configurations to test
    configs: List[Tuple[int, int]] = [
        (16, 8), (16, 16),
        (32, 4), (32, 8), (32, 16),
        (64, 4), (64, 8),
    ]

    # Filter valid configurations (128 <= threads <= 1024)
    valid_configs = [(bx, by) for bx, by in configs if 128 <= bx * by <= 1024]

    results = []
    print("Testing configurations...")
    print("-" * 70)
    print(f"{'Block':>12} {'Threads':>8} {'Time/Step':>12} {'Cell Updates/s':>15}")
    print("-" * 70)

    for block_x, block_y in valid_configs:
        threads = block_x * block_y
        time_ms, cells_per_s = run_benchmark(nz, ny, nx, nt, stencil, block_x, block_y)

        if time_ms < float('inf'):
            results.append((block_x, block_y, time_ms, cells_per_s))
            print(f"{block_x}x{block_y:>6} {threads:>8} {time_ms:>10.4f} ms {cells_per_s:>15.2e}")
        else:
            print(f"{block_x}x{block_y:>6} {threads:>8} {'FAILED':>12}")

    print("-" * 70)

    if results:
        # Sort by cell updates/s (higher is better)
        results.sort(key=lambda x: x[3], reverse=True)
        best = results[0]

        print()
        print("=" * 70)
        print("OPTIMAL CONFIGURATION")
        print("=" * 70)
        print(f"Block size: {best[0]} x {best[1]} ({best[0] * best[1]} threads)")
        print(f"Time per step: {best[2]:.4f} ms")
        print(f"Cell updates/s: {best[3]:.2e}")
        print()
        print("To use this configuration:")
        print(f"  export TIDE_BLOCK_X={best[0]}")
        print(f"  export TIDE_BLOCK_Y={best[1]}")
        print("=" * 70)

        # Show top 3 configurations
        if len(results) > 1:
            print()
            print("Alternative configurations (sorted by performance):")
            for i, (bx, by, t, c) in enumerate(results[:3], 1):
                print(f"  {i}. {bx}x{by} - {t:.4f} ms - {c:.2e} cell/s")
    else:
        print("No valid results obtained.")


def main():
    parser = argparse.ArgumentParser(description="Autotune 3D Maxwell FDTD kernels")
    parser.add_argument("--nz", type=int, default=128, help="Grid size in z")
    parser.add_argument("--ny", type=int, default=128, help="Grid size in y")
    parser.add_argument("--nx", type=int, default=128, help="Grid size in x")
    parser.add_argument("--nt", type=int, default=30, help="Time steps for testing")
    parser.add_argument("--stencil", type=int, default=2, choices=[2, 4, 6, 8])
    args = parser.parse_args()

    autotune(args.nz, args.ny, args.nx, args.nt, args.stencil)


if __name__ == "__main__":
    main()
