#!/usr/bin/env python3
"""
Benchmark 2D Maxwell TM FDTD propagation (CUDA).

This script measures the performance of the 2D Maxwell propagator
and outputs detailed timing information for profiling.

Usage:
    python benchmark_maxwell2d.py [options]

Examples:
    # Basic benchmark
    python benchmark_maxwell2d.py --ny 512 --nx 512 --nt 200

    # Large grid benchmark
    python benchmark_maxwell2d.py --ny 2048 --nx 2048 --nt 500
"""

import argparse
import time
import sys

import torch

try:
    import tide
except ImportError:
    sys.path.insert(0, ".")
    import tide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark 2D Maxwell TM FDTD forward propagation."
    )
    parser.add_argument("--ny", type=int, default=512, help="Grid size in y")
    parser.add_argument("--nx", type=int, default=512, help="Grid size in x")
    parser.add_argument("--nt", type=int, default=200, help="Number of time steps")
    parser.add_argument("--pml-width", type=int, default=20, help="PML width")
    parser.add_argument("--dx", type=float, default=0.01, help="Grid spacing (m)")
    parser.add_argument("--dt", type=float, default=1e-11, help="Time step (s)")
    parser.add_argument("--shots", type=int, default=1, help="Number of shots")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--stencil", type=int, default=2, choices=[2, 4, 6, 8],
                        help="Finite difference stencil order")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float64"], help="Data type")
    parser.add_argument("--gradient-mode", type=str, default="snapshot",
                        choices=["snapshot", "boundary"], help="Gradient mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type != "cuda":
        print("WARNING: CUDA not available, benchmark will be much slower on CPU")

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    ny, nx = args.ny, args.nx
    nt = args.nt
    n_shots = args.shots
    n_sources = 1
    n_receivers = 1

    # Material parameters (2D: [ny, nx])
    epsilon = torch.ones((ny, nx), device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    # Source wavelet
    wavelet = tide.ricker(
        freq=1e9,
        length=nt,
        dt=args.dt,
        dtype=dtype,
        device=device,
    )
    source_amplitude = wavelet.reshape(1, 1, -1).repeat(n_shots, n_sources, 1)

    # Source/receiver locations (2D: [y, x])
    src_y, src_x = ny // 2, nx // 2
    rec_y, rec_x = ny // 2, min(nx - 1, nx // 2 + nx // 4)

    source_location = torch.tensor(
        [[[src_y, src_x]]], device=device
    ).repeat(n_shots, 1, 1)
    receiver_location = torch.tensor(
        [[[rec_y, rec_x]]], device=device
    ).repeat(n_shots, 1, 1)

    pml_width = args.pml_width
    grid_spacing = [args.dx, args.dx]

    # Memory estimation (2D: 3 field components Ey, Hx, Hz + PML)
    cells = ny * nx
    bytes_per_element = 4 if dtype == torch.float32 else 8
    n_field_arrays = 9  # 3 fields + 6 PML memories
    memory_mb = (cells * n_shots * n_field_arrays * bytes_per_element) / (1024**2)

    print("=" * 60)
    print("2D Maxwell TM FDTD Benchmark")
    print("=" * 60)
    print(f"Grid size: {ny} x {nx} = {cells:,} cells")
    print(f"Time steps: {nt}")
    print(f"Shots: {n_shots}")
    print(f"Stencil order: {args.stencil}")
    print(f"PML width: {pml_width}")
    print(f"Grid spacing: {args.dx} m")
    print(f"Time step: {args.dt} s")
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Gradient mode: {args.gradient_mode}")
    print(f"Estimated field memory: {memory_mb:.2f} MB")
    print("=" * 60)

    # Create model (2D TM mode)
    model = tide.MaxwellTM(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        grid_spacing=grid_spacing,
    )

    def run_once():
        result = model(
            dt=args.dt,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=pml_width,
            stencil=args.stencil,
            gradient_mode=args.gradient_mode,
        )
        return result

    # Warmup
    print(f"\nWarming up ({args.warmup} iterations)...")
    for i in range(args.warmup):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"Running benchmark ({args.iters} iterations)...")
    times = []

    for i in range(args.iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
            start = time.perf_counter()
            run_once()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        else:
            start = time.perf_counter()
            run_once()
            elapsed = time.perf_counter() - start

        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.4f} s")

    # Statistics
    times_array = torch.tensor(times)
    mean_time = times_array.mean().item()
    std_time = times_array.std().item()
    min_time = times_array.min().item()
    max_time = times_array.max().item()

    # Performance metrics
    total_cells = cells * nt * n_shots
    cell_updates_per_s = total_cells / mean_time
    time_per_step_ms = (mean_time / nt) * 1e3

    # Memory bandwidth estimation (2D: 3 field arrays read/write)
    bytes_per_step = cells * n_shots * 3 * 2 * bytes_per_element  # 2x for read+write
    bandwidth_gb_s = (bytes_per_step * nt) / mean_time / (1024**3)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total time (mean): {mean_time:.4f} s")
    print(f"Total time (std):  {std_time:.4f} s")
    print(f"Total time (min):  {min_time:.4f} s")
    print(f"Total time (max):  {max_time:.4f} s")
    print(f"Time per step:     {time_per_step_ms:.4f} ms")
    print(f"Cell updates/s:    {cell_updates_per_s:.2e}")
    print(f"Effective bandwidth: {bandwidth_gb_s:.1f} GB/s (estimated)")
    print("=" * 60)

    # Additional GPU info
    if device.type == "cuda":
        print("\nGPU Information:")
        print(f"  Name: {torch.cuda.get_device_name(device)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"  Memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    # Output CSV-friendly summary
    print("\n# CSV Summary (copy-paste friendly):")
    print(f"# ny,nx,nt,shots,stencil,dtype,mean_s,std_s,cells_per_s,bandwidth_gb_s")
    print(f"{ny},{nx},{nt},{n_shots},{args.stencil},{args.dtype},"
          f"{mean_time:.4f},{std_time:.4f},{cell_updates_per_s:.2e},{bandwidth_gb_s:.1f}")


if __name__ == "__main__":
    main()
