#!/usr/bin/env python3
"""
Detailed profiling for 3D Maxwell CUDA kernels using PyTorch profiler.
"""

import argparse
import torch
from torch.profiler import profile, record_function, ProfilerActivity

try:
    import tide
except ImportError:
    import sys
    sys.path.insert(0, ".")
    import tide


def main():
    parser = argparse.ArgumentParser(description="Profile 3D Maxwell kernels")
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--nt", type=int, default=50)
    parser.add_argument("--stencil", type=int, default=2)
    parser.add_argument("--pml-width", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float32

    nz, ny, nx = args.nz, args.ny, args.nx
    nt = args.nt

    # Setup
    epsilon = torch.ones((nz, ny, nx), device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    wavelet = tide.ricker(freq=1e9, length=nt, dt=1e-11, dtype=dtype, device=device)
    source_amplitude = wavelet.reshape(1, 1, -1)

    src_z, src_y, src_x = nz // 2, ny // 2, nx // 2
    source_location = torch.tensor([[[src_z, src_y, src_x]]], device=device)
    receiver_location = torch.tensor([[[src_z, src_y, src_x + 10]]], device=device)

    model = tide.Maxwell3D(
        epsilon=epsilon, sigma=sigma, mu=mu,
        grid_spacing=[0.01, 0.01, 0.01]
    )

    # Warmup
    print("Warming up...")
    for _ in range(2):
        model(
            dt=1e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=args.pml_width,
            stencil=args.stencil,
            gradient_mode="snapshot",
        )
    torch.cuda.synchronize()

    # Profile
    print("\nProfiling with PyTorch profiler...")
    print(f"Grid: {nz}x{ny}x{nx}, {nt} time steps, stencil order: {args.stencil}")
    print("-" * 60)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("maxwell3d_forward"):
            model(
                dt=1e-11,
                source_amplitude=source_amplitude,
                source_location=source_location,
                receiver_location=receiver_location,
                pml_width=args.pml_width,
                stencil=args.stencil,
                gradient_mode="snapshot",
            )
        torch.cuda.synchronize()

    # Print results
    print("\n=== CUDA Kernel Summary (sorted by CUDA time) ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
        max_name_column_width=60
    ))

    print("\n=== Memory Summary ===")
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage",
        row_limit=15,
        max_name_column_width=60
    ))

    # Export trace for visualization
    trace_file = "maxwell3d_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nTrace exported to {trace_file}")
    print("View in Chrome: chrome://tracing")


if __name__ == "__main__":
    main()
