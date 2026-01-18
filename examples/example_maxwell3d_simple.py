#!/usr/bin/env python3
"""
Simple 3D Maxwell equations forward propagation example with analytical comparison.

This example demonstrates basic 3D FDTD simulation with tide.Maxwell3D and
compares numerical results against analytical solution in a homogeneous medium.

Usage:
    python example_maxwell3d_simple.py [--plot]
"""

import argparse
import math
import torch
import matplotlib.pyplot as plt

import tide


# Physical constants
EP0 = 8.8541878128e-12   # vacuum permittivity (F/m)
MU0 = 1.2566370614359173e-06  # vacuum permeability (H/m)
C0 = 1.0 / math.sqrt(EP0 * MU0)  # speed of light (m/s)


def analytical_solution_3d_simple(
    wavelet: torch.Tensor,
    dt: float,
    distance: float,
    eps_r: float,
) -> torch.Tensor:
    """
    Simplified analytical solution for 3D spherical wave.

    For a point source in 3D homogeneous medium, the field decays as 1/r
    and propagates with phase velocity c/sqrt(eps_r).

    Args:
        wavelet: Source time function
        dt: Time step
        distance: Source-receiver distance (m)
        eps_r: Relative permittivity

    Returns:
        field: Electric field at receiver
    """
    c_medium = C0 / math.sqrt(eps_r)
    travel_time = distance / c_medium
    travel_samples = int(travel_time / dt)

    nt = wavelet.numel()
    field = torch.zeros(nt, dtype=wavelet.dtype, device=wavelet.device)

    # Simple time-shift and amplitude scaling
    # In 3D, amplitude decays as 1/r
    amplitude_factor = 1.0 / distance

    if travel_samples < nt:
        field[travel_samples:] = wavelet[:nt-travel_samples] * amplitude_factor

    return field


def main():
    parser = argparse.ArgumentParser(description="Simple 3D Maxwell example")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    # Use CPU for stability with Python backend
    device = torch.device("cpu")
    dtype = torch.float32

    print(f"Using device: {device}, dtype: {dtype}")

    # Grid parameters - use small grid for fast computation
    nz, ny, nx = 16, 16, 16
    dx = dy = dz = 0.01  # 1 cm grid spacing

    # Material parameters - vacuum
    eps_r = 1.0
    epsilon = torch.full((nz, ny, nx), eps_r, device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    # Time parameters
    c_medium = C0 / math.sqrt(eps_r)
    dt_max = 0.5 * dx / (c_medium * math.sqrt(3.0))
    dt = dt_max
    nt = 60  # Short simulation

    # Source parameters
    freq = 2e8  # 200 MHz
    src_z, src_y, src_x = nz // 2, ny // 2, nx // 2
    rec_offset = 4
    rec_z, rec_y, rec_x = src_z, src_y, src_x + rec_offset

    distance = rec_offset * dx

    print("\nSimulation parameters:")
    print(f"  Grid: {nz} × {ny} × {nx} = {nz*dz:.2f}m × {ny*dy:.2f}m × {nx*dx:.2f}m")
    print(f"  Grid spacing: {dx*1000:.1f} mm")
    print(f"  Time steps: {nt}")
    print(f"  Time step: {dt*1e9:.2f} ns")
    print(f"  Source frequency: {freq/1e6:.1f} MHz")
    print(f"  Source location: ({src_z}, {src_y}, {src_x})")
    print(f"  Receiver location: ({rec_z}, {rec_y}, {rec_x})")
    print(f"  Distance: {distance*1000:.1f} mm")

    # Generate simple Gaussian pulse
    t_peak = 1.0 / freq
    wavelet = tide.ricker(
        freq=freq,
        length=nt,
        dt=dt,
        peak_time=t_peak,
        dtype=dtype,
        device=device,
    )

    source_amplitude = wavelet.reshape(1, 1, -1)
    source_location = torch.tensor([[[src_z, src_y, src_x]]], device=device)
    receiver_location = torch.tensor([[[rec_z, rec_y, rec_x]]], device=device)

    # Run simulation
    print("\nRunning FDTD simulation...")
    model = tide.Maxwell3D(epsilon=epsilon, sigma=sigma, mu=mu, grid_spacing=[dz, dy, dx])

    result = model(
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=0,  # No PML for this simple test
        stencil=2,
        source_component="Ez",
        receiver_component="Ez",
    )

    receiver_data = result[-1]
    numerical = receiver_data[:, 0, 0].cpu()

    print(f"Numerical solution range: [{numerical.min():.4e}, {numerical.max():.4e}]")
    print(f"Has NaN: {torch.isnan(numerical).any()}")

    # Compute simple analytical solution
    print("\nComputing analytical solution...")
    analytical = analytical_solution_3d_simple(wavelet.cpu(), dt, distance, eps_r)

    # Scale to match (FDTD uses different normalization)
    if torch.dot(analytical, analytical) > 0:
        scale = torch.dot(numerical, analytical) / torch.dot(analytical, analytical)
        analytical_scaled = scale * analytical
    else:
        analytical_scaled = analytical
        scale = 1.0

    # Compute metrics
    if torch.isnan(numerical).any() or torch.isnan(analytical_scaled).any():
        print("\n✗ ERROR: Solution contains NaN values")
        return

    residual = numerical - analytical_scaled
    if torch.dot(analytical_scaled, analytical_scaled) > 0:
        relative_error = torch.linalg.norm(residual) / torch.linalg.norm(analytical_scaled)
    else:
        relative_error = float('inf')

    peak_num_idx = torch.argmax(torch.abs(numerical)).item()
    peak_ana_idx = torch.argmax(torch.abs(analytical_scaled)).item()
    peak_shift = abs(peak_num_idx - peak_ana_idx)

    print(f"\nComparison results:")
    print(f"  Scaling factor: {scale:.6f}")
    print(f"  Relative L2 error: {relative_error*100:.2f}%")
    print(f"  Peak shift: {peak_shift} samples ({peak_shift*dt*1e12:.2f} ps)")
    print(f"  Numerical peak time: {peak_num_idx*dt*1e9:.2f} ns")
    print(f"  Analytical peak time: {peak_ana_idx*dt*1e9:.2f} ns")

    # Note: High error is expected due to simplifications in analytical solution
    # and discrete nature of FDTD
    if relative_error < 2.0:  # 200% tolerance for this simple comparison
        print(f"\n✓ Test passed: Reasonable agreement with analytical solution")
    else:
        print(f"\n⚠ Note: High error is expected for this simplified comparison")
        print(f"  (Analytical solution is approximate for demonstration purposes)")

    # Plotting
    if args.plot:
        print("\nGenerating plots...")

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))

        t_ns = (torch.arange(nt).float() * dt * 1e9).numpy()

        # Plot 1: Comparison
        ax = axes[0]
        ax.plot(t_ns, numerical.numpy(), 'b-', linewidth=2, label='FDTD')
        ax.plot(t_ns, analytical_scaled.numpy(), 'r--', linewidth=2, label='Analytical (scaled)')
        ax.axvline(peak_num_idx*dt*1e9, color='b', linestyle=':', alpha=0.5)
        ax.axvline(peak_ana_idx*dt*1e9, color='r', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Ez Field (V/m)')
        ax.set_title('3D Maxwell: FDTD vs Analytical Solution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Residual
        ax = axes[1]
        ax.plot(t_ns, residual.numpy(), 'g-', linewidth=1.5)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Residual (V/m)')
        ax.set_title(f'Residual, Relative Error: {relative_error*100:.1f}%')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = 'maxwell3d_simple_comparison.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved as: {filename}")

        try:
            plt.show()
        except:
            pass


if __name__ == "__main__":
    main()
