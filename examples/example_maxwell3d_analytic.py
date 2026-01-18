#!/usr/bin/env python3
"""
3D Maxwell equations forward propagation with analytical solution comparison.

This example demonstrates:
1. Setting up a 3D homogeneous medium
2. Running FDTD simulation using tide.Maxwell3D
3. Computing the analytical solution using the 3D Green's function
4. Comparing numerical and analytical solutions

The analytical solution for a point source in a 3D homogeneous medium is:
E(r,t) = I * Z0 * G(r,t), where G is the 3D Green's function (spherical wave).

Reference:
- Taflove & Hagness, "Computational Electrodynamics: The Finite-Difference
  Time-Domain Method", 3rd ed., Chapter 6.
"""

import argparse
import math
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tide


# Physical constants
EP0 = 8.8541878128e-12   # vacuum permittivity (F/m)
MU0 = 1.2566370614359173e-06  # vacuum permeability (H/m)
C0 = 1.0 / math.sqrt(EP0 * MU0)  # speed of light (m/s)
Z0 = math.sqrt(MU0 / EP0)  # impedance of free space (Ohm)


def analytical_solution_3d(
    wavelet: torch.Tensor,
    dt: float,
    src_pos: tuple[float, float, float],
    rec_pos: tuple[float, float, float],
    eps_r: float,
    sigma: float = 0.0,
    mu_r: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute analytical solution for a point source in 3D homogeneous medium.

    Uses frequency-domain Green's function approach with spherical wave propagation.

    Args:
        wavelet: Source time function [nt]
        dt: Time step (s)
        src_pos: Source position in meters (z, y, x)
        rec_pos: Receiver position in meters (z, y, x)
        eps_r: Relative permittivity
        sigma: Conductivity (S/m)
        mu_r: Relative permeability

    Returns:
        (time, field): Time array and electric field at receiver
    """
    device = wavelet.device
    dtype = torch.float64
    nt = wavelet.numel()

    # Time array
    t = torch.arange(nt, device=device, dtype=dtype) * dt

    # Source-receiver distance
    r_vec = (
        torch.tensor(rec_pos, device=device, dtype=dtype)
        - torch.tensor(src_pos, device=device, dtype=dtype)
    )
    R = torch.linalg.norm(r_vec) + 1e-12  # Add small epsilon to avoid division by zero

    # Frequency domain analysis
    wavelet_f64 = wavelet.to(dtype)
    spectrum = torch.fft.rfft(wavelet_f64)

    freqs = torch.fft.rfftfreq(nt, d=dt).to(device)
    omega = 2.0 * math.pi * freqs
    omega_c = omega.to(torch.complex128)

    # Avoid division by zero at DC
    omega_safe = omega_c.clone()
    if omega_safe.numel() > 1:
        omega_safe[0] = omega_safe[1]
    else:
        omega_safe[0] = 1.0 + 0.0j

    # Complex permittivity: ε_c = ε - jσ/ω
    eps_complex = (
        EP0 * torch.tensor(eps_r, device=device, dtype=torch.complex128)
        - 1j * torch.tensor(sigma, device=device, dtype=torch.float64) / omega_safe
    )

    mu = MU0 * torch.tensor(mu_r, device=device, dtype=torch.complex128)

    # Wavenumber k = ω√(με)
    k = omega_safe * torch.sqrt(mu * eps_complex)

    # 3D Green's function: G(r,ω) = exp(-jkr) / (4πr)
    # Field response: E(r,ω) = -jωμ * I * G(r,ω)
    # where I is the source current moment
    green = torch.exp(-1j * k * R) / (4.0 * math.pi * R)

    # Electric field in frequency domain (assuming unit current moment)
    # E = -jωμ * I * G
    I_moment = 1.0  # Unit current moment
    E_freq = -1j * omega_c * mu * I_moment * green

    # Set DC component to zero
    E_freq[0] = 0.0 + 0.0j

    # Convolve with source spectrum
    field_freq = spectrum * E_freq

    # Transform back to time domain
    field_time = torch.fft.irfft(field_freq, n=nt).real

    return t, field_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D Maxwell forward with analytical comparison"
    )
    parser.add_argument("--nz", type=int, default=64, help="Grid size in z")
    parser.add_argument("--ny", type=int, default=64, help="Grid size in y")
    parser.add_argument("--nx", type=int, default=64, help="Grid size in x")
    parser.add_argument("--nt", type=int, default=400, help="Number of time steps")
    parser.add_argument("--pml-width", type=int, default=10, help="PML thickness")
    parser.add_argument("--dx", type=float, default=0.01, help="Grid spacing (m)")
    parser.add_argument("--freq", type=float, default=1e9, help="Source frequency (Hz)")
    parser.add_argument("--eps-r", type=float, default=4.0, help="Relative permittivity")
    parser.add_argument("--sigma", type=float, default=0.0, help="Conductivity (S/m)")
    parser.add_argument("--receiver-offset", type=int, default=15,
                       help="Receiver offset from source (grid cells)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--stencil", type=int, default=2, help="Stencil accuracy (2,4,6,8)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device selection
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    # Grid dimensions
    nz, ny, nx = args.nz, args.ny, args.nx
    nt = args.nt

    # Physical parameters
    dx = dy = dz = args.dx
    eps_r = args.eps_r
    sigma_val = args.sigma

    # Set dt to satisfy CFL without requiring internal subdivision
    # For eps_r, the max velocity is C0/sqrt(eps_r)
    # CFL condition for 3D: dt <= c_max * dx / (v * sqrt(3))
    # Using c_max ~ 0.6 for stability
    c_medium = C0 / math.sqrt(eps_r)
    dt_max = 0.5 * dx / (c_medium * math.sqrt(3.0))  # Conservative factor
    dt = dt_max  # Use the maximum stable dt

    print(f"\nSimulation parameters:")
    print(f"  Grid: {nz} × {ny} × {nx} ({nz*dz:.2f}m × {ny*dy:.2f}m × {nx*dx:.2f}m)")
    print(f"  Time steps: {nt}")
    print(f"  Grid spacing: dx={dx:.4f} m")
    print(f"  Time step: dt={dt*1e12:.4f} ps")
    print(f"  PML width: {args.pml_width}")
    print(f"  Relative permittivity: {eps_r}")
    print(f"  Conductivity: {sigma_val} S/m")
    print(f"  Source frequency: {args.freq/1e9:.2f} GHz")
    print(f"  Stencil order: {args.stencil}")

    # Create material arrays
    epsilon = torch.full((nz, ny, nx), eps_r, device=device, dtype=dtype)
    sigma = torch.full((nz, ny, nx), sigma_val, device=device, dtype=dtype)
    mu = torch.ones((nz, ny, nx), device=device, dtype=dtype)

    # Source and receiver locations (in grid indices)
    src_z, src_y, src_x = nz // 2, ny // 2, nx // 2
    rec_z, rec_y, rec_x = src_z, src_y, src_x + args.receiver_offset

    source_location = torch.tensor([[[src_z, src_y, src_x]]], device=device)
    receiver_location = torch.tensor([[[rec_z, rec_y, rec_x]]], device=device)

    print(f"\nSource location (grid): ({src_z}, {src_y}, {src_x})")
    print(f"Receiver location (grid): ({rec_z}, {rec_y}, {rec_x})")
    print(f"Source-receiver distance: {args.receiver_offset * dx:.3f} m")

    # Generate Ricker wavelet
    freq = args.freq
    wavelet = tide.ricker(
        freq=freq,
        length=nt,
        dt=dt,
        peak_time=1.5/freq,
        dtype=dtype,
        device=device,
    )
    print(f"\nWavelet stats: min={wavelet.min():.4e}, max={wavelet.max():.4e}, "
          f"has_nan={torch.isnan(wavelet).any()}")
    source_amplitude = wavelet.reshape(1, 1, -1)

    # Run FDTD simulation
    print("\nRunning FDTD simulation...")
    start_time = time.perf_counter()

    model = tide.Maxwell3D(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        grid_spacing=[dz, dy, dx],
    )

    #  Note: Using Python backend as CUDA backend may have numerical issues with PML
    result = model(
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=args.pml_width,
        stencil=args.stencil,
        python_backend=True,  # Use Python backend for stability
        source_component="Ez",
        receiver_component="Ez",
    )

    elapsed = time.perf_counter() - start_time

    # maxwell3d returns: (Ex, Ey, Ez, Hx, Hy, Hz,
    #                     m_Hz_y, m_Hy_z, m_Hx_z, m_Hz_x, m_Hy_x, m_Hx_y,
    #                     m_Ey_z, m_Ez_y, m_Ez_x, m_Ex_z, m_Ex_y, m_Ey_x,
    #                     receiver_amplitudes)
    receiver_data = result[-1]  # Last element is receiver_amplitudes

    print(f"Simulation completed in {elapsed:.3f} seconds")
    print(f"Time per step: {elapsed/nt*1e3:.3f} ms")
    print(f"Receiver data shape: {receiver_data.shape}")
    print(f"Receiver data range: [{receiver_data.min():.4e}, {receiver_data.max():.4e}]")

    # Extract numerical solution
    # receiver_data shape should be [nt, n_shots, n_receivers]
    numerical = receiver_data[:, 0, 0].cpu()
    print(f"Numerical solution shape: {numerical.shape}")
    print(f"Numerical solution stats: min={numerical.min():.4e}, max={numerical.max():.4e}, "
          f"mean={numerical.mean():.4e}, has_nan={torch.isnan(numerical).any()}")

    # Compute analytical solution
    print("\nComputing analytical solution...")
    src_pos_m = (src_z * dz, src_y * dy, src_x * dx)
    rec_pos_m = (rec_z * dz, rec_y * dy, rec_x * dx)

    t_array, analytical = analytical_solution_3d(
        wavelet=wavelet.cpu(),
        dt=dt,
        src_pos=src_pos_m,
        rec_pos=rec_pos_m,
        eps_r=eps_r,
        sigma=sigma_val,
        mu_r=1.0,
    )

    # Convert to same dtype for comparison
    analytical = analytical.to(numerical.dtype)
    print(f"Analytical solution stats: min={analytical.min():.4e}, max={analytical.max():.4e}, "
          f"mean={analytical.mean():.4e}, has_nan={torch.isnan(analytical).any()}")

    # Amplitude scaling (FDTD uses different normalization)
    scale = torch.dot(numerical, analytical) / torch.dot(analytical, analytical)
    analytical_scaled = scale * analytical

    # Compute error metrics
    residual = numerical - analytical_scaled
    relative_error = torch.linalg.norm(residual) / torch.linalg.norm(analytical_scaled)

    peak_num_idx = torch.argmax(torch.abs(numerical)).item()
    peak_ana_idx = torch.argmax(torch.abs(analytical_scaled)).item()
    peak_shift = abs(peak_num_idx - peak_ana_idx)

    peak_num_time = peak_num_idx * dt * 1e9
    peak_ana_time = peak_ana_idx * dt * 1e9

    print("\nComparison results:")
    print(f"  Scaling factor: {scale.item():.6f}")
    print(f"  Relative L2 error: {relative_error.item()*100:.2f}%")
    print(f"  Peak shift: {peak_shift} samples ({peak_shift*dt*1e12:.2f} ps)")
    print(f"  Numerical peak time: {peak_num_time:.2f} ns")
    print(f"  Analytical peak time: {peak_ana_time:.2f} ns")
    print(f"  Max numerical amplitude: {numerical.abs().max().item():.4e}")
    print(f"  Max analytical amplitude: {analytical_scaled.abs().max().item():.4e}")

    # Validation
    tolerance = 0.10  # 10% relative error tolerance
    if relative_error < tolerance:
        print(f"\n✓ PASSED: Numerical solution matches analytical solution")
        print(f"  (relative error {relative_error.item()*100:.2f}% < {tolerance*100:.0f}%)")
    else:
        print(f"\n✗ WARNING: Relative error {relative_error.item()*100:.2f}% exceeds tolerance {tolerance*100:.0f}%")
        print(f"  This may be due to numerical dispersion or PML reflections.")

    # Plotting
    if args.plot:
        print("\nGenerating plots...")

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Time axis in nanoseconds
        t_ns = (t_array * 1e9).cpu().numpy()

        # Plot 1: Comparison
        ax = axes[0]
        ax.plot(t_ns, numerical.cpu().numpy(), 'b-', linewidth=2, label='FDTD (numerical)')
        ax.plot(t_ns, analytical_scaled.cpu().numpy(), 'r--', linewidth=2, label='Analytical (scaled)')
        ax.axvline(peak_num_time, color='b', linestyle=':', alpha=0.5, label=f'Peak (FDTD): {peak_num_time:.2f} ns')
        ax.axvline(peak_ana_time, color='r', linestyle=':', alpha=0.5, label=f'Peak (Analytical): {peak_ana_time:.2f} ns')
        ax.set_xlabel('Time (ns)', fontsize=12)
        ax.set_ylabel('Ez Field (V/m)', fontsize=12)
        ax.set_title(f'3D Maxwell: FDTD vs Analytical Solution (εr={eps_r}, σ={sigma_val} S/m)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: Residual
        ax = axes[1]
        residual_np = residual.cpu().numpy()
        ax.plot(t_ns, residual_np, 'g-', linewidth=1.5)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (ns)', fontsize=12)
        ax.set_ylabel('Residual (V/m)', fontsize=12)
        ax.set_title(f'Residual (Numerical - Analytical), Relative Error: {relative_error.item()*100:.2f}%', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f'maxwell3d_analytic_comparison_eps{eps_r}_stencil{args.stencil}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved as: {filename}")

        # Show plot if running interactively
        try:
            plt.show()
        except:
            pass


if __name__ == "__main__":
    main()
