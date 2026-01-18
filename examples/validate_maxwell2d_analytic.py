#!/usr/bin/env python3
"""
2D TM Maxwell equations validation against analytical solution.

This example demonstrates forward-solver validation by comparing simulated fields
in a homogeneous medium against an analytic Green's-function solution. We use a
compact source-time function and record E-field traces at several receiver offsets.

Evaluation metrics:
- Waveform agreement at receiver points (time series overlay)
- Relative ℓ₂ error over time windows
- Sensitivity to grid/time discretization (CFL and dispersion trends)

The analytical solution uses the 2D TM mode Hankel function Green's function
for a line current source in a homogeneous medium.

Reference:
- Chew, W. C. (1995). Waves and Fields in Inhomogeneous Media. IEEE Press.
"""

import argparse
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import torch
from matplotlib.gridspec import GridSpec

import tide

# Physical constants
EPS0 = 1.0 / (36.0 * math.pi) * 1e-9  # vacuum permittivity (F/m)
MU0 = 4.0 * math.pi * 1e-7  # vacuum permeability (H/m)
C0 = 1.0 / math.sqrt(EPS0 * MU0)  # speed of light (m/s)


def analytic_solution_2d_tm(
    wavelet: torch.Tensor,
    dt: float,
    src_pos_m: tuple[float, float],
    rec_pos_m: tuple[float, float],
    eps_r: float,
    sigma: float = 0.0,
    current: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute analytical 2D TM solution in a homogeneous medium.

    Uses frequency-domain Green's function with Hankel function for 2D cylindrical waves.

    Args:
        wavelet: Source time function [nt]
        dt: Time step (s)
        src_pos_m: Source position in meters (y, x)
        rec_pos_m: Receiver position in meters (y, x)
        eps_r: Relative permittivity
        sigma: Conductivity (S/m)
        current: Source current amplitude

    Returns:
        (time, field): Time array and electric field at receiver
    """
    device = wavelet.device
    dtype = torch.float64
    nt = wavelet.numel()

    # Time array
    t = torch.arange(nt, device=device, dtype=dtype) * dt

    # Source-receiver distance
    r_vec = torch.tensor(rec_pos_m, device=device, dtype=dtype) - torch.tensor(
        src_pos_m, device=device, dtype=dtype
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
        EPS0 * torch.tensor(eps_r, device=device, dtype=torch.complex128)
        - 1j * torch.tensor(sigma, device=device, dtype=torch.float64) / omega_safe
    )

    # Wavenumber
    k = omega_safe * torch.sqrt(MU0 * eps_complex)

    # 2D Green's function using Hankel function of second kind, order 0
    hankel0 = torch.from_numpy(scipy.special.hankel2(0, (k * R).cpu().numpy())).to(
        device=device, dtype=torch.complex128
    )

    # Green's function: G = -jωμ₀ H₀⁽²⁾(kR) / 4
    green = -current * omega_safe * MU0 * hankel0 / 4.0
    green[0] = 0.0 + 0.0j  # Set DC component to zero

    # Apply Green's function in frequency domain
    field_freq = spectrum * green

    # Transform back to time domain
    field_time = torch.fft.irfft(field_freq, n=nt).real

    return t, field_time


def run_simulation_and_validation(
    freq0: float = 9e8,
    dt: float = 1e-11,
    nt: int = 800,
    dx: float = 0.005,
    eps_r: float = 10.0,
    conductivity: float = 1e-3,
    receiver_offsets: list = None,
    device: torch.device = None,
):
    """
    Run FDTD simulation and compare with analytical solution at multiple receivers.

    Args:
        freq0: Source center frequency (Hz)
        dt: Time step (s)
        nt: Number of time steps
        dx: Grid spacing (m)
        eps_r: Relative permittivity
        conductivity: Conductivity (S/m)
        receiver_offsets: List of receiver x-offsets in grid points
        device: Torch device

    Returns:
        Dictionary with simulation results and analytical solutions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if receiver_offsets is None:
        receiver_offsets = [10, 20, 30, 40]

    dtype = torch.float64

    # Grid setup
    ny, nx = 96, 128
    src_idx = (ny // 2, nx // 2)

    # Setup medium properties
    epsilon = torch.full((ny, nx), eps_r, device=device, dtype=dtype)
    sigma = torch.full_like(epsilon, conductivity)
    mu = torch.ones_like(epsilon)

    # Source wavelet (Ricker wavelet)
    wavelet = tide.ricker(
        freq0, nt, dt, peak_time=1.0 / freq0, dtype=dtype, device=device
    )
    source_amplitude = wavelet.view(1, 1, nt)
    source_location = torch.tensor([[src_idx]], device=device)

    # Setup multiple receivers
    receiver_indices = [
        (src_idx[0], src_idx[1] + offset) for offset in receiver_offsets
    ]
    receiver_location = torch.tensor([receiver_indices], device=device)

    print("Running FDTD simulation...")
    print(f"  Grid size: {ny} x {nx}")
    print(f"  Time steps: {nt}")
    print(f"  dt: {dt * 1e12:.3f} ps")
    print(f"  Grid spacing: {dx * 1e3:.3f} mm")
    print(f"  Source frequency: {freq0 * 1e-9:.2f} GHz")
    print(f"  Number of receivers: {len(receiver_offsets)}")

    start_time = time.time()

    # Run FDTD simulation
    _, _, _, _, _, _, _, receivers = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=dx,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        stencil=2,
        pml_width=10,
        save_snapshots=False,
    )

    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.3f} seconds")

    # Get simulated traces
    simulated_traces = receivers[:, 0, :].cpu()  # [nt, n_receivers]

    # Compute analytical solutions for each receiver
    src_pos_m = (src_idx[0] * dx, src_idx[1] * dx)
    analytical_traces = []
    distances = []

    print("\nComputing analytical solutions...")
    for i, rec_idx in enumerate(receiver_indices):
        rec_pos_m = (rec_idx[0] * dx, rec_idx[1] * dx)
        distance = math.sqrt(
            (rec_pos_m[0] - src_pos_m[0]) ** 2 + (rec_pos_m[1] - src_pos_m[1]) ** 2
        )
        distances.append(distance)

        _, analytic = analytic_solution_2d_tm(
            wavelet=wavelet.cpu(),
            dt=dt,
            src_pos_m=src_pos_m,
            rec_pos_m=rec_pos_m,
            eps_r=eps_r,
            sigma=conductivity,
        )
        analytical_traces.append(analytic)

    # Compute metrics
    relative_errors = []
    peak_shifts = []
    scales = []

    print("\nValidation metrics:")
    print(f"{'Offset':<10} {'Distance':<15} {'Rel. Error':<15} {'Peak Shift':<15}")
    print("-" * 60)

    for i in range(len(receiver_offsets)):
        sim = simulated_traces[:, i]
        ana = analytical_traces[i]

        # Amplitude scaling (least-squares fit)
        scale = torch.dot(sim, ana) / torch.dot(ana, ana)
        scales.append(scale.item())

        # Relative L2 error
        rel_error = (
            torch.linalg.norm(sim - scale * ana) / torch.linalg.norm(ana)
        ).item()
        relative_errors.append(rel_error)

        # Peak time shift
        peak_shift = abs(int(sim.abs().argmax()) - int(ana.abs().argmax()))
        peak_shifts.append(peak_shift)

        print(
            f"{receiver_offsets[i]:<10} {distances[i] * 1e3:>10.3f} mm  "
            f"{rel_error:>12.4f}    {peak_shift:>10} steps"
        )

    return {
        "time": torch.arange(nt, dtype=dtype) * dt,
        "simulated_traces": simulated_traces,
        "analytical_traces": analytical_traces,
        "receiver_offsets": receiver_offsets,
        "distances": distances,
        "relative_errors": relative_errors,
        "peak_shifts": peak_shifts,
        "scales": scales,
        "dt": dt,
        "dx": dx,
        "freq0": freq0,
        "eps_r": eps_r,
    }


def plot_validation_results(results: dict, save_path: str = None):
    """
    Create comprehensive validation plots.

    Args:
        results: Dictionary from run_simulation_and_validation
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    time_ms = results["time"].numpy() * 1e9  # Convert to ns
    n_receivers = len(results["receiver_offsets"])

    # (a) Simulated vs analytical traces at several offsets
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.viridis(np.linspace(0, 1, n_receivers))

    for i in range(n_receivers):
        offset = results["receiver_offsets"][i]
        dist_mm = results["distances"][i] * 1e3
        sim = results["simulated_traces"][:, i].numpy()
        ana = results["analytical_traces"][i].numpy()
        scale = results["scales"][i]

        # Normalize for visualization
        sim_norm = sim / np.abs(sim).max()
        ana_norm = ana / np.abs(ana).max()

        ax1.plot(
            time_ms,
            sim_norm + i * 2.5,
            color=colors[i],
            linewidth=1.5,
            label=f"Sim (offset={offset}, {dist_mm:.1f} mm)",
        )
        ax1.plot(
            time_ms,
            ana_norm + i * 2.5,
            color=colors[i],
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="Analytic",
        )

    ax1.set_xlabel("Time (ns)", fontsize=12)
    ax1.set_ylabel("Normalized E-field (offset for clarity)", fontsize=12)
    ax1.set_title(
        "(a) Simulated vs Analytical Traces at Multiple Receiver Offsets",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, ncol=2, loc="upper right")

    # (b) Relative L2 error vs time
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(n_receivers):
        offset = results["receiver_offsets"][i]
        sim = results["simulated_traces"][:, i].numpy()
        ana = results["analytical_traces"][i].numpy()
        scale = results["scales"][i]

        # Compute cumulative relative error
        cumulative_error = []
        for t in range(10, len(sim)):
            err = np.linalg.norm(sim[:t] - scale * ana[:t]) / np.linalg.norm(ana[:t])
            cumulative_error.append(err)

        ax2.plot(
            time_ms[10:],
            cumulative_error,
            color=colors[i],
            linewidth=1.5,
            label=f"Offset {offset}",
        )

    ax2.set_xlabel("Time (ns)", fontsize=12)
    ax2.set_ylabel("Relative ℓ₂ Error", fontsize=12)
    ax2.set_title("(b) Relative Error vs Time", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_ylim(bottom=0)

    # (c) Error vs receiver offset/distance
    ax3 = fig.add_subplot(gs[1, 1])
    distances_mm = [d * 1e3 for d in results["distances"]]
    ax3.plot(
        distances_mm,
        results["relative_errors"],
        "o-",
        linewidth=2,
        markersize=8,
        color="steelblue",
    )
    ax3.set_xlabel("Source-Receiver Distance (mm)", fontsize=12)
    ax3.set_ylabel("Relative ℓ₂ Error", fontsize=12)
    ax3.set_title("(c) Error vs Distance", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # (d) Detailed comparison at furthest receiver
    ax4 = fig.add_subplot(gs[2, 0])
    furthest_idx = n_receivers - 1
    sim = results["simulated_traces"][:, furthest_idx].numpy()
    ana = results["analytical_traces"][furthest_idx].numpy()
    scale = results["scales"][furthest_idx]

    ax4.plot(time_ms, sim, linewidth=2, label="Simulated", color="steelblue")
    ax4.plot(
        time_ms,
        scale * ana,
        linewidth=2,
        linestyle="--",
        label="Analytical (scaled)",
        color="orangered",
        alpha=0.8,
    )
    ax4.set_xlabel("Time (ns)", fontsize=12)
    ax4.set_ylabel("E-field (a.u.)", fontsize=12)
    ax4.set_title(
        f"(d) Detailed Comparison at Offset {results['receiver_offsets'][furthest_idx]}",
        fontsize=13,
        fontweight="bold",
    )
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # (e) Residual error
    ax5 = fig.add_subplot(gs[2, 1])
    residual = sim - scale * ana
    ax5.plot(time_ms, residual, linewidth=1.5, color="crimson")
    ax5.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax5.set_xlabel("Time (ns)", fontsize=12)
    ax5.set_ylabel("Residual (Sim - Analytical)", fontsize=12)
    ax5.set_title("(e) Residual Error", fontsize=13, fontweight="bold")
    ax5.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")

    plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Validate 2D TM Maxwell solver against analytical solution"
    )
    parser.add_argument(
        "--freq0", type=float, default=9e8, help="Source center frequency (Hz)"
    )
    parser.add_argument("--dt", type=float, default=1e-11, help="Time step (s)")
    parser.add_argument("--nt", type=int, default=800, help="Number of time steps")
    parser.add_argument("--dx", type=float, default=0.005, help="Grid spacing (m)")
    parser.add_argument(
        "--eps-r", type=float, default=10.0, help="Relative permittivity"
    )
    parser.add_argument(
        "--conductivity", type=float, default=1e-3, help="Conductivity (S/m)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="maxwell2d_validation.png",
        help="Output figure filename",
    )
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU execution")

    args = parser.parse_args()

    # Setup device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Running on CPU")
    else:
        device = torch.device("cuda")
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")

    # Run simulation and validation
    results = run_simulation_and_validation(
        freq0=args.freq0,
        dt=args.dt,
        nt=args.nt,
        dx=args.dx,
        eps_r=args.eps_r,
        conductivity=args.conductivity,
        receiver_offsets=[10, 20, 30, 40],
        device=device,
    )

    # Create and save plots
    plot_validation_results(results, save_path=args.output)

    # Print final summary
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Average relative ℓ₂ error: {np.mean(results['relative_errors']):.4f}")
    print(f"Maximum relative ℓ₂ error: {np.max(results['relative_errors']):.4f}")
    print(f"All errors < 5%: {all(e < 0.05 for e in results['relative_errors'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
