"""TIDE Supershot Wavefield Visualization

展示超级炮的波场传播过程
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tide
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Model Setup
# ============================================================================
ny, nx = 200, 400  # Grid dimensions
dy, dx = 0.02, 0.02  # Grid spacing (m)
dt = 4e-11  # Time step (s)
nt = 600  # Number of time steps

# Two-layer permittivity model with anomalies
epsilon_r = torch.ones(ny, nx, device=device)
epsilon_r[80:] = 4.0  # Lower layer
# Add anomalies
epsilon_r[100:130, 150:200] = 9.0
epsilon_r[90:110, 280:320] = 6.0

# Conductivity and permeability
sigma = torch.zeros(ny, nx, device=device)
mu_r = torch.ones(ny, nx, device=device)

pml_width = 15

# ============================================================================
# Supershot Source Configuration
# ============================================================================
freq = 500e6
peak_time = 3e-9
n_sources = 7

# Source locations - distributed across the surface
source_y = 25
source_x_positions = torch.linspace(60, 340, n_sources, dtype=torch.long)

# Source locations [1, n_sources, 2]
source_loc = torch.zeros(1, n_sources, 2, dtype=torch.long, device=device)
source_loc[0, :, 0] = source_y
source_loc[0, :, 1] = source_x_positions.to(device)

# Receiver (dummy, just need one)
receiver_loc = torch.tensor([[[source_y, 200]]], device=device)

# Source wavelets [1, n_sources, nt]
base_wavelet = tide.ricker(freq, nt, dt, peak_time).to(device)
source_amp = base_wavelet.reshape(1, 1, -1).repeat(1, n_sources, 1)

# ============================================================================
# Forward simulation with wavefield snapshots via callback
# ============================================================================
print("Running supershot simulation...")

# Store snapshots
snapshots = []
snapshot_times = []

def save_snapshot(state):
    """Callback to save wavefield snapshots"""
    if state.step % 20 == 0:  # Save every 20 steps
        snapshots.append(state._wavefields['Ey'].cpu().clone())
        snapshot_times.append(state.step * dt * 1e9)  # Convert to ns

result = tide.maxwelltm(
    epsilon_r, sigma, mu_r,
    grid_spacing=[dy, dx],
    dt=dt,
    source_amplitude=source_amp,
    source_location=source_loc,
    receiver_location=receiver_loc,
    pml_width=pml_width,
    forward_callback=save_snapshot,
    callback_frequency=1,
)

print(f"Captured {len(snapshots)} snapshots")

# ============================================================================
# Visualization
# ============================================================================
# Select 6 snapshots to display
n_display = 6
indices = np.linspace(2, len(snapshots) - 1, n_display, dtype=int)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, ax in zip(indices, axes):
    snapshot = snapshots[idx].squeeze().numpy()
    time_ns = snapshot_times[idx]
    
    # Clip for visualization
    vmax = np.percentile(np.abs(snapshot), 99)
    
    # Plot wavefield
    im = ax.imshow(snapshot, cmap='seismic', aspect='auto',
                   vmin=-vmax, vmax=vmax,
                   extent=[0, nx*dx, ny*dy, 0])
    
    # Overlay model boundaries (permittivity contours)
    eps_np = epsilon_r.cpu().numpy()
    ax.contour(np.linspace(0, nx*dx, nx), np.linspace(0, ny*dy, ny),
               eps_np, levels=[2, 5, 7], colors='k', linewidths=0.5, alpha=0.5)
    
    # Mark source positions
    ax.scatter(source_x_positions.numpy() * dx,
               np.ones(n_sources) * source_y * dy,
               c='yellow', s=50, marker='*', edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f't = {time_ns:.1f} ns')

plt.suptitle(f'Supershot Wavefield (Ey component)\n{n_sources} simultaneous sources', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('example_supershot_wavefield.png', dpi=150)
plt.show()

print("\nSaved to: example_supershot_wavefield.png")
