"""
Quick script to save epsilon_inverted.npy from the completed baseline run.
Since the experiment already ran, we can extract it from the summary figure or re-run just the final saving.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Just re-run the final part to save the inverted model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load true model
model_path = "examples/data/OverThrust.npy"
epsilon_true_raw = np.load(model_path)
air_layer = 3

# Create initial model (same as in experiment)
sigma_smooth = 8
epsilon_init_raw = gaussian_filter(epsilon_true_raw, sigma=sigma_smooth)
epsilon_init_np = epsilon_init_raw.copy()
epsilon_init_np[:air_layer, :] = 1.0

# For now, save the initial model as a placeholder
# User needs to re-run the full experiment with the updated code
output_dir = Path("outputs/multiscale_fir_base600MHz_lp200-400-600_shots100_nb4_nt1500")
output_file = output_dir / "epsilon_inverted.npy"

if output_file.exists():
    print(f"epsilon_inverted.npy already exists at: {output_file}")
    epsilon_inv = np.load(output_file)
    print(f"Loaded inverted model shape: {epsilon_inv.shape}")
else:
    print("epsilon_inverted.npy not found.")
    print("\nTo generate it, please re-run the experiment:")
    print("  cd /home/vcholerae/projects/tide")
    print("  uv run --with torch --with numpy --with scipy --with matplotlib \\")
    print("    python examples/example_multiscale_filtered.py")
    print("\nThe updated script will now save epsilon_inverted.npy automatically.")
    print("\nAlternatively, if you still have the Python session running,")
    print("you can manually save it with:")
    print(
        "  np.save('outputs/multiscale_fir_base600MHz_lp200-400-600_shots100_nb4_nt1500/epsilon_inverted.npy', eps_result)"
    )
