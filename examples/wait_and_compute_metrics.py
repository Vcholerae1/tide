"""
Wait for experiment to complete and then compute metrics automatically.
"""

import subprocess
import time
from pathlib import Path

# Wait for epsilon_inverted.npy to be created
output_file = Path(
    "outputs/multiscale_fir_base600MHz_lp200-400-600_shots100_nb4_nt1500/epsilon_inverted.npy"
)

print("Waiting for experiment to complete and save epsilon_inverted.npy...")
print(f"Watching for: {output_file}")

while not output_file.exists():
    time.sleep(10)
    print(".", end="", flush=True)

print("\n\n✓ epsilon_inverted.npy found!")
print("Computing reconstruction metrics...")

# Run the metrics computation
subprocess.run(["python3", "examples/compute_reconstruction_metrics.py"])

print("\n✓ Done! Check outputs/ for:")
print("  - reconstruction_metrics_table.tex")
print("  - error_maps_comparison.png")
