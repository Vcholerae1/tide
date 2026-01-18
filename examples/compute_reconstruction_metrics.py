"""
Compute reconstruction quality metrics (SSIM, MSE, RMSE, MAE, PSNR)
for the overthrust multi-scale inversion.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_baseline_results():
    """Load the baseline experiment results."""
    # Load the true model
    model_path = Path("examples/data/OverThrust.npy")
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return None, None, None

    epsilon_true_raw = np.load(model_path)

    # Load results from the summary figure
    summary_dir = Path(
        "outputs/multiscale_fir_base600MHz_lp200-400-600_shots100_nb4_nt1500"
    )

    if not summary_dir.exists():
        print(f"Error: Results directory not found: {summary_dir}")
        return None, None, None

    # We'll need to extract the inverted model from saved results
    # For now, let's check what files we have
    print(f"Found results in: {summary_dir}")

    return epsilon_true_raw, None, summary_dir


def compute_metrics(true_model, inverted_model, air_mask=None):
    """
    Compute reconstruction quality metrics.

    Parameters:
    - true_model: Ground truth permittivity
    - inverted_model: Reconstructed permittivity
    - air_mask: Boolean mask for air region (excluded from metrics)

    Returns:
    - Dictionary of metrics
    """
    if air_mask is None:
        air_mask = np.zeros_like(true_model, dtype=bool)

    # Apply mask (exclude air layer)
    true_masked = true_model[~air_mask]
    inverted_masked = inverted_model[~air_mask]

    # Compute metrics
    metrics = {}

    # Mean Squared Error
    metrics["MSE"] = np.mean((true_masked - inverted_masked) ** 2)

    # Root Mean Squared Error
    metrics["RMSE"] = np.sqrt(metrics["MSE"])

    # Mean Absolute Error
    metrics["MAE"] = np.mean(np.abs(true_masked - inverted_masked))

    # Relative Error (%)
    metrics["Relative_Error"] = (metrics["RMSE"] / np.mean(true_masked)) * 100

    # R² Score (coefficient of determination)
    ss_res = np.sum((true_masked - inverted_masked) ** 2)
    ss_tot = np.sum((true_masked - np.mean(true_masked)) ** 2)
    metrics["R2"] = 1 - (ss_res / ss_tot)

    # SSIM (Structural Similarity Index)
    # For 2D images, compute on full arrays
    data_range = true_model.max() - true_model.min()
    metrics["SSIM"] = ssim(true_model, inverted_model, data_range=data_range)

    # PSNR (Peak Signal-to-Noise Ratio)
    metrics["PSNR"] = psnr(true_model, inverted_model, data_range=data_range)

    # Normalized RMSE (NRMSE)
    metrics["NRMSE"] = metrics["RMSE"] / (true_model.max() - true_model.min())

    return metrics


def compute_metrics_from_saved_results():
    """
    Compute metrics by loading saved numpy arrays or reconstructing from the experiment.
    """
    # Load true model
    model_path = Path("examples/data/OverThrust.npy")
    epsilon_true_raw = np.load(model_path)

    # Air layer setup (same as in the experiment)
    air_layer = 3
    epsilon_true = epsilon_true_raw.copy()
    epsilon_true[:air_layer, :] = 1.0

    # Create air mask
    air_mask = np.zeros_like(epsilon_true, dtype=bool)
    air_mask[:air_layer, :] = True

    # Load initial model (smoothed)
    from scipy.ndimage import gaussian_filter

    sigma_smooth = 8
    epsilon_init_raw = gaussian_filter(epsilon_true_raw, sigma=sigma_smooth)
    epsilon_init = epsilon_init_raw.copy()
    epsilon_init[:air_layer, :] = 1.0

    print("=" * 70)
    print("RECONSTRUCTION QUALITY METRICS")
    print("=" * 70)
    print(f"\nModel size: {epsilon_true.shape}")
    print(
        f"Permittivity range: {epsilon_true_raw.min():.2f} - {epsilon_true_raw.max():.2f}"
    )
    print(f"Air layer: {air_layer} grid points (excluded from metrics)")

    # Compute metrics for initial model
    print("\n" + "=" * 70)
    print("INITIAL MODEL (Smoothed) vs TRUE MODEL")
    print("=" * 70)
    metrics_init = compute_metrics(epsilon_true, epsilon_init, air_mask)
    print_metrics(metrics_init)

    # For inverted model, we need to load from experiment output
    # Check if we have saved the inverted model
    # If not, we'll need to re-extract it
    print("\n" + "=" * 70)
    print("INVERTED MODEL vs TRUE MODEL")
    print("=" * 70)
    print("Note: To compute metrics for inverted model, we need to save epsilon_inv")
    print("      from the experiment. Add this to example_multiscale_filtered.py:")
    print("      np.save('outputs/.../epsilon_inverted.npy', eps_result)")

    # Try to find if user has already saved it
    result_dir = Path(
        "outputs/multiscale_fir_base600MHz_lp200-400-600_shots100_nb4_nt1500"
    )
    inverted_file = result_dir / "epsilon_inverted.npy"

    if inverted_file.exists():
        epsilon_inverted = np.load(inverted_file)
        metrics_inv = compute_metrics(epsilon_true, epsilon_inverted, air_mask)
        print_metrics(metrics_inv)

        # Generate comparison table
        generate_latex_table(metrics_init, metrics_inv)

        # Plot error maps
        plot_error_maps(epsilon_true, epsilon_init, epsilon_inverted, air_mask)

        return metrics_init, metrics_inv
    else:
        print(f"Inverted model not found at: {inverted_file}")
        print("\nTo generate metrics, add this line before 'plt.savefig' in")
        print("example_multiscale_filtered.py (around line 360):")
        print("    np.save(output_dir / 'epsilon_inverted.npy', eps_result)")
        print("\nThen re-run the experiment or just the saving part.")

        return metrics_init, None


def print_metrics(metrics):
    """Print metrics in a formatted way."""
    print(f"  MSE:              {metrics['MSE']:.6f}")
    print(f"  RMSE:             {metrics['RMSE']:.6f}")
    print(f"  MAE:              {metrics['MAE']:.6f}")
    print(f"  NRMSE:            {metrics['NRMSE']:.6f}")
    print(f"  Relative Error:   {metrics['Relative_Error']:.2f}%")
    print(f"  R² Score:         {metrics['R2']:.6f}")
    print(f"  SSIM:             {metrics['SSIM']:.6f}")
    print(f"  PSNR:             {metrics['PSNR']:.2f} dB")


def generate_latex_table(metrics_init, metrics_inv):
    """Generate LaTeX table with reconstruction metrics."""
    print("\n" + "=" * 70)
    print("LaTeX TABLE FOR PAPER")
    print("=" * 70)

    table = r"""\begin{table}[t]
\centering
\caption{Reconstruction quality metrics for the overthrust multi-scale inversion. 
Metrics are computed over the subsurface region (excluding air layer).}
\label{tab:overthrust_metrics}
\begin{tabular}{lcc}
\toprule
Metric & Initial Model & Inverted Model \\
\midrule
"""

    # Add metrics
    if metrics_inv is not None:
        table += f"RMSE & {metrics_init['RMSE']:.4f} & {metrics_inv['RMSE']:.4f} \\\\\n"
        table += f"MAE & {metrics_init['MAE']:.4f} & {metrics_inv['MAE']:.4f} \\\\\n"
        table += f"SSIM & {metrics_init['SSIM']:.4f} & {metrics_inv['SSIM']:.4f} \\\\\n"
        table += (
            f"PSNR (dB) & {metrics_init['PSNR']:.2f} & {metrics_inv['PSNR']:.2f} \\\\\n"
        )
        table += f"R² Score & {metrics_init['R2']:.4f} & {metrics_inv['R2']:.4f} \\\\\n"
        table += f"Relative Error (\\%) & {metrics_init['Relative_Error']:.2f} & {metrics_inv['Relative_Error']:.2f} \\\\\n"
    else:
        table += f"RMSE & {metrics_init['RMSE']:.4f} & TODO \\\\\n"
        table += f"MAE & {metrics_init['MAE']:.4f} & TODO \\\\\n"
        table += f"SSIM & {metrics_init['SSIM']:.4f} & TODO \\\\\n"
        table += f"PSNR (dB) & {metrics_init['PSNR']:.2f} & TODO \\\\\n"
        table += f"R² Score & {metrics_init['R2']:.4f} & TODO \\\\\n"
        table += (
            f"Relative Error (\\%) & {metrics_init['Relative_Error']:.2f} & TODO \\\\\n"
        )

    table += r"""\bottomrule
\end{tabular}
\end{table}"""

    print(table)

    # Save to file
    output_file = Path("outputs/reconstruction_metrics_table.tex")
    with open(output_file, "w") as f:
        f.write(table)
    print(f"\nSaved LaTeX table to: {output_file}")


def plot_error_maps(true_model, init_model, inverted_model, air_mask):
    """Plot error maps for initial and inverted models."""

    error_init = np.abs(true_model - init_model)
    error_inv = np.abs(true_model - inverted_model)

    # Mask air layer for visualization
    error_init_masked = error_init.copy()
    error_inv_masked = error_inv.copy()
    error_init_masked[air_mask] = 0
    error_inv_masked[air_mask] = 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    vmax = max(error_init_masked.max(), error_inv_masked.max())

    im1 = axes[0].imshow(
        error_init_masked, aspect="auto", cmap="hot", vmin=0, vmax=vmax
    )
    axes[0].set_title("Absolute Error: Initial Model", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("X (grid points)")
    axes[0].set_ylabel("Y (grid points)")
    plt.colorbar(im1, ax=axes[0], label="|ε_true - ε_init|")

    im2 = axes[1].imshow(error_inv_masked, aspect="auto", cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title("Absolute Error: Inverted Model", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("X (grid points)")
    axes[1].set_ylabel("Y (grid points)")
    plt.colorbar(im2, ax=axes[1], label="|ε_true - ε_inv|")

    plt.tight_layout()

    output_path = Path("outputs/error_maps_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved error maps to: {output_path}")
    plt.close()


def create_quick_save_snippet():
    """Generate code snippet to save epsilon_inv from experiment."""
    print("\n" + "=" * 70)
    print("CODE SNIPPET TO ADD TO example_multiscale_filtered.py")
    print("=" * 70)
    print("""
Add these lines after line ~367 (after eps_result is computed):

# Save inverted model for metrics computation
np.save(output_dir / 'epsilon_inverted.npy', eps_result)
print(f"Saved inverted model to '{output_dir / 'epsilon_inverted.npy'}'")

Then re-run just the saving part or the full experiment.
""")


def main():
    """Main entry point."""

    print("=" * 70)
    print("COMPUTING RECONSTRUCTION QUALITY METRICS")
    print("=" * 70)

    metrics_init, metrics_inv = compute_metrics_from_saved_results()

    if metrics_inv is None:
        create_quick_save_snippet()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The metrics table shows quantitative reconstruction quality:
- RMSE/MAE: Lower is better (measures average error)
- SSIM: Higher is better (0-1, structural similarity)
- PSNR: Higher is better (signal-to-noise ratio in dB)
- R²: Higher is better (0-1, coefficient of determination)

These metrics replace the FP32/BF16/FP8 performance comparison table
and provide more meaningful assessment of inversion quality.
""")


if __name__ == "__main__":
    main()
