"""
Generate comprehensive figures for Experiment 2 from existing results.
Combines all baseline results and benchmark results into publication-ready figures.
"""

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_benchmark_results():
    """Load benchmark results from JSON."""
    results_file = Path("outputs/benchmark_results.json")
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def create_combined_figure():
    """Create comprehensive figure with (a) models (b) loss (c) runtime (d) memory."""

    # Load the summary image from baseline run
    summary_path = Path(
        "outputs/multiscale_fir_base600MHz_lp200-400-600_shots100_nb4_nt1500/multiscale_filtered_summary.jpg"
    )

    if not summary_path.exists():
        print(f"Error: Summary image not found at {summary_path}")
        return

    # Load benchmark results
    benchmark_results = load_benchmark_results()

    if benchmark_results is None:
        print(
            "Benchmark results not yet available. Run benchmark_configurations.py first."
        )
        create_baseline_figure_only()
        return

    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Load and display the baseline summary (top row)
    summary_img = Image.open(summary_path)
    summary_array = np.array(summary_img)

    # Split the summary image into 4 quadrants
    h, w = summary_array.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    # Top-left: True model
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(summary_array[:mid_h, :mid_w])
    ax1.set_title("(a) True Model", fontsize=14, fontweight="bold")
    ax1.axis("off")

    # Top-middle: Initial model
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(summary_array[:mid_h, mid_w:])
    ax2.set_title("(b) Initial Model (Smoothed)", fontsize=14, fontweight="bold")
    ax2.axis("off")

    # Top-right: Inverted model
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(summary_array[mid_h:, :mid_w])
    ax3.set_title("(c) Multiscale Inverted Result", fontsize=14, fontweight="bold")
    ax3.axis("off")

    # Middle row: Loss curve (spans all columns)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.imshow(summary_array[mid_h:, mid_w:])
    ax4.set_title(
        "(d) Loss Curve (3-stage multiscale inversion)", fontsize=14, fontweight="bold"
    )
    ax4.axis("off")

    # Bottom row: Performance comparison
    successful_results = [r for r in benchmark_results if r.get("success", False)]

    if successful_results:
        names = [r["name"].replace("Mini-batch + ", "") for r in successful_results]
        times = [r["elapsed_time"] for r in successful_results]
        memories = [r["peak_memory_gb"] for r in successful_results]

        # Runtime comparison
        ax5 = fig.add_subplot(gs[2, 0:2])
        bars1 = ax5.barh(names, times, color="steelblue", alpha=0.8)
        ax5.set_xlabel("Total Runtime (seconds)", fontsize=12)
        ax5.set_title("(e) Runtime Comparison", fontsize=14, fontweight="bold")
        ax5.grid(axis="x", alpha=0.3, linestyle="--")

        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars1, times)):
            ax5.text(
                time + max(times) * 0.02, i, f"{time:.1f}s", va="center", fontsize=10
            )

        # Memory comparison
        ax6 = fig.add_subplot(gs[2, 2])
        bars2 = ax6.barh(names, memories, color="coral", alpha=0.8)
        ax6.set_xlabel("Peak GPU Memory (GB)", fontsize=12)
        ax6.set_title("(f) Memory Usage", fontsize=14, fontweight="bold")
        ax6.grid(axis="x", alpha=0.3, linestyle="--")

        # Add value labels
        for i, (bar, mem) in enumerate(zip(bars2, memories)):
            ax6.text(
                mem + max(memories) * 0.02, i, f"{mem:.1f}GB", va="center", fontsize=10
            )

    # No suptitle - removed as requested

    output_path = Path("outputs/experiment2_comprehensive.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved comprehensive figure to {output_path}")
    plt.close()


def create_baseline_figure_only():
    """Create figure with just the baseline results (when benchmark not done yet)."""

    summary_path = Path(
        "outputs/multiscale_fir_base600MHz_lp200-400-600_shots100_nb4_nt1500/multiscale_filtered_summary.jpg"
    )

    if not summary_path.exists():
        print(f"Error: Summary image not found at {summary_path}")
        return

    fig = plt.figure(figsize=(14, 10))

    summary_img = Image.open(summary_path)
    summary_array = np.array(summary_img)

    h, w = summary_array.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(summary_array[:mid_h, :mid_w])
    ax1.set_title("(a) True Model", fontsize=14, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(summary_array[:mid_h, mid_w:])
    ax2.set_title("(b) Initial Model (Smoothed)", fontsize=14, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(summary_array[mid_h:, :mid_w])
    ax3.set_title("(c) Inverted Result", fontsize=14, fontweight="bold")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(summary_array[mid_h:, mid_w:])
    ax4.set_title("(d) Loss Curve", fontsize=14, fontweight="bold")
    ax4.axis("off")

    # No suptitle - removed as requested

    output_path = Path("outputs/experiment2_baseline.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved baseline figure to {output_path}")
    plt.close()


def create_stage_progression_figure():
    """Create figure showing progression through stages."""

    base_dir = Path(
        "outputs/multiscale_fir_base600MHz_lp200-400-600_shots100_nb4_nt1500"
    )

    stage_files = [
        ("epsilon_stage_lp250.jpg", "Stage 1: 200 MHz"),
        ("epsilon_stage_lp500.jpg", "Stage 2: 400 MHz"),
        ("epsilon_stage_lp700.jpg", "Stage 3: 600 MHz"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (filename, title) in zip(axes, stage_files):
        img_path = base_dir / filename
        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(np.array(img))
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, "Not found", ha="center", va="center")
            ax.axis("off")

    # No suptitle - removed as requested
    plt.tight_layout()

    output_path = Path("outputs/experiment2_stage_progression.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved stage progression figure to {output_path}")
    plt.close()


def generate_detailed_latex_table():
    """Generate detailed LaTeX table with all configurations."""

    benchmark_results = load_benchmark_results()

    if benchmark_results is None:
        print("\nBenchmark results not available yet.")
        return

    print("\n" + "=" * 70)
    print("DETAILED LaTeX TABLE FOR EXPERIMENT 2")
    print("=" * 70)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Performance summary for the overthrust multi-scale inversion with different storage compression strategies.}"
    )
    lines.append(r"\label{tab:overthrust_perf}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Configuration & Peak GPU Mem (GB) & Total Time (s) & Iter/s & Final Loss \\"
    )
    lines.append(r"\midrule")

    for res in benchmark_results:
        if res.get("success", False):
            name = res["name"]
            mem = f"{res['peak_memory_gb']:.2f}" if res.get("peak_memory_gb") else "N/A"
            time_s = f"{res['elapsed_time']:.1f}" if res.get("elapsed_time") else "N/A"
            iter_s = f"{res['iter_per_sec']:.3f}" if res.get("iter_per_sec") else "N/A"
            loss = f"{res['final_loss']:.2e}" if res.get("final_loss") else "N/A"

            name_escaped = name.replace("_", r"\_").replace("+", r"\&")

            lines.append(f"{name_escaped} & {mem} & {time_s} & {iter_s} & {loss} \\\\")
        else:
            name_escaped = res["name"].replace("_", r"\_").replace("+", r"\&")
            lines.append(f"{name_escaped} & - & - & - & Failed \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    print(table)

    # Save to file
    latex_file = Path("outputs/experiment2_table.tex")
    with open(latex_file, "w") as f:
        f.write(table)
    print(f"\nSaved LaTeX table to {latex_file}")

    return table


def print_summary_statistics():
    """Print summary statistics from all runs."""

    benchmark_results = load_benchmark_results()

    print("\n" + "=" * 70)
    print("EXPERIMENT 2 SUMMARY STATISTICS")
    print("=" * 70)

    # Baseline info
    print("\nBaseline Configuration:")
    print("  Model: Overthrust (200x400 grid)")
    print("  Shots: 100, Mini-batches: 4")
    print("  Time steps: 1500")
    print("  Stages: 3 (200 MHz → 400 MHz → 600 MHz)")
    print("  Total epochs per stage: AdamW (40,30,10) + LBFGS (6,6,6)")

    if benchmark_results:
        print("\nPerformance Comparison:")
        successful = [r for r in benchmark_results if r.get("success", False)]

        if successful:
            # Find baseline (FP32)
            fp32 = next((r for r in successful if "FP32" in r["name"]), None)

            print(
                f"\n{'Configuration':<25} {'Time (s)':<12} {'Memory (GB)':<15} {'Speedup':<10} {'Mem Reduction'}"
            )
            print("-" * 70)

            for res in successful:
                name = res["name"].replace("Mini-batch + ", "")
                time = res["elapsed_time"]
                mem = res["peak_memory_gb"]

                if fp32:
                    speedup = fp32["elapsed_time"] / time if time > 0 else 0
                    mem_reduction = (
                        (1 - mem / fp32["peak_memory_gb"]) * 100 if mem > 0 else 0
                    )
                    print(
                        f"{name:<25} {time:<12.1f} {mem:<15.2f} {speedup:<10.2f}x {mem_reduction:>5.1f}%"
                    )
                else:
                    print(f"{name:<25} {time:<12.1f} {mem:<15.2f} {'N/A':<10} {'N/A'}")


def main():
    """Generate all figures and tables for Experiment 2."""

    print("=" * 70)
    print("GENERATING EXPERIMENT 2 FIGURES AND TABLES")
    print("=" * 70)

    # Create comprehensive figure
    create_combined_figure()

    # Create stage progression figure
    create_stage_progression_figure()

    # Generate LaTeX table
    generate_detailed_latex_table()

    # Print summary statistics
    print_summary_statistics()

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - outputs/experiment2_comprehensive.png (or experiment2_baseline.png)")
    print("  - outputs/experiment2_stage_progression.png")
    print("  - outputs/experiment2_table.tex")
    print("  - outputs/benchmark_results.json")
    print("\nThese files are ready for inclusion in your paper/report.")


if __name__ == "__main__":
    main()
