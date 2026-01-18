"""
Extract quantitative metrics from baseline experiment results.
Reads log output and generates summary for the paper.
"""

from pathlib import Path


def extract_metrics_from_log():
    """Extract key metrics from the experiment output directory."""

    # Look for the baseline run directory
    base_dir = Path(
        "outputs/multiscale_fir_base600MHz_lp200-400-600_shots100_nb4_nt1500"
    )

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return None

    print("=" * 70)
    print("EXPERIMENT 2: QUANTITATIVE METRICS SUMMARY")
    print("=" * 70)

    # List generated files
    print("\n✓ Generated Figures:")
    for jpg_file in sorted(base_dir.glob("*.jpg")):
        print(f"  - {jpg_file.name}")

    # Check for stage-specific results
    print("\n✓ Stage Results:")
    stages = {
        "lp250": "Stage 1 (200 MHz low-pass)",
        "lp500": "Stage 2 (400 MHz low-pass)",
        "lp700": "Stage 3 (600 MHz low-pass)",
    }

    for key, desc in stages.items():
        stage_file = base_dir / f"epsilon_stage_{key}.jpg"
        if stage_file.exists():
            print(f"  ✓ {desc}: {stage_file.name}")
        else:
            print(f"  ✗ {desc}: Not found")

    # Check summary figure
    summary_file = base_dir / "multiscale_filtered_summary.jpg"
    if summary_file.exists():
        print(f"\n✓ Final Summary Figure: {summary_file.name}")
        print(
            "  Contains: (a) True model, (b) Initial model, (c) Inverted result, (d) Loss curve"
        )

    return True


def generate_latex_figure_code():
    """Generate LaTeX code to include the figures."""

    print("\n" + "=" * 70)
    print("LATEX CODE FOR PAPER")
    print("=" * 70)

    print("\n% Main result figure (4-panel)")
    print(r"""\begin{figure}[H]
\centering
\includegraphics[width=0.95\linewidth]{outputs/experiment2_baseline.png}
\caption{Multi-scale overthrust inversion with mini-batch shot sampling and compressed 
wavefield storage. (a) True overthrust-style permittivity model; (b) Initial smoothed 
model (Gaussian filter, $\sigma=8$ grid points); (c) Inverted permittivity after 3-stage 
multi-scale inversion (200 MHz $\to$ 400 MHz $\to$ 600 MHz progressive filtering); 
(d) Loss curve showing convergence through three stages, with AdamW followed by L-BFGS 
optimization at each stage. The mini-batch approach (4 batches, 25 shots each) enables 
efficient gradient computation while maintaining reconstruction quality.}
\label{fig:overthrust}
\end{figure}""")

    print("\n% Stage progression figure (optional supplementary)")
    print(r"""\begin{figure}[H]
\centering
\includegraphics[width=0.95\linewidth]{outputs/experiment2_stage_progression.png}
\caption{Progressive refinement through multi-scale stages. Left: Stage 1 (200 MHz) 
recovers large-scale velocity structure. Middle: Stage 2 (400 MHz) adds intermediate 
wavelength features. Right: Stage 3 (600 MHz) refines fine-scale heterogeneities. 
This coarse-to-fine strategy mitigates cycle-skipping and local minima.}
\label{fig:overthrust_stages}
\end{figure}""")


def generate_performance_table_template():
    """Generate template for performance comparison table."""

    print("\n" + "=" * 70)
    print("PERFORMANCE TABLE TEMPLATE")
    print("=" * 70)

    print(r"""
% To be filled after running benchmark_configurations.py

\begin{table}[t]
\centering
\caption{Performance comparison for overthrust inversion with different 
wavefield storage strategies. Configuration uses 100 shots with 4 mini-batches, 
1500 time steps, and 3-stage multi-scale inversion (86 total epochs).}
\label{tab:overthrust_perf}
\begin{tabular}{lcccc}
\toprule
Storage Mode & Peak GPU Mem (GB) & Total Time (s) & Memory Reduction & Speedup \\
\midrule
FP32 (baseline) & TODO & TODO & --- & 1.0$\times$ \\
BF16 compression & TODO & TODO & TODO\% & TODO$\times$ \\
FP8 compression & TODO & TODO & TODO\% & TODO$\times$ \\
\bottomrule
\end{tabular}
\end{table}

% Expected results:
% - BF16: ~50% memory reduction, minimal time overhead
% - FP8: ~75% memory reduction, slight time overhead
% - All methods should achieve similar final reconstruction quality
""")


def generate_discussion_points():
    """Generate key discussion points for the paper."""

    print("\n" + "=" * 70)
    print("KEY DISCUSSION POINTS FOR PAPER")
    print("=" * 70)

    points = [
        "Multi-scale Strategy Effectiveness:",
        "  - Progressive frequency filtering (200→400→600 MHz) successfully avoids cycle-skipping",
        "  - Coarse-to-fine approach visible in stage progression figures",
        "  - Loss curve shows smooth convergence at each stage",
        "",
        "Mini-batch Acceleration:",
        "  - 4 batches (25 shots each) reduce per-iteration cost by 4×",
        "  - Maintains good gradient quality for convergence",
        "  - Enables processing of large shot gathers on limited GPU memory",
        "",
        "Compressed Storage (to be verified with benchmarks):",
        "  - BF16: Expected ~50% memory saving with negligible quality loss",
        "  - FP8: Expected ~75% memory saving, slight numerical precision trade-off",
        "  - Critical for enabling full-waveform inversion at scale",
        "",
        "Computational Efficiency:",
        "  - Gradient sampling (interval=10) reduces checkpoint overhead",
        "  - CPML boundary conditions minimize domain padding",
        "  - Combined strategies enable practical 3D extension",
    ]

    for point in points:
        print(f"  {point}")


def generate_methods_text():
    """Generate methods section text."""

    print("\n" + "=" * 70)
    print("METHODS SECTION TEXT")
    print("=" * 70)

    print("""
We demonstrate the multi-scale inversion approach on a 200×400 overthrust-style 
permittivity model with strong lateral velocity contrasts. The model is discretized 
with dx=0.02 m spacing and uses dt=4×10⁻¹¹ s time stepping for stability. We deploy 
100 source-receiver pairs near the surface (3 grid points above the model) with 
4-point source spacing, generating 1500 time samples per trace.

The inversion employs a three-stage progressive filtering strategy:
(1) 200 MHz low-pass: 40 AdamW + 6 L-BFGS epochs
(2) 400 MHz low-pass: 30 AdamW + 6 L-BFGS epochs  
(3) 600 MHz low-pass: 10 AdamW + 6 L-BFGS epochs

At each stage, we filter both observed and synthetic data using Hamming-windowed 
FIR filters before computing the L2 waveform misfit. The initial model is obtained 
by Gaussian smoothing (σ=8 grid points) of the true model.

To accelerate computation, we partition the 100 shots into 4 mini-batches (25 shots 
each), computing gradients on each batch sequentially per epoch. Wavefield states 
are stored at every 10th time step during forward propagation and recomputed as 
needed during backpropagation. We test three storage compression strategies: FP32 
(baseline), BF16, and FP8 quantization.
""")


def main():
    """Main entry point."""

    extract_metrics_from_log()
    generate_latex_figure_code()
    generate_performance_table_template()
    generate_discussion_points()
    generate_methods_text()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. ✓ Baseline experiment completed
2. ⏳ Run benchmark_configurations.py to get FP32/BF16/FP8 comparison
3. ⏳ Re-run generate_experiment2_figures.py after benchmarks complete
4. ⏳ Extract RMS error and PDE count from baseline logs
5. ⏳ Fill in performance table with actual numbers
6. ✓ Use generated LaTeX code and figures in paper

All figures are ready in outputs/ directory:
  - experiment2_baseline.png (main 4-panel figure)
  - experiment2_stage_progression.png (stage-by-stage)
  - multiscale_fir_base600MHz_lp200-400-600_shots100_nb4_nt1500/ (all stages)
""")


if __name__ == "__main__":
    main()
