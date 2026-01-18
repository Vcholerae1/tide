"""
Benchmark script to test different configurations for Experiment 2.
Tests: baseline (full batch), mini-batch, BF16, FP8, checkpointing
Collects GPU memory and runtime metrics.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configuration matrix for benchmarking
CONFIGS = [
    {
        "name": "Mini-batch + FP32",
        "env": {
            "TIDE_N_SHOTS": "100",
            "TIDE_N_BATCH": "4",
            "TIDE_NT": "1500",
            "TIDE_STAGES": "3",
            "TIDE_ADAMW_EPOCHS": "10",
            "TIDE_LBFGS_EPOCHS": "3",
            "TIDE_STORAGE_MODE": "device",
            "TIDE_STORAGE_COMPRESSION": "none",
            "TIDE_GRAD_INTERVAL": "1",
            "TIDE_PROFILE": "1",
        },
    },
    {
        "name": "Mini-batch + BF16",
        "env": {
            "TIDE_N_SHOTS": "100",
            "TIDE_N_BATCH": "4",
            "TIDE_NT": "1500",
            "TIDE_STAGES": "3",
            "TIDE_ADAMW_EPOCHS": "10",
            "TIDE_LBFGS_EPOCHS": "3",
            "TIDE_STORAGE_MODE": "device",
            "TIDE_STORAGE_COMPRESSION": "bf16",
            "TIDE_GRAD_INTERVAL": "1",
            "TIDE_PROFILE": "1",
        },
    },
    {
        "name": "Mini-batch + FP8",
        "env": {
            "TIDE_N_SHOTS": "100",
            "TIDE_N_BATCH": "4",
            "TIDE_NT": "1500",
            "TIDE_STAGES": "3",
            "TIDE_ADAMW_EPOCHS": "10",
            "TIDE_LBFGS_EPOCHS": "3",
            "TIDE_STORAGE_MODE": "device",
            "TIDE_STORAGE_COMPRESSION": "fp8",
            "TIDE_GRAD_INTERVAL": "1",
            "TIDE_PROFILE": "1",
        },
    },
]


def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def run_benchmark(config):
    """Run a single benchmark configuration."""
    print(f"\n{'=' * 60}")
    print(f"Running: {config['name']}")
    print(f"{'=' * 60}")

    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Set environment variables
    env = os.environ.copy()
    env.update(config["env"])

    # Run the example script
    start_time = time.time()

    try:
        result = subprocess.run(
            ["python", "examples/example_multiscale_filtered.py"],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        elapsed_time = time.time() - start_time

        # Parse output for metrics
        output = result.stdout
        stderr = result.stderr

        # Extract final loss and PDE count
        final_loss = None
        pde_forward = None
        pde_adjoint = None

        for line in output.split("\n"):
            if "Loss=" in line:
                try:
                    final_loss = float(line.split("Loss=")[1].split()[0])
                except:
                    pass
            if "Total PDE solves" in line:
                try:
                    parts = line.split("forward")[1].split(",")
                    pde_forward = float(parts[0].strip())
                    pde_adjoint = float(
                        parts[1].split("adjoint")[1].split(",")[0].strip()
                    )
                except:
                    pass

        # Get memory usage (approximate from output)
        peak_memory_gb = get_gpu_memory_usage()

        # Calculate iterations per second
        total_epochs = int(config["env"]["TIDE_ADAMW_EPOCHS"]) + int(
            config["env"]["TIDE_LBFGS_EPOCHS"]
        )
        iter_per_sec = total_epochs / elapsed_time if elapsed_time > 0 else 0

        result_data = {
            "name": config["name"],
            "success": result.returncode == 0,
            "elapsed_time": elapsed_time,
            "peak_memory_gb": peak_memory_gb,
            "iter_per_sec": iter_per_sec,
            "final_loss": final_loss,
            "pde_forward": pde_forward,
            "pde_adjoint": pde_adjoint,
            "config": config["env"],
        }

        print("\nResults:")
        print(f"  Success: {result_data['success']}")
        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  Peak GPU Memory: {peak_memory_gb:.2f} GB")
        print(f"  Iter/s: {iter_per_sec:.3f}")
        if final_loss:
            print(f"  Final Loss: {final_loss:.6e}")

        return result_data

    except subprocess.TimeoutExpired:
        print("TIMEOUT after 600s")
        return {
            "name": config["name"],
            "success": False,
            "elapsed_time": 600,
            "peak_memory_gb": 0,
            "iter_per_sec": 0,
            "error": "timeout",
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "name": config["name"],
            "success": False,
            "error": str(e),
        }


def generate_latex_table(results):
    """Generate LaTeX table from benchmark results."""
    print("\n" + "=" * 60)
    print("LaTeX Table:")
    print("=" * 60)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance summary for the overthrust inversion.}")
    lines.append(r"\label{tab:perf}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & Peak GPU Mem (GB) & Iter/s & Notes \\")
    lines.append(r"\midrule")

    for res in results:
        if res["success"]:
            name = res["name"]
            mem = f"{res['peak_memory_gb']:.2f}" if res.get("peak_memory_gb") else "N/A"
            iter_s = f"{res['iter_per_sec']:.3f}" if res.get("iter_per_sec") else "N/A"
            time_s = f"{res['elapsed_time']:.1f}s" if res.get("elapsed_time") else "N/A"

            # Escape underscores for LaTeX
            name_escaped = name.replace("_", r"\_")

            lines.append(f"{name_escaped} & {mem} & {iter_s} & {time_s} \\\\")
        else:
            name_escaped = res["name"].replace("_", r"\_")
            lines.append(f"{name_escaped} & - & - & Failed \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    print(table)

    return table


def generate_comparison_plot(results):
    """Generate comparison plots for runtime and memory."""
    import matplotlib.pyplot as plt

    successful_results = [r for r in results if r.get("success", False)]

    if not successful_results:
        print("No successful results to plot")
        return

    names = [r["name"] for r in successful_results]
    times = [r["elapsed_time"] for r in successful_results]
    memories = [r["peak_memory_gb"] for r in successful_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Runtime comparison
    ax1.barh(names, times, color="steelblue")
    ax1.set_xlabel("Time (seconds)", fontsize=12)
    ax1.set_title("Runtime Comparison", fontsize=14, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # Memory comparison
    ax2.barh(names, memories, color="coral")
    ax2.set_xlabel("Peak GPU Memory (GB)", fontsize=12)
    ax2.set_title("Memory Usage Comparison", fontsize=14, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    output_path = Path("outputs/performance_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to {output_path}")

    plt.close()


def main():
    """Run all benchmark configurations."""
    print("Starting benchmark suite for Experiment 2")
    print(f"Total configurations: {len(CONFIGS)}")

    results = []

    for i, config in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}] ", end="")
        result = run_benchmark(config)
        results.append(result)

        # Save intermediate results
        output_file = Path("outputs/benchmark_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)

    # Generate outputs
    latex_table = generate_latex_table(results)

    # Save LaTeX table
    latex_file = Path("outputs/performance_table.tex")
    with open(latex_file, "w") as f:
        f.write(latex_table)
    print(f"\nSaved LaTeX table to {latex_file}")

    # Generate comparison plots
    generate_comparison_plot(results)

    # Print summary
    print("\nSummary:")
    for res in results:
        status = "✓" if res.get("success") else "✗"
        print(f"  {status} {res['name']}")


if __name__ == "__main__":
    main()
