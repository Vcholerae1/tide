#!/bin/bash
# Profile script for 3D Maxwell CUDA kernels
# Usage: ./profile_maxwell3d.sh [options]
#
# Requires: NVIDIA Nsight Systems (nsys) and/or Nsight Compute (ncu)

set -e

# Default settings
NZ=${NZ:-128}
NY=${NY:-128}
NX=${NX:-128}
NT=${NT:-50}
SHOTS=${SHOTS:-1}
STENCIL=${STENCIL:-2}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_maxwell3d.py"

echo "=============================================="
echo "3D Maxwell CUDA Profiling"
echo "=============================================="
echo "Grid: ${NZ} x ${NY} x ${NX}"
echo "Time steps: ${NT}"
echo "Shots: ${SHOTS}"
echo "Stencil: ${STENCIL}"
echo "=============================================="

# Check for required tools
check_nsys() {
    if command -v nsys &> /dev/null; then
        echo "Found nsys: $(which nsys)"
        return 0
    else
        echo "WARNING: nsys not found"
        return 1
    fi
}

check_ncu() {
    if command -v ncu &> /dev/null; then
        echo "Found ncu: $(which ncu)"
        return 0
    else
        echo "WARNING: ncu not found"
        return 1
    fi
}

# Profile with Nsight Systems (timeline)
profile_nsys() {
    echo ""
    echo "=== Nsight Systems Timeline Profile ==="
    local output_name="maxwell3d_nsys_${NZ}x${NY}x${NX}_nt${NT}"

    nsys profile \
        --output="${output_name}" \
        --force-overwrite=true \
        --trace=cuda,nvtx \
        --cuda-memory-usage=true \
        python3 "${BENCHMARK_SCRIPT}" \
            --nz ${NZ} --ny ${NY} --nx ${NX} \
            --nt ${NT} \
            --shots ${SHOTS} \
            --stencil ${STENCIL} \
            --warmup 1 \
            --iters 1

    echo "Profile saved: ${output_name}.nsys-rep"
    echo "View with: nsys-ui ${output_name}.nsys-rep"
}

# Profile with Nsight Compute (kernel metrics)
profile_ncu() {
    echo ""
    echo "=== Nsight Compute Kernel Profile ==="
    local output_name="maxwell3d_ncu_${NZ}x${NY}x${NX}_nt${NT}"

    # Use smaller grid and fewer timesteps for detailed kernel analysis
    local NCU_NZ=$((NZ < 64 ? NZ : 64))
    local NCU_NY=$((NY < 64 ? NY : 64))
    local NCU_NX=$((NX < 64 ? NX : 64))
    local NCU_NT=$((NT < 10 ? NT : 10))

    ncu \
        --output="${output_name}" \
        --force-overwrite \
        --set=full \
        --kernel-name-base=function \
        --launch-skip=2 \
        --launch-count=4 \
        python3 "${BENCHMARK_SCRIPT}" \
            --nz ${NCU_NZ} --ny ${NCU_NY} --nx ${NCU_NX} \
            --nt ${NCU_NT} \
            --shots ${SHOTS} \
            --stencil ${STENCIL} \
            --warmup 1 \
            --iters 1

    echo "Profile saved: ${output_name}.ncu-rep"
    echo "View with: ncu-ui ${output_name}.ncu-rep"
}

# Quick performance summary with CUDA events
quick_profile() {
    echo ""
    echo "=== Quick Performance Summary ==="
    python3 "${BENCHMARK_SCRIPT}" \
        --nz ${NZ} --ny ${NY} --nx ${NX} \
        --nt ${NT} \
        --shots ${SHOTS} \
        --stencil ${STENCIL} \
        --warmup 3 \
        --iters 5
}

# Print roofline analysis guidance
print_roofline_guidance() {
    echo ""
    echo "=== Roofline Analysis Guidance ==="
    echo ""
    echo "For 3D Maxwell FDTD, key performance metrics:"
    echo ""
    echo "1. Memory Bandwidth Bound Analysis:"
    echo "   - Each cell update reads 6 fields (E, H) from neighbors"
    echo "   - With stencil order N, reads ~12*N values per cell"
    echo "   - Writes 6 field values per cell"
    echo "   - Arithmetic intensity: ~20-40 FLOP/byte (depends on stencil)"
    echo ""
    echo "2. Expected Bottlenecks:"
    echo "   - L2 cache bandwidth for interior regions"
    echo "   - Global memory bandwidth for large grids"
    echo "   - Shared memory bank conflicts (32-bank, 4-byte)"
    echo ""
    echo "3. Key Metrics to Check in NCU:"
    echo "   - sm__throughput.avg_pct (SM utilization)"
    echo "   - gpu__compute_memory_throughput.avg_pct"
    echo "   - l2__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum"
    echo "   - smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ld.sum"
    echo ""
}

# Main
main() {
    case "${1:-quick}" in
        quick)
            quick_profile
            ;;
        nsys)
            if check_nsys; then
                profile_nsys
            fi
            ;;
        ncu)
            if check_ncu; then
                profile_ncu
            fi
            ;;
        full)
            quick_profile
            if check_nsys; then
                profile_nsys
            fi
            if check_ncu; then
                profile_ncu
            fi
            print_roofline_guidance
            ;;
        roofline)
            print_roofline_guidance
            ;;
        *)
            echo "Usage: $0 [quick|nsys|ncu|full|roofline]"
            echo "  quick    - Run benchmark with timing (default)"
            echo "  nsys     - Profile with Nsight Systems"
            echo "  ncu      - Profile with Nsight Compute"
            echo "  full     - Run all profiling"
            echo "  roofline - Print roofline analysis guidance"
            exit 1
            ;;
    esac
}

main "$@"
