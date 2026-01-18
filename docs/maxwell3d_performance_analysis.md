# 3D Maxwell FDTD Performance Analysis

## Test Environment
- **GPU**: NVIDIA GeForce RTX 4070
- **Peak Memory Bandwidth**: 504 GB/s
- **CUDA Compute Capability**: 8.9
- **PyTorch**: 2.9.1+cu128

## Performance Summary

### Grid Size vs Performance

| Grid Size | Cells | Time/Step (ms) | Cell Updates/s | Bandwidth (GB/s) | Efficiency |
|-----------|-------|----------------|----------------|------------------|------------|
| 64³ | 262K | 0.64 | 4.1e8 | 18 | 3.6% |
| 128³ | 2.1M | 1.68 | 1.25e9 | 56 | 11.1% |
| 192³ | 7.1M | 4.20 | 1.69e9 | 75 | 14.9% |
| 256³ | 16.8M | 8.10 | 2.07e9 | 93 | 18.5% |

*Efficiency = Achieved Bandwidth / Peak Bandwidth*

### Key Findings

1. **Memory Bandwidth Limited**: Performance scales linearly with grid size, indicating memory bandwidth is the bottleneck, not compute.

2. **Larger Grids Better**: Larger grids achieve better bandwidth utilization due to:
   - Better amortization of kernel launch overhead
   - More data reuse in L2 cache
   - Higher GPU occupancy

3. **Stencil Order Impact**:
   - 2nd and 4th order stencils have similar performance
   - 8th order is ~15-20% slower due to increased shared memory usage

## Kernel Analysis

### Forward Kernel Time Distribution (from PyTorch Profiler)

| Kernel | Time (ms) | % of Total | Per-Call (μs) |
|--------|-----------|------------|---------------|
| forward_kernel_h_3d | 20.9 | 47.1% | 697 |
| forward_kernel_e_3d | 21.0 | 47.3% | 699 |
| Other (fill, mul, etc.) | 2.5 | 5.6% | - |

### Memory Access Pattern

Each time step requires:
- **Read**: 6 field components × (1 + 2×FD_PAD) neighbors = ~18-42 values per cell
- **Write**: 6 field components = 6 values per cell
- **PML regions**: Additional 12 memory variables

For 128³ grid with stencil=2:
- Read: 2.1M cells × 18 values × 4 bytes = 151 MB
- Write: 2.1M cells × 6 values × 4 bytes = 50 MB
- Total: ~201 MB/step

Theoretical minimum time: 201 MB / 504 GB/s = 0.40 ms
Actual time: 1.68 ms
Memory efficiency: 24%

## Optimization Recommendations

### 1. Block Size Tuning (Implemented)

The optimal block size varies with grid size:
- **Small grids (<100³)**: 32×4 or 32×8 threads
- **Medium grids (100³-300³)**: 32×8 threads (default)
- **Large grids (>300³)**: 32×16 threads

Environment variables `TIDE_BLOCK_X` and `TIDE_BLOCK_Y` allow runtime tuning.

### 2. Further Optimization Opportunities

#### A. Vectorized Memory Access
Use `float4` for coalesced reads when nx is multiple of 4:
```cuda
// Instead of:
float val = field[idx];
// Use:
float4 vec = *reinterpret_cast<float4*>(&field[idx]);
```

#### B. Kernel Fusion
Fuse H and E update kernels to reduce memory traffic:
- Current: H kernel reads E, writes H; E kernel reads H, writes E
- Fused: Single kernel does both, keeping values in registers

#### C. PML Separation
Split kernels into PML and non-PML versions to eliminate branch divergence.

#### D. Persistent Thread Block
For small grids, use persistent thread blocks to reduce launch overhead.

### 3. Fused H+E Kernel

A fused kernel using CUDA Cooperative Groups combines H and E field updates in a single kernel launch, reducing kernel launch overhead.

**Status**: ✅ Working - enabled via `TIDE_FUSED_KERNEL=1`

**Performance improvement** (RTX 4070):

| Grid Size | Default | Fused | Speedup |
|-----------|---------|-------|---------|
| 48³ | 0.44 ms | 0.33 ms | **25%** |
| 64³ | 0.44 ms | 0.33 ms | **25%** |
| 80³ | 0.88 ms | 0.72 ms | **18%** |
| 96³ | 1.11 ms | 1.10 ms | 1% |
| 112³ | 1.43 ms | 1.30 ms | 9% |
| 128³ | 1.66 ms | 1.51 ms | **10%** |
| 160³ | 2.59 ms | 2.35 ms | **10%** |
| 192³+ | N/A | Falls back to default (grid too large) |

**Design considerations for gradient efficiency**:
- Snapshot storage occurs during E update phase (after global sync)
- `ex_store`, `ey_store`, `ez_store`: Store E values before update
- `curl_store`: Store curl(H) values computed during E update
- This preserves correct backward pass computation

**To enable**:
```bash
export TIDE_FUSED_KERNEL=1
```

**Limitations**:
- Requires GPU with compute capability >= 6.0
- Grid must fit in available SMs (automatically falls back if too large)
- Best for small-to-medium grids (≤160³)

### 4. Hardware-Specific Tuning

For different GPU architectures:
- **Ampere (RTX 30xx)**: 32×8 block, consider async copy
- **Ada Lovelace (RTX 40xx)**: 32×8 block, L2 cache optimization
- **Hopper (H100)**: 32×16 block, TMA for memory loads

## Roofline Analysis

```
                          Compute Bound
                               |
                               v
    +--------------------------+------------------+
    |                          |                  |
    |  Memory Bound            |    xxxxxxxxxx    |
    |     xxxxx                |                  |
    |   xx                     |                  |
    +-xx-----------------------+------------------+
     ^                         ^
   Current                   Peak
   (~20 FLOP/byte)         (compute)
```

The 3D Maxwell FDTD kernel has arithmetic intensity of ~20-40 FLOP/byte,
placing it in the memory-bound region of the roofline.

## Benchmark Commands

```bash
# Basic benchmark
python examples/benchmark_maxwell3d.py --nz 256 --ny 256 --nx 256 --nt 100

# With block size tuning
TIDE_BLOCK_X=32 TIDE_BLOCK_Y=16 python examples/benchmark_maxwell3d.py ...

# Profile with PyTorch
python examples/profile_maxwell3d_detailed.py --nz 128 --ny 128 --nx 128

# Profile with NVIDIA Nsight
nsys profile -o profile python examples/benchmark_maxwell3d.py ...
```
