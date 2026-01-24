# TIDE

**T**orch-based **I**nversion & **D**evelopment **E**ngine

TIDE is a PyTorch-based library for  high frequa electromagnetic wave propagation and inversion, built on Maxwell's equations. It provides efficient CPU and CUDA implementations for forward modeling, gradient computation, and full waveform inversion (FWI).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Maxwell Equation Solvers**: 
  - 2D TM mode propagation (`MaxwellTM`)
  - Other propagation is on the way 
- **Automatic Differentiation**: Gradient support through PyTorch's autograd
- **High Performance**: Optimized C/CUDA kernels for critical operations
- **Flexible Storage**: Multiple storage modes for gradient computation (memory/disk/optional BF16 compressed)
- **Staggered Grid**: Industry-standard FDTD staggered grid implementation
- **PML Boundaries**: Perfectly Matched Layer absorbing boundaries

## Installation

### From PyPI


```bash
uv pip install tide-GPR
```

or

```bash
pip install tide-GPR
```

### From Source

We recommend using [uv](https://github.com/astral-sh/uv) for building:

```bash
git clone https://github.com/vcholerae1/tide.git
cd tide
uv build
```

**Requirements:**
- Python >= 3.12
- PyTorch >= 2.9.1
- CUDA Toolkit (optional, for GPU support)
- CMake >= 3.28 (optional, for building from source)

## Quick Start

```python
import torch
import tide

# Create a simple model
nx, ny = 200, 100
epsilon = torch.ones(ny, nx) * 4.0  # Relative permittivity
epsilon[50:, :] = 9.0  # Add a layer

# Set up source
source_amplitudes = tide.ricker(
    freq=1e9,           # 1 GHz
    nt=1000,
    dt=1e-11,
    peak_time=5e-10
).reshape(1, 1, -1)

source_locations = torch.tensor([[[10, 100]]])
receiver_locations = torch.tensor([[[10, 150]]])

# Run forward simulation
receiver_data = tide.maxwelltm(
    epsilon=epsilon,
    dx=0.01,
    dt=1e-11,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    pml_width=10
)

print(f"Recorded data shape: {receiver_data.shape}")
```

## Core Modules

- **`tide.maxwelltm`**: 2D TM mode Maxwell solver
- **`tide.wavelets`**: Source wavelet generation (Ricker, etc.)
- **`tide.staggered`**: Staggered grid finite difference operators
- **`tide.callbacks`**: Callback state and factories
- **`tide.resampling`**: Upsampling/downsampling utilities
- **`tide.cfl`**: CFL condition helpers
- **`tide.padding`**: Padding and interior masking helpers
- **`tide.validation`**: Input validation helpers
- **`tide.storage`**: Gradient checkpointing and storage management

## Examples

See the [`examples/`](examples/) directory for complete workflows:

- [`example_multiscale_filtered.py`](examples/example_multiscale_filtered.py): Multi-scale FWI with frequency filtering
- [`example_multiscale_random_sources.py`](examples/example_multiscale_random_sources.py): FWI with random source encoding
- [`wavefield_animation.py`](examples/wavefield_animation.py): Visualize wave propagation

## Documentation

For detailed API documentation and tutorials, visit: [Documentation]() *(coming soon)*

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This project includes code derived from [Deepwave](https://github.com/ar4/deepwave) by Alan Richardson. We gratefully acknowledge the foundational work that made TIDE possible.

## Citation

If you use TIDE in your research, please cite:

```bibtex
@software{tide2025,
  author = {Vcholerae1},
  title = {TIDE: Torch-based Inversion \& Development Engine},
  year = {2025},
  url = {https://github.com/vcholerae1/tide}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



```mermaid
graph TD
    %% --- Styling Definitions ---
    classDef userLayer fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef apiLayer fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#f57f17;
    classDef coreLayer fill:#ffe0b2,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef supportLayer fill:#e8f5e9,stroke:#388e3c,stroke-width:1px,color:#1b5e20;
    classDef nativeLayer fill:#cfd8dc,stroke:#455a64,stroke-width:3px,color:#263238;
    classDef torchFrame fill:none,stroke:#ee3333,stroke-width:2px,stroke-dasharray: 5 5,color:#ee3333;

    %% --- Top Level ---
    UserCode[用户代码 / 脚本<br>FWI, 建模任务]:::userLayer

    subgraph PyTorch_Environment [PyTorch 生态系统 <br> Autograd Engine & Tensor Management]
        style PyTorch_Environment fill:#fff3f3,stroke:#ffcdd2

        %% --- Layer 1: User API ---
        subgraph L1_UserAPI [第1层: 用户 API 层]
            direction LR
            MaxwellTM([MaxwellTM <br> nn.Module 子类]):::apiLayer
            Utils[工具函数 <br> ricker, wavelets, validation]:::apiLayer
        end

        UserCode -->|初始化 & 调用| MaxwellTM
        UserCode -.->|使用| Utils

        %% --- Layer 3: Support (Side by Side) ---
        subgraph L3_Support [第3层: 数值计算与辅助支持]
            Staggered[staggered.py <br> 交错网格/PML定义]:::supportLayer
            Callbacks[callbacks.py <br> 状态监控/观察者]:::supportLayer
            Storage[storage.py <br> 波场快照存储策略<br>CPU/GPU/DISK/Compress]:::supportLayer
        end

        %% --- Layer 2: Core Solver ---
        subgraph L2_Core [第2层: 核心求解器 maxwell.py]
            MaxwellTMForwardFunc{{MaxwellTMForwardFunc <br> autograd.Function}}:::coreLayer
            BackendDispatch[后端分发器 <br> maxwell_func]:::coreLayer
        end

        %% Connections within Python Levels
        MaxwellTM -->|调用 forward| MaxwellTMForwardFunc
        MaxwellTMForwardFunc -->|管理| BackendDispatch
        MaxwellTMForwardFunc -.->|依赖| Storage
        MaxwellTM -.->|配置| Staggered
        MaxwellTM -->|触发| Callbacks

        %% --- Data Flow Arrows ---
        %% Forward Path
        MaxwellTMForwardFunc == "正向传播 (Forward Pass)<br>FDTD 时间步进" ==> BackendDispatch
        BackendDispatch -- 路径 A: Python (调试) --> PyImpl[纯 Python 实现<br>update_E/H]:::supportLayer

        %% Backward Path
        PyTorch_Environment -.- |"loss.backward() <br> 触发自动微分"| MaxwellTMForwardFunc
        BackendDispatch <== "反向传播 (Backward Pass)<br>伴随状态法 / 梯度计算" == MaxwellTMForwardFunc
    end

    %% --- Layer 4: Native Backend ---
    subgraph L4_Native [第4层: C/CUDA 内核层 libtide_C.so]
        direction LR
        CUDA_Kernels[maxwell.cu <br> GPU FDTD 内核]:::nativeLayer
        Born_Kernels[maxwell_born.cu <br> Born 近似内核]:::nativeLayer
        C_Staggered[staggered_grid.h <br> C数据结构]:::nativeLayer
    end

    %% Crossing the boundary
    BackendDispatch -- "路径 B: C/CUDA (高性能)<br>ctypes 调用" --> CUDA_Kernels
    BackendDispatch -- "反向路径调用" --> Born_Kernels
    CUDA_Kernels -.-> C_Staggered
    Born_Kernels -.-> C_Staggered
    Storage -.->|C++实现| CUDA_Kernels

    %% Legend
    subgraph Legend [图例]
        L_Py[Python 模块]:::apiLayer
        L_Core[核心 Autograd]:::coreLayer
        L_Sup[辅助支持]:::supportLayer
        L_Nat[C/CUDA 原生库]:::nativeLayer
        L_Flow_F[正向数据流] ==> L_Flow_F_End
        L_Flow_B[反向数据流] <== L_Flow_B_End
    end
```