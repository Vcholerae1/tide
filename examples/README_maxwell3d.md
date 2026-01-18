# 3D Maxwell方程正演示例

本目录包含两个三维Maxwell方程FDTD（有限差分时域）正演示例，用于演示如何使用`tide.Maxwell3D`进行三维电磁波传播模拟并与解析解对比。

## 示例文件

### 1. `example_maxwell3d_simple.py` (推荐)

**简化版三维正演示例** - 适合快速入门和理解基本概念。

**特点：**
- 使用小网格（16×16×16）进行快速计算
- Python后端保证数值稳定性
- 简化的解析解用于快速验证
- 不使用PML边界
- 相对误差约10%，峰值时间匹配良好

**使用方法：**
```bash
python example_maxwell3d_simple.py [--plot]
```

**参数：**
- `--plot`: 生成对比图

**示例输出：**
```
Simulation parameters:
  Grid: 16 × 16 × 16 = 0.16m × 0.16m × 0.16m
  Grid spacing: 10.0 mm
  Time steps: 60
  Source frequency: 2.0 GHz
  Distance: 40.0 mm

Comparison results:
  Relative L2 error: 9.58%
  Peak shift: 0 samples
  ✓ Test passed: Reasonable agreement with analytical solution
```

---

### 2. `example_maxwell3d_analytic.py`

**完整版三维正演示例** - 包含更精确的解析解和更多参数选项。

**特点：**
- 可配置的网格大小和参数
- 使用频域Green函数的精确解析解
- 支持PML吸收边界
- 支持不同介质参数（介电常数、电导率）
- 详细的误差分析

**使用方法：**
```bash
python example_maxwell3d_analytic.py [选项]
```

**可用选项：**
```
--nz SIZE          z方向网格点数 (默认: 64)
--ny SIZE          y方向网格点数 (默认: 64)
--nx SIZE          x方向网格点数 (默认: 64)
--nt STEPS         时间步数 (默认: 400)
--pml-width WIDTH  PML层厚度 (默认: 10)
--dx SPACING       网格间距，单位m (默认: 0.01)
--freq FREQUENCY   源频率，单位Hz (默认: 1e9)
--eps-r VALUE      相对介电常数 (默认: 4.0)
--sigma VALUE      电导率，单位S/m (默认: 0.0)
--receiver-offset  接收器偏移，单位网格 (默认: 15)
--stencil ORDER    模板精度 (2, 4, 6, 8) (默认: 2)
--device DEVICE    计算设备 (cuda/cpu, 默认: 自动)
--plot             生成对比图
```

**示例：**
```bash
# 小规模快速测试
python example_maxwell3d_analytic.py --nz 32 --ny 32 --nx 32 --nt 150 --pml-width 8

# 更精确的模拟（需要更多计算资源）
python example_maxwell3d_analytic.py --nz 64 --ny 64 --nx 64 --nt 400 --stencil 4 --plot
```

---

## 关键概念

### 3D Maxwell方程

完整的3D Maxwell方程包括6个场分量：
- 电场：Ex, Ey, Ez
- 磁场：Hx, Hy, Hz

### CFL稳定性条件

对于3D FDTD，时间步长必须满足：

```
dt <= dx / (c * sqrt(3))
```

其中`c`是介质中的光速，`sqrt(3)`来自三维情况。

示例代码会自动计算满足CFL条件的时间步长。

### 解析解

**简化解析解** (`example_maxwell3d_simple.py`):
- 基于球面波1/r衰减和时间延迟
- 适合快速验证和教学

**精确解析解** (`example_maxwell3d_analytic.py`):
- 使用频域Green函数方法
- 考虑色散和损耗
- 更适合定量对比

### PML边界条件

PML (Perfectly Matched Layer) 是一种吸收边界条件，用于模拟无限大空间：
- 推荐厚度：8-15个网格点
- 会增加计算量和内存使用
- 对于小网格测试，可以设置为0

---

## 已知问题和注意事项

### 1. CUDA后端数值稳定性

当前CUDA后端在某些参数组合下可能出现数值不稳定（特别是使用PML时）。建议：
- 对于生产用途，使用Python后端：`python_backend=True`
- 确保CFL条件满足，避免内部时间步细分
- 使用较小的dt值

### 2. 接收器采样问题

当CFL条件需要内部时间步细分时（警告："CFL condition requires N internal time steps"），接收器数据的采样可能出现问题。解决方法：
- 使用更小的dt值，避免细分
- 或确保接收器在合理的位置

### 3. 内存使用

三维FDTD模拟的内存使用量与`nz × ny × nx`成正比：
- 64³网格 ≈ 几十MB
- 128³网格 ≈ 几百MB
- 256³网格 ≈ 几GB

---

## 性能建议

1. **快速测试**：使用小网格（16-32）和短时间（50-100步）
2. **精度要求**：增加网格密度或使用更高阶模板（stencil=4或6）
3. **并行计算**：对于大规模模拟，考虑使用CUDA后端（待稳定性改进）

---

## 参考资料

1. Taflove, A., & Hagness, S. C. (2005). *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed.
2. Berenger, J. P. (1994). "A perfectly matched layer for the absorption of electromagnetic waves." *Journal of Computational Physics*, 114(2), 185-200.

---

## 对比二维示例

如果你需要二维TM模式的示例，请参考：
- `example_multiscale_filtered.py` - 二维多尺度反演示例
- `wavefield_animation.py` - 二维波场动画示例
- `benchmark_maxwell.py` - 二维性能基准测试

三维示例与二维示例的主要区别：
- 三维需要6个场分量（vs 二维TM的3个）
- 三维的CFL条件更严格（sqrt(3) vs sqrt(2)）
- 三维的计算和内存成本显著增加

---

## 故障排除

**问题：接收器数据包含NaN**
- 检查CFL条件警告
- 尝试减小dt或增大网格间距dx
- 使用Python后端：`python_backend=True`
- 减小PML宽度或设为0

**问题：数值爆炸（非常大的值）**
- CFL条件不满足
- 源强度过大
- PML参数不当

**问题：解析解和数值解误差很大**
- 网格可能太粗糙（增加网格密度）
- 边界反射（增加PML宽度或使用更大的计算域）
- 数值色散（使用更高阶模板）

---

## 贡献

如果你发现示例中的问题或有改进建议，欢迎提交Issue或Pull Request。
