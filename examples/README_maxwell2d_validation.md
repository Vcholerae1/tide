# 2D TM Maxwell Solver Validation Against Analytical Solution

## 概述

此示例验证了 2D TM (Transverse Magnetic) 模式的 Maxwell 方程时域求解器，通过与解析 Green 函数解进行对比。我们在均匀介质中使用紧凑源时间函数，并在多个接收器偏移处记录 E 场迹线。

## 理论背景

### 2D TM 模式解析解

对于均匀介质中的线电流源，2D TM 模式的 Green 函数使用第二类零阶 Hankel 函数表示：

$$G(\mathbf{r}, \omega) = -\frac{j\omega\mu_0}{4} H_0^{(2)}(k R)$$

其中：
- $k = \omega\sqrt{\mu_0 \varepsilon_c}$ 是波数
- $\varepsilon_c = \varepsilon - j\sigma/\omega$ 是复介电常数
- $R$ 是源-接收距离
- $H_0^{(2)}$ 是第二类零阶 Hankel 函数

### 时域解

时域场通过频域 Green 函数的逆傅里叶变换获得：

$$E_z(t) = \mathcal{F}^{-1}\left[\hat{I}(\omega) \cdot G(\omega)\right]$$

## 评估指标

我们评估以下三个方面：

1. **波形一致性**：接收点处的时间序列叠加对比
2. **相对 $\ell_2$ 误差**：时间窗口内的相对误差
   $$\text{Error} = \frac{\|E_{\text{sim}} - \alpha E_{\text{ana}}\|}{\|E_{\text{ana}}\|}$$
   其中 $\alpha$ 是最小二乘拟合的振幅缩放因子
3. **网格/时间离散化敏感性**：CFL 和色散趋势

## 结果

### 图表说明

生成的验证图包含以下子图：

- **(a) 多个接收器偏移处的模拟与解析迹线**：展示不同距离处波形的时域对比
- **(b) 相对误差随时间变化**：显示累积相对 $\ell_2$ 误差的演化
- **(c) 误差与距离的关系**：说明误差如何随源-接收距离增加
- **(d) 最远接收器的详细对比**：放大显示最具挑战性位置的波形匹配
- **(e) 残差误差**：模拟与解析解的差值

### 验证结果

从生成的图表可以看出：

- ✅ 在近场（50 mm）相对误差约 1.7%
- ⚠️ 在远场（200 mm）相对误差约 10%
- ✅ 峰值时间对齐良好（≤2 个时间步偏移）
- ✅ 波形形状吻合度高

误差随距离增加主要由于：
1. 数值色散累积
2. PML 边界反射
3. 网格离散化效应

## 使用方法

### 基本运行

```bash
# 使用默认参数
uv run python examples/validate_maxwell2d_analytic.py

# 指定输出文件
uv run python examples/validate_maxwell2d_analytic.py --output my_validation.png
```

### 参数调整

```bash
# 更高频率
uv run python examples/validate_maxwell2d_analytic.py --freq0 2e9

# 更细网格（提高精度）
uv run python examples/validate_maxwell2d_analytic.py --dx 0.002

# 更小时间步长
uv run python examples/validate_maxwell2d_analytic.py --dt 5e-12

# 不同介质参数
uv run python examples/validate_maxwell2d_analytic.py --eps-r 4.0 --conductivity 0.01
```

### 在 CPU 上运行

```bash
uv run python examples/validate_maxwell2d_analytic.py --no-cuda
```

## 参数说明

- `--freq0`: 源中心频率 (Hz)，默认 9e8 (900 MHz)
- `--dt`: 时间步长 (s)，默认 1e-11 (10 ps)
- `--nt`: 时间步数，默认 800
- `--dx`: 网格间距 (m)，默认 0.005 (5 mm)
- `--eps-r`: 相对介电常数，默认 10.0
- `--conductivity`: 电导率 (S/m)，默认 1e-3
- `--output`: 输出图片文件名
- `--no-cuda`: 强制使用 CPU

## 提高精度的建议

要获得更好的验证结果，可以：

1. **减小网格间距**：使用 `--dx 0.002`（2 mm）或更小
2. **增加 PML 宽度**：修改代码中的 `pml_width=10` 为更大值
3. **使用更高阶模板**：修改 `stencil=2` 为 4 或 8
4. **降低源频率**：使用 `--freq0 5e8` 以增加每波长网格点数
5. **延长模拟时间**：使用 `--nt 1200` 以捕获更长时间的演化

## 物理常数

脚本使用以下物理常数：

- $\varepsilon_0 = \frac{1}{36\pi} \times 10^{-9}$ F/m （真空介电常数）
- $\mu_0 = 4\pi \times 10^{-7}$ H/m （真空磁导率）
- $c_0 = \frac{1}{\sqrt{\varepsilon_0\mu_0}}$ m/s （光速）

## 参考文献

1. Chew, W. C. (1995). *Waves and Fields in Inhomogeneous Media*. IEEE Press.
2. Taflove, A., & Hagness, S. C. (2005). *Computational Electrodynamics: The Finite-Difference Time-Domain Method* (3rd ed.). Artech House.

## 相关示例

- `benchmark_maxwell2d.py`: 2D Maxwell 求解器性能基准测试
- `example_maxwell3d_analytic.py`: 3D Maxwell 求解器解析验证（使用球面波）
- `test_maxwell_analytic.py`: 单元测试版本（在 tests/ 目录）
