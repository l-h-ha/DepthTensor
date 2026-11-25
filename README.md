# DepthTensor

DepthTensor provides broadcasted tensor math, reductions, matrix multiplication, and reverse-mode autodiff. It uses NumPy on CPU and CuPy on CUDA. For neural networks, see [DepthML](https://github.com/l-h-ha/DepthML).

DepthTensor supports real-valued tensors. Autograd requires a floating-point dtype.

## Install

Use Python 3.10 or later. Pip installs NumPy and builds the C++ extension from source. Install a C++ compiler and Python development headers on Linux, or MSVC and the Windows SDK on Windows.

### CPU

```bash
python -m pip install depthtensor
```

### CPU + CUDA

On Linux or Windows, use a supported NVIDIA GPU and a compatible driver. Run `nvidia-smi` to check the CUDA version your driver supports. Choose one install command. [NVIDIA driver check](https://docs.nvidia.com/deploy/nvidia-smi/index.html#cuda-umd-version)

| CUDA | Install |
| --- | --- |
| 12.x | `python -m pip install depthtensor "cupy-cuda12x[ctk]"` |
| 13.x | `python -m pip install depthtensor "cupy-cuda13x[ctk]"` |

`[ctk]` installs CUDA libraries through pip. Remove `[ctk]` to use an installed CUDA Toolkit. Set `device = "cuda"` in the example. [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi)

`python -m pip install "depthtensor[cuda]"` uses CuPy's source package and requires an installed CUDA Toolkit. Use one CuPy package per environment. [CuPy source builds](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-source)

## Build from source

Install from the repository root.

```bash
python -m pip install .
```

Build a wheel in `dist/` with pip.

```bash
python -m pip wheel . --no-deps --wheel-dir dist
```

## Example

Minimize the Rosenbrock function with gradient descent.

```python
import depthtensor as dt

device = "cpu"
x = dt.Tensor([1.2], device=device, requires_grad=True)
y = dt.Tensor([1.2], device=device, requires_grad=True)
a = dt.Tensor([1], device=device)
b = dt.Tensor([100], device=device)

lr = 0.001
for i in range(500):
    loss = (a - x) ** 2 + b * (y - x**2) ** 2
    loss.backward()
    x.data -= lr * x.grad
    y.data -= lr * y.grad
    x.zero_grad()
    y.zero_grad()

    if i % 10 == 0:
        print(loss.item())

print(f"({x.data}, {y.data})")
```

## CPU benchmarks

Intel i5-11400H, Windows 26200, Python 3.11.15.
DepthTensor 3.0.0, PyTorch 2.12.1+cpu, TensorFlow 2.21.0, NumPy 2.3.5.

Rounded median (min–max) in µs/call.

| Tasks | Input shapes | DepthTensor | PyTorch | TensorFlow |
| --- | --- | ---: | ---: | ---: |
| Vector add | `1024 each` | 26 (21–41) | **4** (4–12) | 36 (24–66) |
| Broadcast add | `1024×1024 + 1024` | 1747 (1354–2957) | **481** (312–1024) | 793 (535–932) |
| ReLU | `1024×1024` | 7961 (6033–9598) | **412** (337–620) | 516 (413–1148) |
| Row sum | `1024×1024` | 588 (427–800) | 274 (189–467) | **214** (187–384) |
| Matmul | `512×512 each` | 4192 (3350–5444) | 3210 (3079–3755) | **3077** (2760–4022) |
| Matmul → ReLU → MSE + backward | `256×128 @ 128×128` | 2255 (1858–3681) | **782** (580–911) | 1832 (1531–2672) |

DepthTensor ranked third in all six matmul runs.

Run from the repo root in a Python 3.11 venv.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.12.1+cpu"
python -m pip install . "tensorflow-cpu==2.21.0" "numpy==2.3.5"
python benchmarks/run.py --runs 6
```

## License

[MIT](LICENSE).
