# DepthTensor: A Hardware-Accelerated Tensor Computation and Autograd Engine

**DepthTensor** is a light-weight, high-performance library for reverse-mode automatic differentiation (AD). It is useful in building the mathematical foundation of deep learning frameworks, with the use of a `Tensor` object which dynamically builds computational graphs and computes gradients using **Vector-Jacobian Products (VJP)**, generalized for tensors of arbitrary rank.

> **Note**: This is the core autograd and tensor computation engine. For the full deep learning framework, refer to [DepthML](https://github.com/l-h-ha/DepthML).

## 1. Mathematical Foundation

The goal is to compute the gradient of scalar field $L$ with respect to an input tensor $X$, denoted as the adjoint $\bar{X}$.

### 1.1. Generalized VJPs via Tensor Contractions

Let $X$ be a tensor of shape $\mathcal{I} = (i_1, \dots, i_n)$ and $Y = f(X)$ be a tensor of shape $\mathcal{J} = (j_1, \dots, j_m)$.

Mathematically, the Jacobian is the tensor of rank $n + m$ which contains all the partial derivatives $\frac{\partial Y_{\mathcal{J}}}{\partial X_{\mathcal{I}}}$. DepthTensor does not compute this object, but, rather, it computes the contraction of the incoming adjoint $\bar{Y}$ with the local derivative:

$$
\bar{X}_{i_1 \dots i_n} = \sum_{j_1 \dots j_m} \bar{Y}_{j_1 \dots j_m} \frac{\partial Y_{j_1 \dots j_m}}{\partial X_{i_1 \dots i_n}}
$$

If X and Y are scalars, indices vanish, and this reduces to:

$$
\bar{x} = \bar{y} \cdot f'(x)
$$

If X and Y are matrices, this reduces to:

$$
\bar{X} = \bar{Y} W^{T}
$$

### 1.2. Graph Topology and Gradient Accumulation

Gradients are accumulated using a Depth-First Search (DFS) topological sort, which ensures that for any node $X$ with multiple gradient flow streams ${Y^{(1)}, \dots, Y^{k}}$, the gradient sum is the sum of contractions:

$$
\bar{X} = \sum_{k} \mathrm{VJP}(Y^{(k)}, X)
$$

All gradients are aggregated from all gradient downstreams.

## 2. Architecture and System Design

### 2.1. The Differentiable Primitive (`Tensor`)

The `Tensor` class acts like a node in the Directed Acyclic Graph (DAG).

Operations are automatically dispatched to backend computers, which consists of `numpy` (CPU) and `cupy` (GPU).

Computational graphs are built dynamically at runtime.

## 3. Empirical Validation

We will verify the tensor engine by minimizing the Rosenbrock function, which is a non-convex optimization benchmark:

$$
f(x, y) = (a - x)^2 + b(y - x^2)^2
$$

```python
import depthtensor as dt

# Initialize tensors
x = dt.Tensor([1.2], device="gpu", requires_grad=True)
y = dt.Tensor([1.2], device="gpu", requires_grad=True)

a, b = dt.Tensor([1], device="gpu"), dt.Tensor(
    [100], device="gpu"
)

# Optimization Loop
lr = 0.001
for i in range(500):
    # Rosenbrock: f(x,y) = (a-x)^2 + b(y-x^2)^2
    loss = (a - x) ** 2 + b * (y - x**2) ** 2

    # Backward pass
    dt.differentiate(loss)

    # Gradient Descent
    x.data -= lr * x.grad  # type: ignore
    y.data -= lr * y.grad  # type: ignore

    # Zero grads
    x.zeros_grad()
    y.zeros_grad()

    if i % 10 == 0:
        print(loss.item())

print(f"Converged: ({x.data}, {y.data})")
# Target: (1.0, 1.0)
```

## 4. Installation

Requirements: 
- `numpy`
- `cupy` (optional: for NVIDIA GPU acceleration)

### 4.1. Bash

```bash
git clone https://github.com/YourUsername/DepthTensor.git
pip install -e .
```

### 4.2. Pip

```bash
pip install depthtensor
```

## Author: Le Hong Ha