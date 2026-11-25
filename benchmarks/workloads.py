import argparse
from collections.abc import Callable
from dataclasses import dataclass
import gc
import json
from statistics import median
from time import perf_counter_ns
from typing import Any

import numpy as np

@dataclass(frozen=True)
class Backend:
    version: str
    tensor: Callable[..., Any]
    relu: Callable[[Any], Any]
    sum_axis: Callable[[Any, int], Any]
    mean: Callable[[Any], Any]
    to_numpy: Callable[[Any], np.ndarray]
    backward: Callable[..., tuple[Any, ...]]

@dataclass(frozen=True)
class Workload:
    name: str
    shape: str
    repeats: int
    run: Callable[[], tuple[Any, ...]]
    expected: tuple[np.ndarray, ...]

def _backward(loss_fn, x, w):
    x.grad = None
    w.grad = None

    loss = loss_fn()
    loss.backward()
    return loss, x.grad, w.grad

def _load_backend(framework, threads):
    if framework == "depthtensor":
        import depthtensor as dt

        return Backend(
            dt.__version__,
            dt.Tensor,
            dt.relu,
            lambda x, axis: x.sum(axis=axis),
            lambda x: x.mean(),
            lambda x: np.asarray(x.data if isinstance(x, dt.Tensor) else x),
            _backward,
        )

    if framework == "torch":
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)

        def tensor(data, requires_grad=False):
            return torch.from_numpy(data).requires_grad_(requires_grad)

        return Backend(
            torch.__version__,
            tensor,
            torch.relu,
            lambda x, axis: x.sum(dim=axis),
            torch.mean,
            lambda x: x.detach().numpy(),
            _backward,
        )

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.set_synchronous_execution(True)

    def tensor(data, requires_grad=False):
        return tf.convert_to_tensor(data)

    def backward(loss_fn, x, w):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch((x, w))
            loss = loss_fn()

        dx, dw = tape.gradient(loss, (x, w))
        return loss, dx, dw

    return Backend(
        tf.__version__,
        tensor,
        tf.nn.relu,
        tf.reduce_sum,
        tf.reduce_mean,
        lambda x: x.numpy(),
        backward,
    )

def _workloads(backend):
    rng = np.random.default_rng(42)

    a, b = rng.standard_normal((2, 1024), dtype=np.float32)
    vector_a, vector_b = backend.tensor(a), backend.tensor(b)
    yield Workload("Vector add", "1,024 each", 200, lambda: (vector_a + vector_b,), (a + b,))

    a = rng.standard_normal((1024, 1024), dtype=np.float32)
    b = rng.standard_normal(1024, dtype=np.float32)
    matrix, row = backend.tensor(a), backend.tensor(b)
    yield Workload("Broadcast add", "1,024 × 1,024 + 1,024", 30, lambda: (matrix + row,), (a + b,))

    yield Workload("ReLU", "1,024 × 1,024", 30, lambda: (backend.relu(matrix),), (np.maximum(a, 0),))

    yield Workload("Row sum", "1,024 × 1,024", 30, lambda: (backend.sum_axis(matrix, 1),), (a.sum(axis=1, dtype=np.float64),))

    a, b = rng.standard_normal((2, 512, 512), dtype=np.float32)
    left, right = backend.tensor(a), backend.tensor(b)
    yield Workload("Matmul", "512 × 512 each", 15, lambda: (left @ right,), (a.astype(np.float64) @ b,))

    inputs = rng.standard_normal((256, 128), dtype=np.float32)
    weights = rng.standard_normal((128, 128), dtype=np.float32)
    labels = rng.standard_normal((256, 128), dtype=np.float32)
    x = backend.tensor(inputs, requires_grad=True)
    w = backend.tensor(weights, requires_grad=True)
    target = backend.tensor(labels)

    def loss():
        error = backend.relu(x @ w) - target
        return backend.mean(error * error)

    x64, w64 = inputs.astype(np.float64), weights.astype(np.float64)
    activation = x64 @ w64
    error = np.maximum(activation, 0) - labels
    gradient = (2.0 / error.size) * error * (activation > 0)
    expected = (np.mean(error**2), gradient @ w64.T, x64.T @ gradient)

    yield Workload(
        "Matmul → ReLU → MSE + backward",
        "256 × 128 @ 128 × 128",
        10,
        lambda: backend.backward(loss, x, w),
        expected,
    )

def _validate(workload, backend):
    for value, expected in zip(workload.run(), workload.expected, strict=True):
        array = backend.to_numpy(value)
        if array.shape != expected.shape or array.dtype != np.float32:
            raise ValueError(f"{workload.name}: unexpected result shape or dtype")

        np.testing.assert_allclose(array, expected, rtol=2e-4, atol=5e-5, err_msg=workload.name)

def _measure(workload, backend, warmups, samples):
    _validate(workload, backend)

    for _ in range(warmups):
        workload.run()

    elapsed = []
    for _ in range(samples):
        gc.collect()

        start = perf_counter_ns()
        for _ in range(workload.repeats):
            workload.run()

        elapsed.append((perf_counter_ns() - start) / (workload.repeats * 1000))

    return {
        "workload": workload.name,
        "shape": workload.shape,
        "median_us": median(elapsed),
        "repeats": workload.repeats,
        "samples_us": elapsed,
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark one framework on CPU.")
    parser.add_argument("framework", choices=["depthtensor", "torch", "tensorflow"])
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    args = parser.parse_args()

    backend = _load_backend(args.framework, args.threads)

    print(json.dumps({
        "framework": args.framework,
        "version": backend.version,
        "numpy": np.__version__,
        "rows": [_measure(w, backend, args.warmups, args.samples) for w in _workloads(backend)],
    }, ensure_ascii=True))

if __name__ == "__main__":
    main()
