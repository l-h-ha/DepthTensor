# Changelog

Uses [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [SemVer](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-11-25

### Changed

- Replace `dt.differentiate(loss)` with `loss.backward()`.
- Use `device="cuda"` for GPU tensors. Rename the `gpu` extra to `cuda`.
- Reject complex values and dtypes. Autograd accepts floating-point tensors.
- Share arithmetic dispatch and backward setup.
- Separate graph execution, storage tracking, and indexing.
- Publish a source distribution and document the compiler requirements.
- Declare the MIT license with SPDX metadata.

### Fixed

- Fix repeated backward and restore gradients after failures.
- Copy NumPy and CuPy gradient seeds into writable buffers of the tensor dtype.
- Use the operation's compute dtype in backward.
- Traverse deep graphs iteratively.
- Disable gradient tracking when `requires_grad=False`. Reject gradients for integer and Boolean operations.
- Clear history when disabling gradients. Detached copies start as leaves.
- Preserve history and independent gradients in Tensor copies.
- Preserve leaf write protection after Python copy or pickle.
- Reject Python copy and pickle for active graphs. Use `Tensor.copy()`.
- Track writes through views, shared arrays, and `out=`.
- Reject aliased `out=` writes and in-place arithmetic with gradients.
- Reject indexed or ReLU writes that overwrite differentiable results.
- Save forward operands and indices for backward.
- Reject stale views, gathers, and device transfers during backward.
- Preserve history in `to_device(clear_prev=False)`.
- Fix masked `sum()`, `mean()`, and `max()` on CPU and CUDA.
- Split `max()` gradients evenly across tied values.
- Accumulate means before casting to `out=`.
- Update the device after in-place `mean()`.
- Route `where()` gradients across broadcasts. Skip inactive nonlinear derivatives.
- Apply pointwise masks on CUDA and in backward.
- Fix vector and batched matmul gradients.
- Return zero gradients for singleton axes broadcast to empty axes.
- Accumulate repeated indices and accept Tensor indices and values.
- Check CUDA integer indices before reads and writes.
- Fix CUDA `ones_like()` and device overrides for binary operations and `zeros_like()`.
- Fix operators with a NumPy operand on the left. Reject ambiguous tensor truth values.
- Infer devices for scalars and `where()`. Transfer CPU scalar results to CUDA.
- Keep CPU scalar results writable.
- Default `transpose()` to reversed axes and handle negative axes in backward.
- Accept writable C- and Fortran-contiguous buffers in CPU ReLU.
- Preserve CPU ReLU dtypes and write to `out` when supplied. Track writes on both devices.
- Preserve in-place ReLU history and zero inactive gradients.
- Combine successive ReLU masks and preserve them in copies.
- Fix zero-exponent, zero-base, and subnormal power gradients.
- Route reversed and equal clip bounds to the upper bound.
- Use the requested CUDA output layout and apply `clip()` casting rules.
- Stabilize division gradients at extreme scales.
- Fix scalar CPU `uniform()` and preserve its half-open interval after rounding.
- Support CPU seeding and dtype names in `rand()` and `randn()`.
- Package native headers and lowercase module paths. Require Python 3.10+.
- Use public imports and `backward()` in examples.
- Preserve large mean reductions and gradients with float16 outputs.
- Use `requires_grad` in arithmetic. Reject in-place results that track gradients.
- Run backward through chains of history-preserving copies without recursion.
- Preserve large integers when constructing tensors with an explicit dtype.
- Skip inactive nonfinite terms in matrix multiplication gradients.
- Avoid float16 overflow in `square()` and `power()` backward.
- Use input precision for mean gradients with narrow outputs.
- Run backward through mixed copy and in-place ReLU chains without recursion.
- Fix float16 and wide-range float32 CUDA uniform sampling.
- Preserve Boolean ReLU dtype and update the device after an in-place ReLU move.
- Check CUDA bounds for lists of scalar Tensor indices.
- Prevent partial CPU writes when indexed assignment conversion fails.
- Invalidate saved graphs after tensor data is replaced.
- Accumulate broadcast gradients in sufficient precision for the input dtype.
- Apply `sum()` initial values before narrowing CUDA output.
- Accept a one-element output tuple in masked CUDA pointwise operations.
- Preserve uniform sampling across wide float64 ranges.
- Accept disjoint interleaved CPU output buffers.
- Build lowercase wheels from either source directory spelling.
- Widen mixed-precision gradients before arithmetic and matrix multiplication.
- Avoid signed integer exponent wrap in `power()` backward.
- Avoid divisor overflow in CUDA float16 `mean()`.
- Check overlap for the elements written by CPU indexing and for interleaved CUDA outputs.
- Invalidate active results after augmented `.data` writes.
- Restore built-in `abs()` and reject unsupported device names.
- Use a shared working dtype for backward arithmetic and reductions.
- Widen saved floating values for derivatives while preserving forward rounding.
- Keep backward seeds and contributions at working precision until storing `.grad`.
- Save slice bounds, scalar indices, and reduction arguments during forward.
- Use the same code to save Boolean masks for pointwise operations, reductions, and `where()`.
- Update versions and clear incompatible gradients when replacing `.data`.
- Normalize reduction axes, masks, and identity checks before CPU and CUDA dispatch.
- Share `out` checks. Compute into a temporary buffer before writing to `out`.
- Accept singleton output tuples in reductions and preserve exact integer maxima before casting.
- Accept element indices in `Tensor.item()`.
- Keep the existing Ruff rules in CI.
- Attach provenance to the versioned release.

## [2.9.0rc3] - 2026-03-29

### Added

- Add a C++ backend to replace the Python implementation in DepthML.

### Changed

- List public names in `__init__.py`.

- Avoid reusing the outer quote inside f-strings for Python 3.8.

## [2.8.0] - 2026-02-08

### Fixed

- Infer `requires_grad` from the inputs to `Tensor.where()`.

### Added

- Add elementwise `Tensor` methods `sqrt`, `square`, `log`, `exp`, `sign`, `abs`, and `neg`. They support `out`, `device`, `in_place`, and other NumPy/CuPy arguments.
- Add `Tensor.__abs__()`.
- Test tensor creation, properties, arithmetic, reductions, elementwise operations, backward, and in-place operations.

## [2.7.0] - 2026-02-07

### Fixed

- Skip repeated checks in `Tensor._fast_init()`. Lightweight operations were about 4× faster.

- Use `NoValue` to distinguish a missing `initial` from `None` in `sum()` and `max()`.

### Documentation

- Add NumPy-style docstrings to `Tensor` methods, `autodiff.py`, and `ops/`.

## [2.6.0] - 2026-01-11

### Fixed

- Reuse the existing gradient array in `zero_grad()`.

- Reuse operand type checks for device selection and gradient tracking.


### Changed

- ```dt.random.uniform()``` now accepts a ```dtype``` argument, which defaults to ```float32```.

- Show `requires_grad` when printing tensors.


## [2.5.1] - 2026-01-10

### Fixed

- Initialize `prev` before differentiation.

## [2.5.0] - 2026-01-08

### Added

- Add `Tensor.name` for debugging.

- Add `set_name()` to set the name and return the tensor.

- Add `transpose()` with gradient support.

- Support gradients through slicing.

### Changed

- Rename `Tensor.zeros_grad()` to `Tensor.zero_grad()`.

- Raise `RuntimeError` in `Tensor.zero_grad()` when gradient tracking is disabled.

- Use `Function` subclasses for add, subtract, and multiply.

### Removed

- Replace custom unary and binary operation protocols with `Function`.

## [2.4.3] - 2025-12-18

### Fixed

- Initialize gradients automatically during differentiation.

- Fix the default `initial` value in `max()`.

### Changed

- Require `device` in custom binary backward functions.

## [2.4.2] - 2025-12-14

### Fixed

- Add the gradients returned by custom backward functions directly. The previous update multiplied the returned value by the output gradient.

$$
\bar{x} := \bar{y} \odot f(y, x)
$$

Here, $f(y, x)$ is the custom backward function.

## [2.4.1] - 2025-12-14

### Changed

- Rename ```TensorLike, DeviceLike, ScalarLike, ShapeLike, AxisLike, NDArrayLike, NDArrayLikeBool, OperandLike, cfunc_2in_1out_pro, cop_2in_1out_pro, cdiff_2in_1out_pro, cfunc_1in_1out_pro, cop_1in_1out_pro, cdiff_1in_1out_pro``` to ```TensorType, Device, Scalar, Shape, Axis, TensorData, TensorDataBool, TensorLike, BinaryFunc, BinaryOp, BinaryDiff, UnaryFunc, UnaryOp, UnaryDiff```, in that order.

## [2.4.0] - 2025-12-14

### Added

- Add `Tensor.item()` to return an array element as a scalar.

- Add a `grad` argument to `differentiate()` to set the initial gradient.

## [2.3.1] - 2025-12-10

### Fixed

- Allow backward when inputs have gradient tracking disabled.

## [2.3.0] - 2025-12-04

### Added

- Add `Tensor.mean()` with gradient support.

### Fixed

- Fix device selection in backward for unary operations.

## [2.2.1] - 2025-11-26

### Fixed

- Raise `RuntimeError` for in-place operations on tensors that track gradients, including indexed assignment.

## [2.2.0] - 2025-10-5

### Added

- Add `zeros()` and `ones()`.

## [2.1.0] - 2025-09-4

### Added

- Allocate gradients when backward reaches a tensor.

### Changed

- Speed up `DTensor.random.rand()` and `DTensor.random.randn()`.

### Fixed

- Reduce repeated calls to `astype()`.

## [2.0.2] - 2025-09-2

### Fixed

- Keep gradients on the GPU when creating a tensor from a CuPy array without specifying `device`.

## [2.0.1] - 2025-09-2

### Fixed

- Fix device selection in operations such as `square()` and `add()`.

## [2.0.0] - 2025-09-1

### Fixed

- Default `device` to `None` so operations can use the operand devices.

### Changed

- Move operations such as `Tensor.square()` and `Tensor.add()` to the `DepthTensor` module.

## [1.12.0] - 2025-08-29

### Added

- Accept keyword arguments in the backward functions for `create_2in_1out()` and `create_1in_1out()`.

## [1.11.1] - 2025-08-29

### Fixed

- Default `device` to `None` for operations created with `create_2in_1out()` or `create_1in_1out()`.

## [1.11.0] - 2025-08-28

### Fixed

- Reject nonnumeric tensor inputs.

- Keep results of operations such as add and subtract on the GPU when both inputs are on the GPU and `device` is omitted.

### Added

- Add `Tensor.make_differentiable()` to enable gradient tracking in place.

- Add `Tensor.to_device()` to move a tensor in place or return a copy on the requested device.

### Removed

- Remove `AxisShapeLike`. Use `ShapeLike`.

### Changed

- Rename `ArrayLike` to `NDArrayLike` and `ArrayLikeBool` to `NDArrayLikeBool`.

- Default `device` to `None` in more operations.

## [1.10.0] - 2025-08-28

### Added

- Add `DepthTensor.create_2in_1out()` and `DepthTensor.create_1in_1out()` for custom binary and unary operations with backward functions.

## [1.9.0] - 2025-08-28

### Added

- Add `requires_grad` to `Tensor.where()`.

## [1.8.0] - 2025-08-27

### Added

- Add `requires_grad` to `Tensor.clip()`.

### Fixed

- Fix gradients for `square()`.

- Set `requires_grad` from the inputs.

### Changed

- Support gradients when only one operand is a tensor, as in `tensor**2`.

- Store only tensors in `prev`.

## [1.7.0] - 2025-08-26

### Added

- Add `requires_grad` to `random.rand()`, `randn()`, `randint()`, and `uniform()`.

## [1.6.0] - 2025-08-25

### Added

- Add `requires_grad` to `Tensor.zeros_like()`, `ones_like()`, `max()`, `maximum()`, and `sum()`.

- Add `Tensor.copy()`.

- Add `Tensor.to_device()` to copy a tensor to the requested device.

## [1.5.0] - 2025-08-24

### Added

- Read and write tensor elements directly with `__getitem__()` and `__setitem__()`.

- Make tensors iterable with `__iter__()`.


## [1.4.0] - 2025-08-23

### Added

- Add `Tensor.__pow__()`, `Tensor.power()`, `Tensor.log()`, and `Tensor.square()` with gradient support.

## [1.3.1] - 2025-08-22

### Changed

- Accept non-Tensor operands in comparison operators.

### Fixed

- Fix `initial` handling in reductions.

## [1.2.1] - 2025-08-21

### Fixed

- Fix `matmul()` when `where` is passed.

## [1.2.0] - 2025-08-20

### Added
- Add `device` to comparison, creation, elementwise, and reduction functions.
- Add the `OperandLike` type alias.

### Changed
- Accept NumPy-compatible inputs in tensor operations.
