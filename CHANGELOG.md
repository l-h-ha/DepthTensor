# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.7.0] - 2026-02-07

### Fixed

- Fixed a bug in `Tensor.sum` and `Tensor.max` (and their functional counterparts) where the `initial` argument was incorrectly ignored or caused crashes due to incorrect default value handling. A new internal `NoValue` sentinel is now used.

- Fixed `Tensor.__neg__` (unary negation) to correctly propagate `requires_grad`, enabling autodifferentiation for negated tensors.

- Fixed a `NameError` in the backward pass of unary operations (e.g., `negative`, `abs`, `exp`) where the variable `result` was used instead of `y`.

- Fixed `to_tensordata` utility to correctly infer the device when `device=None` is passed.

- Fixed `TypeError` in `maximum` reduction operation where internal helper `get_two_operand_op_device` was missing required arguments.

### Added

- Comprehensive unit test suite covering:
  - Tensor initialization, properties, and device management (`tests/test_tensor_init.py`).
  
  - Forward operations including arithmetic, comparison, and reduction (`tests/test_ops_forward.py`).
  
  - Autodifferentiation (backward pass) for scalars, broadcasting, and complex graphs (`tests/test_autodiff.py`).

  - Extended autodifferentiation tests for unary/binary ops, structural ops, and non-differentiable reductions (`tests/test_autodiff_extended.py`).

  - Error handling for in-place operations and device safety (`tests/test_tensor_errors.py`).

### Improved

- Significantly improved performance of internal Tensor creation and operations by implementing a fast initialization path (`Tensor._fast_init`), bypassing redundant checks for internal operations. This results in ~4x speedup for lightweight operations.

### Documentation

- Integrated comprehensive NumPy-style docstrings across the codebase, including `Tensor` class methods, autodifferentiation engine (`autodiff.py`), and core operations (`ops/`), improving code readability and developer experience.

## [2.6.0] - 2026-01-11

### Fixed

- Optimized the ```zero_grad``` method by filling the array (if already instantiated) instead of instantiating everytime.

- Some internal functions are optimized by caching variables.


### Changed

- ```dt.random.uniform()``` now accepts a ```dtype``` argument, which defaults to ```float32```.

- Printing tensors now shows the ```requires_grad``` attribute for verbosity.


## [2.5.1] - 2026-01-10

### Fixed

- A bug where the ```prev``` attribute is only initialized upon differentiation.

## [2.5.0] - 2026-01-08

### Added

- A ```name``` attribute in tensor objects assisting in debugging.

- Added a method ```set_name```, which returns the tensor object itself and sets the object's ```name``` attribute.

- Added a differentiable method ```transpose`` which tranposes the tensor object.

- The slicing operation can now be differentiable.

### Changed

- Changed the method ```Tensor.zeros_grad()``` to ```Tensor.zero_grad()``` as it is more conventional.

- Calling ```Tensor.zero_grad()``` now raises a run-time error for undifferentiable tensors.

- Operations, like ```Add, Subtract, or Multiply``` now behave like objects inheriting from the ```Function``` class.

### Removed

- Removed unary-related and binary-related protocols for creating custom differentiable functions, which now works via the class `Function`.

## [2.4.3] - 2025-12-18

### Fixed
 
- Fixed a bug where differentiating tensors does not automatically initialize the gradients of them.

- Fixed a bug where the `max()` function errors due to the default `initial` argument being invalid.

### Changed

- Custom binary differentiation functions now require explicit definition of the `device` argument.

## [2.4.2] - 2025-12-14

### Fixed

- Fixed a potentially unwanted and unexpected behavior in gradient update of custom operation functions, where the previous default update behavior was accumulating the Hadamard product of the resulting node with the local derivatives (the domain of manual definition beyond which is unsupported) of the preceding nodes. Essentially: 

$$
\bar{x} := \bar{y} \odot f(y, x)
$$

- where $f(y, x)$ is manually defined by users.

## [2.4.1] - 2025-12-14

### Changed

- Renamed ```TensorLike, DeviceLike, ScalarLike, ShapeLike, AxisLike, NDArrayLike, NDArrayLikeBool, OperandLike, cfunc_2in_1out_pro, cop_2in_1out_pro, cdiff_2in_1out_pro, cfunc_1in_1out_pro, cop_1in_1out_pro, cdiff_1in_1out_pro``` to ```TensorType, Device, Scalar, Shape, Axis, TensorData, TensorDataBool, TensorLike, BinaryFunc, BinaryOp, BinaryDiff, UnaryFunc, UnaryOp, UnaryDiff```, respectively, for easier typehinting.

## [2.4.0] - 2025-12-14

### Added

- A method ```item()``` that returns the item of the ```numpy``` or ```cupy``` data.

- ```differentiate()``` function now accepts ```grad``` argument which allows manually setting the gradient stream source.

## [2.3.1] - 2025-12-10

### Fixed

- A critical bug related to autograd which raises a ```GradientComputationError``` exception when encountering a tensor that is not differentiable. This means that unrelated data such as datasets, X, had to be differentiable.

### Added

- The differentiate function now accepts

## [2.3.0] - 2025-12-04

### Added

- Added a method ```mean()```, which outputs differentiable tensors, in the ```Tensor``` class.

### Fixed

-  A bug where differentiating tensors that resulted from 1-input-1-output operations, e.g. $y = f(x)$, bugs out due to an oversight in device management.

## [2.2.1] - 2025-11-26

### Fixed

- A bug where it is possible to perform in-place operations on differentiable tensors in computational graphs, which can lead to unexpected behaviors related to gradients. Now in-place operations ```iadd```, ```isub```, etc., and notably ```setitem``` would raise a `RunTimeError` exception on differentiable tensors.

## [2.2.0] - 2025-10-5

### Added

- New functions ```zeros()``` and ```ones()```.

## [2.1.0] - 2025-09-4

### Added

- Gradient optimization where it is only initialized when the tensor unit is about to perform differentiation.

### Changed

- Different implementations for ```DTensor.random.rand()``` and ```DTensor.random.randn()``` which boost performance.

### Fixed

- Minor fix in code of several functions that boosts performance as type casting relies less on ```arr.astype()```. 

## [2.0.2] - 2025-09-2

### Fixed

- Fixed the way tensor instantiation handles device. Prior to this change, putting a ```cupy.ndarray``` into the non-positional argument, while leaving the ```device``` argument blank, results in a tensor of which the ```grad``` attribute is a ```numpy.ndarray``` instead of ```cupy.ndarray``` as expected.

## [2.0.1] - 2025-09-2

### Fixed

- Fixed the way that devices are handled in built-in differentiable functions (```square()```, ```add()```, etc.).

## [2.0.0] - 2025-09-1

### Fixed

- Some functions' parameters of which ```device```, in particular, defaults to ```"cpu"```, which causes unwanted behavior. It is now defaulted to ```None```.

### Changed

- Some functions such as ```Tensor.square()```, ```Tensor.add()```, etc. are now only accessible via the ```DepthTensor``` package. An example would be ```DepthTensor.square()```.

## [1.12.0] - 2025-08-29

### Added

- Keywords arguments in diff functions used for ```create_2in_1out``` and ```create_1in_1out```, enabling further customization.

## [1.11.1] - 2025-08-29

### Fixed

- Generator function produced by either ```create_2in_1out``` or ```create_1in_1out``` has their ```device``` arguments defaulting to ```"cpu"```, which introduces unwanted behaviors. The parameters now default to ```None```.

## [1.11.0] - 2025-08-28

### Fixed

- Instantiating tensors with non-numerical values now results in an error.

- Some computation functions, such as add or subtract, might output tensors of unexpected devices adhering to the device parameter having a default value of "cpu" (now defaults to None as of these occurrences,) leading to two gpu tensors outputting a cpu tensor.

### Added

- Added method ```Tensor.make_differentiable()``` which transforms (in-place) a tensor into a differentiable one.

- Added method ```Tensor.to_device()``` which either transforms (in-place) a tensor into another of given device, or creates a copy of given device.

### Removed

- Type Alias (in DepthTensor.typing) ```AxisShapeLike``` as ```ShapeLike``` already does the job.

### Changed

- Type Aliases (in DepthTensor.typing) ```ArrayLike``` and ```ArrayLikeBool``` to ```NDArrayLike``` and ```NDArrayLikeBool```, respectively.

- Some computational functions now have the ```device``` arguments defaulting to ```None```.

## [1.10.0] - 2025-08-28

### Added

- Function ```DepthTensor.create_2in_1out()``` which returns a generator function that allows custom implemented operations of two operands with custom differentiations.

- Function ```DepthTensor.create_1in_1out()``` which returns a generator function that allows custom implemented operations of one operand with custom differentiations.

## [1.9.0] - 2025-08-28

### Added

- Additional ```requires_grad``` parameter for static method ```Tensor.where()```.

## [1.8.0] - 2025-08-27

### Added

- Additional ```requires_grad``` parameter for static method ```Tensor.clip()```.

### Fixed

- Gradient miscalculation in the square operation.

- Output tensors' ```requires_grad``` attributes now adheres to operand tensors'.

### Changed

- Differentiation now does not require both operands as tensor objects, which means that ```tensor**2``` is now differentiable.

- Operations now return tensors of which ```prev``` attributes contain only tensors.

## [1.7.0] - 2025-08-26

### Added

- Additional ```requires_grad``` parameter for function ```random.rand()```.

- Additional ```requires_grad``` parameter for function ```random.randn()```.

- Additional ```requires_grad``` parameter for function ```random.randint()```.

- Additional ```requires_grad``` parameter for function ```random.uniform()```.

## [1.6.0] - 2025-08-25

### Added

- Additional ```requires_grad``` parameter for static method ```Tensor.zeros_like()```.

- Additional ```requires_grad``` parameter for static method ```Tensor.ones_like()```.

- Additional ```requires_grad``` parameter for static method ```Tensor.max()```.

- Additional ```requires_grad``` parameter for static method ```Tensor.maximum()```.

- Additional ```requires_grad``` parameter for static method ```Tensor.sum()```.

- Method ```Tensor.copy()``` which copies tensors.

- Method ```Tensor.to_device()``` returns a copy of which attribute ```device``` 

## [1.5.0] - 2025-08-24

### Added

- Dunder method ```Tensor.__getitem__()```, enabling entry access without directing to ```data``` attribute.

- Dunder method ```Tensor.__setitem__()```, enabling entry overwriting without directing to ```data``` attribute.

- Dunder method ```Tensor.__iter__()```, enabling iteratable tensor objects.


## [1.4.0] - 2025-08-23

### Added

- Dunder method ```Tensor.__pow__()```, along with static method ```Tensor.power()```, with differentiation functionalities.

- Static methods ```Tensor.log()``` and ```Tensor.square()``` with differentiation functionalities.

## [1.3.1] - 2025-08-22

### Changed

- Comparison dunder methods now accept regular datatypes as opposed to primarily tensors.

### Fixed

- Initial argument values handled correctly for some functions.

## [1.2.1] - 2025-08-21

### Fixed

- Operation matmul bug where argument where was given.

## [1.2.0] - 2025-08-20

### Added
- Device parameters in comparison/creation/elementwise/reduction functions.
- New type alias: OperandLike

### Changed
- Computational functions now accept regular datatypes (the kinds that NumPy also accepts), as opposed to only tensors.