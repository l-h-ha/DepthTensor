# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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