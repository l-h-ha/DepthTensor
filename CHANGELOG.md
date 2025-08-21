# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.1] - 2025-08-21

### Fixed

- Operation matmul bug where argument where was given.

## [1.2.0] - 2025-08-20

### Added
- Device parameters in comparison/creation/elementwise/reduction functions.
- New type alias: OperandLike

### Changed
- Computational functions now accept regular datatypes (the kinds that NumPy also accepts), as opposed to only tensors.