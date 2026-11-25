from __future__ import annotations
from contextlib import contextmanager
from operator import index as _integer_index
from typing import Any, Callable, Iterator

from .typing import (
    TensorData,
    DTypeLike,
    Order,
    Device,
    Shape,
    TensorDataBool,
    Casting,
    TensorLike,
    Axis,
)

from ._core.exceptions import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
    GradientComputationError as GradientComputationError,
)
from ._core.indexing import _check_cuda_index_bounds, _prepare_index, _write_index
from ._core.ops.function import _ChainedBackward
from ._core.ops.elementwise import (
    add,
    subtract,
    multiply,
    matmul,
    divide,
    negative,
    power,
    clip,
    mean,
    sqrt,
    square,
    exp,
    log,
    abs,
    sign,
)

from ._core.ops.comparison import (
    equal,
    not_equal,
    greater,
    greater_equal,
    less,
    less_equal,
)

from ._core.ops.reduction import (
    max,
    maximum,
    sum,
)

from ._core.storage import _mark_external_write, _mark_out_modified, _register_tensor, _shares_storage
from ._core.utils import (
    _accumulate_operand_grad,
    _backward_active,
    _cast_gradient_work,
    _check_device,
    _check_value_dtype,
    _gradient_accumulator,
    _backward_compute_dtype,
    get_device,
    to_tensordata,
    tensordata_to_device,
    NoValue,
)

from .enum import InitializeGrad
import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None

def _check_grad_dtype(dtype: np.dtype) -> None:
    if dtype.kind != "f":
        raise RuntimeError(f"Gradients require a floating-point dtype. Got {dtype}.")

def _check_grad_compatibility(
    grad: TensorData, data: TensorData, device: Device
) -> tuple[bool, str]:
    mismatches = []
    if get_device(grad) != device:
        mismatches.append("device")

    if grad.shape != data.shape:
        mismatches.append("shape")

    if grad.dtype != data.dtype:
        mismatches.append("dtype")

    return not mismatches, ", ".join(mismatches)

class _CopyBackward(_ChainedBackward):
    def __init__(
        self,
        source: Tensor,
        result: Tensor,
        data: TensorData,
        previous_backward: Callable[[], None],
    ) -> None:
        self.source = source
        self.result = result
        self.data = data
        self.device = source.device
        self.previous_backward = previous_backward

    @contextmanager
    def _use_saved_output(self) -> Iterator[None]:
        source, result = self.source, self.result
        if result.grad is None:
            result.zero_grad()

        source_dtype, result_dtype = self.data.dtype, result.dtype
        preserve_work_dtype = _backward_active.get() and source_dtype == result_dtype
        if _backward_active.get():
            if preserve_work_dtype:
                source_dtype = result.grad.dtype
            else:
                source_dtype = _backward_compute_dtype(source_dtype)
                result_dtype = _backward_compute_dtype(result_dtype)

        grad = _cast_gradient_work(to_tensordata(result.grad, self.device), source_dtype, copy=True)

        # Backward callbacks capture the original output.
        state = source.data, source.grad, source.device, source.requires_grad
        try:
            object.__setattr__(source, "data", self.data)
            source.grad = grad
            source.device = self.device
            source.requires_grad = True

            yield

            if source.grad is not None and source.grad is not grad:
                result_grad = to_tensordata(source.grad, result.device)
                result.grad = (
                    result_grad if preserve_work_dtype else _cast_gradient_work(result_grad, result_dtype)
                )
        finally:
            object.__setattr__(source, "data", state[0])
            source.grad, source.device, source.requires_grad = state[1:]

class Tensor:
    """
    NumPy or CuPy data with optional autograd.

    Attributes
    ----------
    data : TensorData
        NumPy or CuPy array. Assignments like `tensor.data += x` can invalidate results needed for backward.
    device : Device
        Storage device.
    grad : TensorData | None
        Accumulated gradient.
    grad_fn : Callable[[], None] | None
        Backward callback.
    requires_grad : bool
        Gradient tracking.
    name : str
        Optional tensor name.
    """

    data: TensorData
    device: Device
    grad: TensorData | None
    grad_fn: Callable[[], None] | None
    requires_grad: bool
    name: str
    _version_counter: list[int]

    # Defer ndarray operators to the Tensor's reflected methods.
    __array_priority__ = 1000

    def __init__(
        self,
        obj: TensorLike,
        /,
        *,
        dtype: DTypeLike | None = None,
        device: Device | None = None,
        prev: tuple = (),
        requires_grad: bool = False,
        name: str = "",
    ) -> None:
        """
        Create a tensor.

        Parameters
        ----------
        obj : TensorLike
            Scalar, array-like value, or tensor.
        dtype : DTypeLike | None, optional
            Data dtype. Inferred if None.
        device : Device | None, optional
            Storage device. Inferred if None.
        prev : tuple, optional
            Graph parents.
        requires_grad : bool, optional
            Track gradients.
        name : str, optional
            Tensor name.

        Raises
        ------
        ValueError
            Unsupported device.
        RuntimeError
            Complex data and gradients on non-floating tensors are unsupported.
        """
        if device is None:
            self.device = get_device(obj)
        else:
            self.device = device

        self.data = to_tensordata(obj, self.device, dtype=dtype)

        if requires_grad:
            _check_grad_dtype(self.data.dtype)

        self.prev = prev
        self.requires_grad = requires_grad
        self.grad_fn = None
        self.grad = None
        self.name = name

        if isinstance(obj, Tensor) and self.data is obj.data:
            self._version_counter = obj._version_counter
        else:
            self._version_counter = [0]

        _register_tensor(self)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "data" and "_version_counter" in self.__dict__:
            if value is self.data:
                _mark_external_write(value)
                return

            data = to_tensordata(value)
            if self.requires_grad and (self.prev or self.grad_fn is not None):
                raise RuntimeError("Cannot replace data on a differentiable result.")
            self._replace_data(data)
            return

        if name in ("data", "grad") and value is not None:
            _check_value_dtype(value)

        object.__setattr__(self, name, value)

    @classmethod
    def _fast_init(
        cls,
        data: TensorData,
        device: Device,
        requires_grad: bool = False,
        prev: tuple = (),
        name: str = "",
    ) -> Tensor:
        if isinstance(data, np.generic):
            data = np.asarray(data)

        obj = object.__new__(cls)
        obj.data = data
        if requires_grad:
            _check_grad_dtype(data.dtype)

        obj.device = device
        obj.requires_grad = requires_grad
        obj.prev = prev
        obj.grad = None
        obj.grad_fn = None
        obj.name = name
        obj._version_counter = [0]

        _register_tensor(obj)
        return obj

    @property
    def _version(self) -> int:
        return self._version_counter[0]

    @_version.setter
    def _version(self, version: int) -> None:
        self._version_counter[0] = version

    def _replace_data(self, data: TensorData, *, aliases: tuple[Tensor, ...] = ()) -> None:
        if isinstance(data, np.generic):
            data = np.asarray(data)

        _check_value_dtype(data)
        device = get_device(data)

        if self.requires_grad:
            if device != self.device:
                raise RuntimeError("In-place device changes are forbidden on differentiable tensors.")
            _check_grad_dtype(data.dtype)

        clear_grad = (
            self.grad is not None and not _check_grad_compatibility(self.grad, data, device)[0]
        )

        same_data = self.data is data
        object.__setattr__(self, "data", data)
        self.device = device

        if not same_data:
            # Old aliases retain their version counter.
            self._version_counter = [self._version]
        self._version += 1

        if clear_grad:
            self.grad = None

        _mark_out_modified(aliases, replaced=self)

    def _check_backward_version(self, source_version: int) -> None:
        if self._version != source_version:
            raise RuntimeError("Tensor needed for backward was modified by an in-place operation.")

    def _fill_grad(self, value: int) -> None:
        if not self.requires_grad:
            raise RuntimeError("Cannot initialize gradients on an undifferentiable tensor.")

        if self.grad is None:
            dtype = _backward_compute_dtype(self.dtype) if _backward_active.get() else self.dtype
            xp = np if self.device == "cpu" else cp
            if xp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

            self.grad = xp.full_like(self.data, value, dtype=dtype)
        else:
            self.grad.fill(value)

    def zero_grad(self) -> None:
        """
        Reset the gradient to zeros.

        Raises
        ------
        RuntimeError
            Tracking is disabled.
        CuPyNotFound
            CUDA storage requires CuPy.
        """
        self._fill_grad(0)

    def one_grad(self) -> None:
        """
        Reset the gradient to ones.

        Raises
        ------
        RuntimeError
            Tracking is disabled.
        CuPyNotFound
            CUDA storage requires CuPy.
        """
        self._fill_grad(1)

    def copy(
        self,
        *,
        order: Order = "K",
        dtype: DTypeLike | None = None,
        device: Device | None = None,
        copy_prev: bool = False,
        copy_requires_grad: bool = False,
        copy_grad: bool = False,
    ) -> Tensor:
        """
        Copy the tensor.

        Parameters
        ----------
        order : Order, optional
            Memory layout. K keeps input order.
        dtype : DTypeLike | None, optional
            Copy dtype. None keeps the source dtype.
        device : Device | None, optional
            Copy device. None keeps the source device.
        copy_prev : bool, optional
            Keep history if the copy tracks gradients.
        copy_requires_grad : bool, optional
            Preserve gradient tracking.
        copy_grad : bool, optional
            Copy gradients to separate storage.

        Returns
        -------
        Tensor
            Copied tensor.
        """
        saved_data = self.data.copy(order=order)
        requires_grad = copy_requires_grad and self.requires_grad
        t = Tensor(
            saved_data,
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
            prev=self.prev if copy_prev and requires_grad else (),
            requires_grad=requires_grad,
        )

        if copy_grad and self.grad is not None:
            t.grad = to_tensordata(self.grad, t.device).astype(t.dtype, order=order, copy=True)

        original_backward = self.grad_fn
        if copy_prev and t.requires_grad and original_backward is not None:
            if t.data is saved_data:
                saved_data = saved_data.copy(order=order)
            t.grad_fn = _CopyBackward(self, t, saved_data, original_backward)

        return t

    def toggle_grad(
        self,
        state: bool | None = None,
        initialize_grad: InitializeGrad = InitializeGrad.NONE,
    ) -> None:
        """
        Set gradient tracking. Disabling clears gradients and history.

        Parameters
        ----------
        state : bool | None, optional
            Tracking state. None flips it.
        initialize_grad : InitializeGrad, optional
            Initialize gradients to zeros or ones when enabling tracking.

        Raises
        ------
        RuntimeError
            Unsupported dtype or incompatible stored gradient.
        """
        enabled = not self.requires_grad if state is None else state
        if not enabled:
            self.requires_grad = False
            self.grad = None
            self.grad_fn = None
            self.prev = ()
            return

        _check_grad_dtype(self.dtype)
        if self.grad is not None:
            compatible, mismatches = _check_grad_compatibility(self.grad, self.data, self.device)
            if not compatible:
                raise RuntimeError(f"Gradient mismatch in {mismatches}.")

        self.requires_grad = True

        if initialize_grad == InitializeGrad.ZEROS:
            self.zero_grad()
        elif initialize_grad == InitializeGrad.ONES:
            self.one_grad()

    def _link_transform_backward(
        self, result: Tensor, inverse: Callable[[TensorData], TensorData] | None = None
    ) -> Tensor:
        if not result.requires_grad:
            return result

        source_version = self._version

        def grad_fn() -> None:
            if not self.requires_grad:
                return

            self._check_backward_version(source_version)

            if result.grad is None:
                result.zero_grad()
            grad = result.grad if inverse is None else inverse(result.grad)
            _accumulate_operand_grad(self, grad)

        result.prev = (self,)
        result.grad_fn = grad_fn
        return result

    def to_device(
        self, device: Device, in_place: bool = False, clear_prev: bool = True
    ) -> Tensor:
        """
        Move data between devices.

        Parameters
        ----------
        device : Device
            Destination device.
        in_place : bool, optional
            Move this tensor in place.
        clear_prev : bool, optional
            Detach the copy. False keeps the link to the source for backward.

        Returns
        -------
        Tensor
            Self for in-place moves, otherwise a copy.

        Raises
        ------
        RuntimeError
            In-place device change on a differentiable tensor.
        ValueError
            Unsupported device.
        """
        _check_device(device)

        if in_place:
            if device == self.device:
                return self

            if self.requires_grad:
                raise RuntimeError(
                    "In-place device changes are forbidden on differentiable tensors."
                )

            self._replace_data(tensordata_to_device(self.data, device=device))
            if clear_prev:
                self.prev = ()

            return self

        result = self.copy(device=device, copy_requires_grad=not clear_prev)
        return self._link_transform_backward(result)

    def get_device(self) -> Device:
        """
        Return the storage device.

        Returns
        -------
        Device
            cpu or cuda.
        """
        return self.device

    def is_device(self, device: Device) -> bool:
        """
        Check the tensor's device.

        Parameters
        ----------
        device : Device
            Device to compare.

        Returns
        -------
        bool
            True if the devices match.
        """
        return self.device == device

    def is_cpu(self) -> bool:
        """
        Check for CPU storage.

        Returns
        -------
        bool
            True for CPU storage.
        """
        return self.device == "cpu"

    def is_cuda(self) -> bool:
        """
        Check for CUDA storage.

        Returns
        -------
        bool
            True for CUDA storage.
        """
        return self.device == "cuda"

    def is_contiguous(self) -> bool:
        """
        Check for C-contiguous storage.

        Returns
        -------
        bool
            True for C-contiguous storage.
        """
        if self.is_cpu():
            return self.data.flags["C_CONTIGUOUS"]
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)

            return self.data.flags.c_contiguous

    def contiguous(self) -> Tensor:
        """
        Make the tensor C-contiguous.

        Returns
        -------
        Tensor
            Self if C-contiguous, otherwise a copy.
        """
        if self.is_contiguous():
            return self

        if self.is_cpu():
            new_data = np.ascontiguousarray(self.data)
        else:
            new_data = cp.ascontiguousarray(self.data)

        result = Tensor._fast_init(
            new_data,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        return self._link_transform_backward(result)

    def set_name(self, name: str) -> Tensor:
        """
        Set the tensor name.

        Parameters
        ----------
        name : str
            New tensor name.

        Returns
        -------
        Tensor
            Self.
        """
        self.name = name
        return self

    def transpose(self, axes: Shape | None = None) -> Tensor:
        """
        Permute the axes.

        Parameters
        ----------
        axes : Shape | None, optional
            Axis permutation. None reverses the axes.

        Returns
        -------
        Tensor
            Transposed tensor.

        Raises
        ------
        ValueError
            Invalid axis permutation.
        """
        result = Tensor._fast_init(
            self.data.transpose(axes),
            device=self.device,
            requires_grad=self.requires_grad,
        )

        if _shares_storage(self.data, result.data):
            result._version_counter = self._version_counter

        if not result.requires_grad:
            return result

        inverse_axes = None
        if axes is not None:
            normalized_axes = np.atleast_1d(axes) % self.ndim
            inverse_axes = tuple(np.argsort(normalized_axes))

        return self._link_transform_backward(result, lambda grad: grad.transpose(inverse_axes))

    def backward(self, grad: Tensor | TensorData | None = None) -> list[Tensor]:
        """
        Backpropagate. Ancestor gradients accumulate on success.
        Use at least float32 for temporary gradients. Store them in each tensor's dtype.

        Parameters
        ----------
        grad : Tensor | TensorData | None, optional
            Seed matching the shape and device. None uses ones.

        Returns
        -------
        list[Tensor]
            Graph nodes in parent-first order.

        Raises
        ------
        RuntimeError
            Disabled tracking, incompatible seed, or a mutated operand.
        GradientComputationError
            Cyclic graph or missing backward function.
        CuPyNotFound
            CUDA storage requires CuPy.
        """
        from ._core.autograd import _backward

        return _backward(self, grad)

    @property
    def dtype(self) -> DTypeLike:
        """
        Element dtype.
        """
        return self.data.dtype

    @property
    def shape(self) -> Shape:
        """
        Tuple of dimension sizes.
        """
        return self.data.shape

    @property
    def ndim(self) -> int:
        """
        Number of dimensions.
        """
        return self.data.ndim

    @property
    def size(self) -> int:
        """
        Number of elements.
        """
        self.item
        return self.data.size

    def item(self, *indices: int | tuple[int, ...]) -> Any:
        """
        Return one element as a Python scalar.

        Parameters
        ----------
        *indices : int or tuple of int, optional
            Flat index or one per axis. Omit for a one-element tensor.

        Returns
        -------
        Any
            Python scalar.
        """
        if self.device == "cpu" or not indices:
            return self.data.item(*indices)

        if len(indices) == 1 and isinstance(indices[0], tuple):
            indices = indices[0]

        if not indices:
            return self.data.item()

        if any(isinstance(index, (bool, np.bool_)) for index in indices):
            raise TypeError("Indices must be integers.")

        coordinates = tuple(_integer_index(index) for index in indices)

        if len(coordinates) == 1:
            flat_index = coordinates[0]
            _check_cuda_index_bounds(flat_index, (self.size,))
            coordinates = np.unravel_index(flat_index % self.size, self.shape)
        elif len(coordinates) != self.ndim:
            raise ValueError("Use a flat index or one per axis.")
        else:
            _check_cuda_index_bounds(coordinates, self.shape)

        return self.data[coordinates].item()

    def clip(
        self,
        a_min: TensorLike,
        a_max: TensorLike,
        /,
        out: TensorData | None = None,
        *,
        requires_grad: bool | None = None,
        device: Device | None = None,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
    ) -> Tensor:
        """
        Clamp each element.

        Parameters
        ----------
        a_min : TensorLike | None
            Lower bound. None leaves it open.
        a_max : TensorLike | None
            Upper bound. None leaves it open.
        out : TensorData | None, optional
            Destination array.
        requires_grad : bool | None, optional
            Track gradients. None infers tracking from the inputs.
        device : Device | None, optional
            Output device. None infers the device from the inputs.
        where : TensorDataBool | bool, optional
            Broadcastable mask of elements to compute.
        casting : Casting, optional
            Casting policy.
        order : Order, optional
            Output memory layout.
        dtype : DTypeLike | None, optional
            Output dtype.
        subok : bool, optional
            Allow array subclasses.

        Returns
        -------
        Tensor
        """
        return clip(
            self,
            a_min,
            a_max,
            out=out,
            requires_grad=requires_grad,
            device=device,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def mean(
        self,
        /,
        axis: Axis | None = None,
        dtype: DTypeLike | None = None,
        out: TensorData | None = None,
        keepdims: bool = False,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        requires_grad: bool | None = None,
    ) -> Tensor:
        """
        Average selected elements.

        Parameters
        ----------
        axis : Axis | None, optional
            Reduction axes. None reduces all.
        dtype : DTypeLike | None, optional
            Accumulator dtype.
        out : TensorData | None, optional
            Destination array.
        keepdims : bool, optional
            Keep reduced axes with length one.
        device : Device | None, optional
            Output device. None infers the device from the inputs.
        in_place : bool, optional
            Update this tensor.
        where : TensorDataBool | bool, optional
            Elements to include.
        requires_grad : bool | None, optional
            Track gradients. None infers tracking from the inputs.

        Returns
        -------
        Tensor
        """
        return mean(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            device=device,
            in_place=in_place,
            where=where,
            requires_grad=requires_grad,
        )

    def sqrt(
        self,
        /,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
        requires_grad: bool | None = None,
    ) -> Tensor:
        """
        Take elementwise square roots.

        Parameters
        ----------
        out : TensorData | None, optional
            Destination array.
        device : Device | None, optional
            Output device. None infers the device from the inputs.
        in_place : bool, optional
            Update this tensor.
        where : TensorDataBool | bool, optional
            Broadcastable mask of elements to compute.
        casting : Casting, optional
            Casting policy.
        order : Order, optional
            Output memory layout.
        dtype : DTypeLike | None, optional
            Output dtype.
        subok : bool, optional
            Allow array subclasses.
        requires_grad : bool | None, optional
            Track gradients. None infers tracking from the inputs.

        Returns
        -------
        Tensor
        """
        return sqrt(
            self,
            out=out,
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            requires_grad=requires_grad,
        )

    def square(
        self,
        /,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
        requires_grad: bool | None = None,
    ) -> Tensor:
        """
        Square each element.

        Parameters
        ----------
        out : TensorData | None, optional
            Destination array.
        device : Device | None, optional
            Output device. None infers the device from the inputs.
        in_place : bool, optional
            Update this tensor.
        where : TensorDataBool | bool, optional
            Broadcastable mask of elements to compute.
        casting : Casting, optional
            Casting policy.
        order : Order, optional
            Output memory layout.
        dtype : DTypeLike | None, optional
            Output dtype.
        subok : bool, optional
            Allow array subclasses.
        requires_grad : bool | None, optional
            Track gradients. None infers tracking from the inputs.

        Returns
        -------
        Tensor
        """
        return square(
            self,
            out=out,
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            requires_grad=requires_grad,
        )

    def log(
        self,
        /,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
        requires_grad: bool | None = None,
    ) -> Tensor:
        """
        Take elementwise natural logs.

        Parameters
        ----------
        out : TensorData | None, optional
            Destination array.
        device : Device | None, optional
            Output device. None infers the device from the inputs.
        in_place : bool, optional
            Update this tensor.
        where : TensorDataBool | bool, optional
            Broadcastable mask of elements to compute.
        casting : Casting, optional
            Casting policy.
        order : Order, optional
            Output memory layout.
        dtype : DTypeLike | None, optional
            Output dtype.
        subok : bool, optional
            Allow array subclasses.
        requires_grad : bool | None, optional
            Track gradients. None infers tracking from the inputs.

        Returns
        -------
        Tensor
        """
        return log(
            self,
            out=out,
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            requires_grad=requires_grad,
        )

    def abs(
        self,
        /,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
        requires_grad: bool | None = None,
    ) -> Tensor:
        """
        Take elementwise absolute values.

        Parameters
        ----------
        out : TensorData | None, optional
            Destination array.
        device : Device | None, optional
            Output device. None infers the device from the inputs.
        in_place : bool, optional
            Update this tensor.
        where : TensorDataBool | bool, optional
            Broadcastable mask of elements to compute.
        casting : Casting, optional
            Casting policy.
        order : Order, optional
            Output memory layout.
        dtype : DTypeLike | None, optional
            Output dtype.
        subok : bool, optional
            Allow array subclasses.
        requires_grad : bool | None, optional
            Track gradients. None infers tracking from the inputs.

        Returns
        -------
        Tensor
        """
        return abs(
            self,
            out=out,
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            requires_grad=requires_grad,
        )

    def sign(
        self,
        /,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
        requires_grad: bool | None = None,
    ) -> Tensor:
        """
        Return each element's sign.

        Parameters
        ----------
        out : TensorData | None, optional
            Destination array.
        device : Device | None, optional
            Output device. None infers the device from the inputs.
        in_place : bool, optional
            Update this tensor.
        where : TensorDataBool | bool, optional
            Broadcastable mask of elements to compute.
        casting : Casting, optional
            Casting policy.
        order : Order, optional
            Output memory layout.
        dtype : DTypeLike | None, optional
            Output dtype.
        subok : bool, optional
            Allow array subclasses.
        requires_grad : bool | None, optional
            Track gradients. None infers tracking from the inputs.

        Returns
        -------
        Tensor
        """
        return sign(
            self,
            out=out,
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            requires_grad=requires_grad,
        )

    def exp(
        self,
        /,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
        requires_grad: bool | None = None,
    ) -> Tensor:
        """
        Exponentiate each element.

        Parameters
        ----------
        out : TensorData | None, optional
            Destination array.
        device : Device | None, optional
            Output device. None infers the device from the inputs.
        in_place : bool, optional
            Update this tensor.
        where : TensorDataBool | bool, optional
            Broadcastable mask of elements to compute.
        casting : Casting, optional
            Casting policy.
        order : Order, optional
            Output memory layout.
        dtype : DTypeLike | None, optional
            Output dtype.
        subok : bool, optional
            Allow array subclasses.
        requires_grad : bool | None, optional
            Track gradients. None infers tracking from the inputs.

        Returns
        -------
        Tensor
        """
        return exp(
            self,
            out=out,
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            requires_grad=requires_grad,
        )

    def sum(
        self,
        /,
        *,
        device: Device | None = None,
        requires_grad: bool | None = None,
        axis: Axis | None = None,
        dtype: DTypeLike | None = None,
        out: TensorData | None = None,
        keepdims: bool = False,
        initial: Any = NoValue,
        where: TensorDataBool | bool = True,
    ) -> Tensor:
        """
        Sum selected elements.

        Parameters
        ----------
        device : Device | None, optional
            Output device. None infers the device from the inputs.
        requires_grad : bool | None, optional
            Track gradients. None infers tracking from the inputs.
        axis : Axis | None, optional
            Reduction axes. None reduces all.
        dtype : DTypeLike | None, optional
            Accumulator and result dtype.
        out : TensorData | None, optional
            Destination array.
        keepdims : bool, optional
            Keep reduced axes with length one.
        initial : Any, optional
            Starting value.
        where : TensorDataBool | bool, optional
            Elements to include.

        Returns
        -------
        Tensor
        """
        return sum(
            self,
            axis=axis,
            device=device,
            requires_grad=requires_grad,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def max(
        self,
        /,
        *,
        device: Device | None = None,
        requires_grad: bool | None = None,
        axis: Axis | None = None,
        out: TensorData | None = None,
        keepdims: bool = False,
        initial: Any = NoValue,
        where: TensorDataBool | bool = True,
    ) -> Tensor:
        """
        Find the largest selected value.

        Parameters
        ----------
        device : Device | None, optional
            Output device. None infers the device from the inputs.
        requires_grad : bool | None, optional
            Track gradients. None infers tracking from the inputs.
        axis : Axis | None, optional
            Reduction axes. None reduces all.
        out : TensorData | None, optional
            Destination array.
        keepdims : bool, optional
            Keep reduced axes with length one.
        initial : Any, optional
            Lower bound. Required for masked or empty reductions.
        where : TensorDataBool | bool, optional
            Elements to include.

        Returns
        -------
        Tensor

        Raises
        ------
        ValueError
            A masked or empty reduction has no initial value.
        """
        return max(
            self,
            axis=axis,
            device=device,
            requires_grad=requires_grad,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def maximum(
        self,
        x2: TensorLike,
        /,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        requires_grad: bool | None = None,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
    ) -> Tensor:
        """
        Take elementwise maxima.

        Parameters
        ----------
        x2 : TensorLike
            Other operand.
        out : TensorData | None, optional
            Destination array.
        device : Device | None, optional
            Output device. None infers the device from the inputs.
        requires_grad : bool | None, optional
            Track gradients. None infers tracking from the inputs.
        where : TensorDataBool | bool, optional
            Broadcastable mask of elements to compute.
        casting : Casting, optional
            Casting policy.
        order : Order, optional
            Output memory layout.
        dtype : DTypeLike | None, optional
            Output dtype.
        subok : bool, optional
            Allow array subclasses.

        Returns
        -------
        Tensor
        """
        return maximum(
            self,
            x2,
            out=out,
            device=device,
            requires_grad=requires_grad,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def __add__(self, t: TensorLike) -> Tensor:
        """
        Add element-wise.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
        """
        return add(self, t)

    def __radd__(self, t: TensorLike) -> Tensor:
        """
        Add with this tensor on the right.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
        """
        return add(t, self)

    def __iadd__(self, t: TensorLike) -> Tensor:
        """
        Add in place without gradient tracking.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
            Self.

        Raises
        ------
        RuntimeError
            An operand tracks gradients.
        """
        return add(self, t, in_place=True)

    def __sub__(self, t: TensorLike) -> Tensor:
        """
        Subtract element-wise.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
        """
        return subtract(self, t)

    def __rsub__(self, t: TensorLike) -> Tensor:
        """
        Subtract this tensor from the left operand.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
        """
        return subtract(t, self)

    def __isub__(self, t: TensorLike) -> Tensor:
        """
        Subtract in place without gradient tracking.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
            Self.

        Raises
        ------
        RuntimeError
            An operand tracks gradients.
        """
        return subtract(self, t, in_place=True)

    def __mul__(self, t: TensorLike) -> Tensor:
        """
        Multiply element-wise.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
        """
        return multiply(self, t)

    def __rmul__(self, t: TensorLike) -> Tensor:
        """
        Multiply by this tensor.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
        """
        return multiply(t, self)

    def __imul__(self, t: TensorLike) -> Tensor:
        """
        Multiply in place without gradient tracking.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
            Self.

        Raises
        ------
        RuntimeError
            An operand tracks gradients.
        """
        return multiply(self, t, in_place=True)

    def __matmul__(self, t: TensorLike) -> Tensor:
        """
        Matrix multiplication.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
        """
        return matmul(self, t)

    def __rmatmul__(self, t: TensorLike) -> Tensor:
        """
        Compute `value @ self`.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
        """
        return matmul(t, self)

    def __imatmul__(self, t: TensorLike) -> Tensor:
        """
        Compute `self @ value` in place without tracking gradients.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
            Self.

        Raises
        ------
        RuntimeError
            An operand tracks gradients.
        """
        return matmul(self, t, in_place=True)

    def __truediv__(self, t: TensorLike) -> Tensor:
        """
        Divide element-wise.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
        """
        return divide(self, t)

    def __rtruediv__(self, t: TensorLike) -> Tensor:
        """
        Divide the left operand by this tensor.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
        """
        return divide(t, self)

    def __itruediv__(self, t: TensorLike) -> Tensor:
        """
        Divide in place without gradient tracking.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
            Self.

        Raises
        ------
        RuntimeError
            An operand tracks gradients.
        """
        return divide(self, t, in_place=True)

    def __pow__(self, t: TensorLike) -> Tensor:
        """
        Raise each element to its given power.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
        """
        return power(self, t)

    def __rpow__(self, t: TensorLike) -> Tensor:
        """
        Raise the left operand to this tensor's powers.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
        """
        return power(t, self)

    def __ipow__(self, t: TensorLike) -> Tensor:
        """
        Raise to powers in place without gradient tracking.

        Parameters
        ----------
        t : TensorLike
            Other operand.

        Returns
        -------
        Tensor
            Self.

        Raises
        ------
        RuntimeError
            An operand tracks gradients.
        """
        return power(self, t, in_place=True)

    def __eq__(self, value: Any) -> Tensor:  # type: ignore[override]
        """
        Compare elements for equality.

        Parameters
        ----------
        value : Any
            Other operand.

        Returns
        -------
        Tensor
        """
        return equal(self, value)

    def __ne__(self, value: Any) -> Tensor:  # type: ignore[override]
        """
        Compare elements for inequality.

        Parameters
        ----------
        value : Any
            Other operand.

        Returns
        -------
        Tensor
        """
        return not_equal(self, value)

    def __gt__(self, value: Any) -> Tensor:  # type: ignore[override]
        """
        Compare elements using >.

        Parameters
        ----------
        value : Any
            Other operand.

        Returns
        -------
        Tensor
        """
        return greater(self, value)

    def __ge__(self, value: Any) -> Tensor:  # type: ignore[override]
        """
        Compare elements using >=.

        Parameters
        ----------
        value : Any
            Other operand.

        Returns
        -------
        Tensor
        """
        return greater_equal(self, value)

    def __lt__(self, value: Any) -> Tensor:  # type: ignore[override]
        """
        Compare elements using <.

        Parameters
        ----------
        value : Any
            Other operand.

        Returns
        -------
        Tensor
        """
        return less(self, value)

    def __le__(self, value: Any) -> Tensor:  # type: ignore[override]
        """
        Compare elements using <=.

        Parameters
        ----------
        value : Any
            Other operand.

        Returns
        -------
        Tensor
        """
        return less_equal(self, value)

    def __neg__(self) -> Tensor:
        """
        Negate each element.

        Returns
        -------
        Tensor
        """
        return negative(self)

    def __abs__(self) -> Tensor:
        """
        Take elementwise absolute values.

        Returns
        -------
        Tensor
            Absolute values.
        """
        return self.abs()

    def __bool__(self) -> bool:
        if self.size != 1:
            raise ValueError("Tensor truth value is ambiguous unless it has one element.")

        return bool(self.item())

    def __getitem__(self, index) -> Any:
        """
        Return indexed elements.

        Parameters
        ----------
        index : int, slice, tuple, array-like, or Tensor
            Indices or slices.

        Returns
        -------
        Tensor
            Indexed values.
        """
        index = _prepare_index(index, self.device, save=self.requires_grad)
        if self.is_cuda():
            _check_cuda_index_bounds(index, self.shape)

        y = Tensor._fast_init(
            self.data[index],
            device=self.device,
            requires_grad=self.requires_grad,
        )

        shares_storage = _shares_storage(self.data, y.data)
        if shares_storage:
            if y.size == self.size and y.dtype == self.dtype:
                y._version_counter = self._version_counter
            else:
                y._version = self._version

        if y.requires_grad:
            source_version = self._version
            xp = np if self.is_cpu() else cp

            def grad_fn() -> None:
                if not self.requires_grad:
                    return

                self._check_backward_version(source_version)

                if y.grad is None:
                    y.zero_grad()
                grad = y.grad
                accumulator = _gradient_accumulator(self, grad)
                if _backward_active.get():
                    grad = _cast_gradient_work(grad, accumulator.dtype)

                if shares_storage:
                    accumulator[index] += grad
                else:
                    xp.add.at(accumulator, index, grad)

            y.prev = (self,)
            y.grad_fn = grad_fn

        return y

    def __setitem__(self, index, value) -> Any:
        """
        Assign at an index or slice.

        Parameters
        ----------
        index : int, slice, tuple, array-like, or Tensor
            Indices or slices.
        value : Any
            Values to assign. Tensors move to the destination.

        Raises
        ------
        RuntimeError
            The write would overwrite a differentiable result.
        """
        _write_index(self.data, index, value, self.device)

    def __reduce_ex__(self, protocol: int) -> Any:
        if self.prev or self.grad_fn is not None:
            raise TypeError(
                "Tensor has computation history. Use Tensor.copy() to detach, or "
                "pass copy_prev=True and copy_requires_grad=True to keep the graph."
            )

        return super().__reduce_ex__(protocol)

    def __setstate__(self, state: dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]) -> None:
        slots = {}
        if isinstance(state, tuple):
            state, slots = state

        _check_value_dtype(state.get("data"))
        _check_value_dtype(state.get("grad"))

        for name, value in slots.items():
            setattr(self, name, value)
        self.__dict__.update(state)
        self.__dict__.setdefault("_version_counter", [0])

        if isinstance(self.data, np.generic):
            object.__setattr__(self, "data", np.asarray(self.data))

        _register_tensor(self)

    def __iter__(self) -> Iterator:
        """
        Iterate over the data.

        Returns
        -------
        Iterator
            Array iterator.
        """
        return iter(self.data)

    def __repr__(self) -> str:
        """
        Format the tensor.

        Returns
        -------
        str
            Data, device, and tracking flag.
        """
        return f"Tensor({self.data}, device={self.device}{f', name={self.name}' if self.name != '' else ''}, req_grad={self.requires_grad})"

    def __hash__(self) -> int:
        """
        Hash by identity.

        Returns
        -------
        int
            Object ID.
        """
        return id(self)
