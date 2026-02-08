from __future__ import annotations
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

from ._core import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
    # * elementwise
    add,
    subtract,
    multiply,
    matmul,
    divide,
    negative,
    power,
    clip,
    abs,
    mean,
    sqrt,
    square,
    log,
    exp,
    sign,
    # * comparison
    equal,
    not_equal,
    greater,
    greater_equal,
    less,
    less_equal,
    # * reduction
    max,
    maximum,
    sum,
)

from ._core.utils import get_device, to_tensordata, tensordata_to_device, NoValue

import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None


###
###
###


def infer_differentiation_status(a: Tensor, b: TensorLike) -> bool:
    if isinstance(b, Tensor):
        return a.requires_grad or b.requires_grad
    return a.requires_grad


###
###
###


class Tensor:
    """
    A multi-dimensional array (tensor) with automatic differentiation support.

    This class provides a wrapper around NumPy and CuPy arrays, enabling
    GPU acceleration and automatic gradient computation.

    Attributes
    ----------
    data : TensorData
        The underlying data of the tensor (numpy.ndarray or cupy.ndarray).
    device : Device
        The device where the tensor data resides ('cpu' or 'gpu').
    grad : TensorData | None
        The gradient of the tensor. None if no gradient is computed.
    backward : Callable[[], None] | None
        The backward function for automatic differentiation.
    requires_grad : bool
        Whether the tensor requires gradient computation.
    name : str
        Optional name for the tensor.
    """

    data: TensorData
    device: Device
    grad: TensorData | None
    backward: Callable[[], None] | None

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
        Initialize a Tensor object.

        Parameters
        ----------
        obj : TensorLike
            The data to initialize the tensor with. Can be a list, tuple,
            numpy array, cupy array, or another Tensor.
        dtype : DTypeLike | None, optional
            The desired data type of the tensor. If None, it is inferred from `obj`.
        device : Device | None, optional
            The device to place the tensor on ('cpu' or 'gpu'). If None,
            it is inferred from `obj`.
        prev : tuple, optional
            Previous tensors in the computation graph (used for autodiff).
            Default is empty tuple.
        requires_grad : bool, optional
            Whether the tensor requires gradient computation. Default is False.
        name : str, optional
            Optional name for the tensor. Default is empty string.
        """
        # Device init
        if device is None:
            self.device = get_device(obj)
        else:
            self.device = device

        # Data init
        if isinstance(obj, Tensor):
            self.data = obj.data
        elif isinstance(obj, np.ndarray):
            self.data = obj
        elif cp is not None and isinstance(obj, cp.ndarray):
            self.data = obj
        else:
            self.data = to_tensordata(obj, self.device)

        # Conversion
        if dtype is not None and dtype != self.data.dtype:
            self.data = self.data.astype(dtype)
        if get_device(self.data) != self.device:
            self.data = tensordata_to_device(self.data, self.device)

        # Other inits
        self.prev = prev
        self.requires_grad = requires_grad
        self.backward = None
        self.grad = None
        self.name = name

    @classmethod
    def _fast_init(
        cls,
        data: TensorData,
        device: Device,
        requires_grad: bool = False,
        prev: tuple = (),
        name: str = "",
    ) -> Tensor:
        obj = object.__new__(cls)
        obj.data = data
        obj.device = device
        obj.requires_grad = requires_grad
        obj.prev = prev
        obj.grad = None
        obj.backward = None
        obj.name = name
        return obj

    def zero_grad(self) -> None:
        """
        Clears the gradients of the tensor.

        Raises
        ------
        RuntimeError
            If the tensor does not require gradients.
        CuPyNotFound
            If the device is 'gpu' and CuPy is not available.
        """
        if not self.requires_grad:
            raise RuntimeError(
                "Attempted to zero-gradient initialize an undifferentiable tensor."
            )
        if self.grad is None:
            if self.device == "cpu":
                grad = np.zeros_like(self.data)
            else:
                if cp is None:
                    raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                grad = cp.zeros_like(self.data)
            self.grad = grad
        else:
            self.grad.fill(0)

    ###
    ###
    ###

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
        Return a copy of the tensor.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout of the copy. 'C' means C-order,
            'F' means Fortran-order, 'A' means 'F' if a is Fortran contiguous,
            'C' otherwise. 'K' means match the layout of a as closely
            as possible. Default is 'K'.
        dtype : DTypeLike | None, optional
            The data type of the copy. If None, preserves the original dtype.
        device : Device | None, optional
            The device for the copy. If None, preserves the original device.
        copy_prev : bool, optional
            Whether to copy the computation graph history. Default is False.
        copy_requires_grad : bool, optional
            Whether to copy the requires_grad flag. Default is False.
        copy_grad : bool, optional
            Whether to copy the gradients. Default is False.

        Returns
        -------
        Tensor
            A copy of the tensor.
        """
        t = Tensor(
            self.data.copy(order=order),
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
            prev=self.prev if copy_prev else (),
            requires_grad=self.requires_grad if copy_requires_grad else False,
        )
        if copy_grad:
            t.grad = self.grad
        return t

    def make_differentiable(self, grad: Tensor | TensorData | None = None) -> None:
        """
        Enables gradient tracking for the tensor and optionally initializes gradients.

        Parameters
        ----------
        grad : Tensor | TensorData | None, optional
            Initial gradient values. If None, gradients are initialized to zero.

        Raises
        ------
        RuntimeError
            If there is a mismatch in device or type between the tensor and the provided grad.
        CuPyNotFound
            If the device is 'gpu' and CuPy is not available.
        """
        if not self.requires_grad:
            self.requires_grad = True

            if grad is None:
                if self.device == "cpu":
                    self.grad = np.zeros(self.shape)
                else:
                    if cp is None:
                        raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                    self.grad = cp.zeros(self.shape)
            else:
                if isinstance(grad, Tensor):
                    if grad.device != self.device:
                        raise RuntimeError(
                            "There is a mismatch in grad's device and tensor's device."
                        )
                    self.grad = grad.data
                elif isinstance(grad, np.ndarray):
                    if self.device == "gpu":
                        raise RuntimeError(
                            "Expected grad parameter to be a cupy.ndarray."
                        )
                    self.grad = grad
                elif cp is not None and isinstance(grad, cp.ndarray):
                    if self.device == "cpu":
                        raise RuntimeError(
                            "Expected grad parameter to be a numpy.ndarray."
                        )
                    self.grad = grad
                else:
                    raise RuntimeError(
                        "Expected grad parameter of specific types: Tensor, numpy.ndarray, cupy.ndarray."
                    )

    def to_device(
        self, device: Device, in_place: bool = False, clear_prev: bool = True
    ) -> Tensor:
        """
        Moves the tensor to the specified device.

        Parameters
        ----------
        device : Device
            The target device ('cpu' or 'gpu').
        in_place : bool, optional
            Whether to perform the operation in-place. Default is False.
        clear_prev : bool, optional
            Whether to clear the computation graph history. Default is True.

        Returns
        -------
        Tensor
            The tensor on the specified device.

        Raises
        ------
        RuntimeError
            If in-place operation is attempted on a differentiable tensor.
        """
        if device == self.device:
            if in_place:
                return self
            return self.copy()
        else:
            if in_place:
                if self.requires_grad:
                    raise RuntimeError(
                        "In-place operations (device switch) are forbidden on differentiable tensors."
                    )

                self.device = device
                self.prev = () if clear_prev else self.prev
                self.data = tensordata_to_device(self.data, device=device)
                return self
            return self.copy(device=device)

    def get_device(self) -> Device:
        """
        Get the device of the tensor.

        Returns
        -------
        Device
            The device of the tensor ('cpu' or 'gpu').
        """
        return self.device

    def is_device(self, device: Device) -> bool:
        """
        Check if the tensor is on the specified device.

        Parameters
        ----------
        device : Device
            The device to check against.

        Returns
        -------
        bool
            True if the tensor is on the specified device, False otherwise.
        """
        return self.device == device

    def is_cpu(self) -> bool:
        """
        Check if the tensor is on the CPU.

        Returns
        -------
        bool
            True if the tensor is on the CPU, False otherwise.
        """
        return self.device == "cpu"

    def is_gpu(self) -> bool:
        """
        Check if the tensor is on the GPU.

        Returns
        -------
        bool
            True if the tensor is on the GPU, False otherwise.
        """
        return self.device == "gpu"

    def set_name(self, name: str) -> Tensor:
        """
        Set the name of the tensor.

        Parameters
        ----------
        name : str
            The name to set.

        Returns
        -------
        Tensor
            The tensor itself (for method chaining).
        """
        self.name = name
        return self

    def transpose(self, axes: Shape | None) -> Tensor:
        """
        Permute the dimensions of the tensor.

        Parameters
        ----------
        axes : Shape | None
            A list of integers. By default, reverse the dimensions,
            otherwise permute the axes according to the values given.

        Returns
        -------
        Tensor
            The transposed tensor.

        Raises
        ------
        CuPyNotFound
            If the device is 'gpu' and CuPy is not available.
        """
        y = Tensor._fast_init(
            self.data.transpose(axes),
            device=self.device,
            requires_grad=self.requires_grad,
        )
        if y.requires_grad:

            def backward() -> None:
                if y.grad is None:
                    y.zero_grad()
                if self.grad is None:
                    self.zero_grad()

                if axes is None:
                    self.grad += y.grad.transpose(None)  # type: ignore
                else:
                    if self.is_cpu():
                        inverse_axes = np.argsort(axes)
                    else:
                        if cp is None:
                            raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                        inverse_axes = cp.argsort(axes)
                    self.grad += y.grad.transpose(inverse_axes)  # type: ignore

            y.prev = (self,)
            y.backward = backward
        return y

    ###
    ### Property
    ###

    @property
    def dtype(self) -> DTypeLike:
        """
        Data type of the tensor's elements.
        """
        return self.data.dtype

    @property
    def shape(self) -> Shape:
        """
        Tuple of tensor dimensions.
        """
        return self.data.shape

    @property
    def ndim(self) -> int:
        """
        Number of tensor dimensions.
        """
        return self.data.ndim

    @property
    def size(self) -> int:
        """
        Number of elements in the tensor.
        """
        self.item
        return self.data.size

    def item(self, **kwargs: Any) -> Any:
        """
        Copy an element of an array to a standard Python scalar and return it.

        Parameters
        ----------
        **kwargs : Any
            Arguments passed to the underlying item() method.

        Returns
        -------
        Any
            The element as a standard Python scalar.
        """
        return self.data.item(**kwargs)

    ###
    ### Element-wise
    ###

    def sqrt(
        self,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
    ) -> Tensor:
        """
        Return the non-negative square-root of the tensor, element-wise.

        Parameters
        ----------
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.

        Returns
        -------
        Tensor
            A new tensor with the square-root of each element.
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
            differentiate=self.requires_grad,
        )

    def square(
        self,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
    ) -> Tensor:
        """
        Return the element-wise square of the tensor.

        Parameters
        ----------
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.

        Returns
        -------
        Tensor
            A new tensor with the square of each element.
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
            differentiate=self.requires_grad,
        )

    def log(
        self,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
    ) -> Tensor:
        """
        Natural logarithm, element-wise.

        Parameters
        ----------
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.

        Returns
        -------
        Tensor
            A new tensor with the natural logarithm of each element.
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
            differentiate=self.requires_grad,
        )

    def exp(
        self,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
    ) -> Tensor:
        """
        Calculate the exponential of all elements in the tensor.

        Parameters
        ----------
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.

        Returns
        -------
        Tensor
            A new tensor with the exponential of each element.
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
            differentiate=self.requires_grad,
        )

    def sign(
        self,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
    ) -> Tensor:
        """
        Returns an element-wise indication of the sign of a number.

        Parameters
        ----------
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.

        Returns
        -------
        Tensor
            A new tensor with the sign of each element.
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
            differentiate=self.requires_grad,
        )

    def abs(
        self,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
    ) -> Tensor:
        """
        Calculate the absolute value element-wise.

        Parameters
        ----------
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.

        Returns
        -------
        Tensor
            A new tensor with the absolute value of each element.
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
            differentiate=self.requires_grad,
        )

    def neg(
        self,
        out: TensorData | None = None,
        *,
        device: Device | None = None,
        in_place: bool = False,
        where: TensorDataBool | bool = True,
        casting: Casting = "same_kind",
        order: Order = "K",
        dtype: DTypeLike | None = None,
        subok: bool = True,
    ) -> Tensor:
        """
        Numerical negative, element-wise.

        Parameters
        ----------
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.

        Returns
        -------
        Tensor
            A new tensor with the negative of each element.
        """
        return negative(
            self,
            out=out,
            device=device,
            in_place=in_place,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
            differentiate=self.requires_grad,
        )

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
        Clip (limit) the values in the tensor.

        Parameters
        ----------
        a_min : TensorLike
            Minimum value.
        a_max : TensorLike
            Maximum value.
        out : TensorData | None, optional
            The results will be placed in this array.
        requires_grad : bool | None, optional
            Whether the result requires gradient computation. If None, inferred from inputs.
        device : Device | None, optional
            The device to place the result on. If None, inferred from inputs.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input. At locations where the
            condition is True, the out array will be set to the ufunc result.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.

        Returns
        -------
        Tensor
            A tensor with the elements of this tensor, but where values
            < a_min are replaced with a_min, and those > a_max with a_max.
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
    ) -> Tensor:
        """
        Returns the average of the array elements.

        Parameters
        ----------
        axis : Axis | None, optional
            Axis or axes along which the means are computed.
        dtype : DTypeLike | None, optional
            Type to use in computing the mean.
        out : TensorData | None, optional
            Alternate output array in which to place the result.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place.
        where : TensorDataBool | bool, optional
            Elements to include in the mean.

        Returns
        -------
        Tensor
            A new tensor containing the mean values.
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
            differentiate=self.requires_grad,
        )

    ###
    ### Reduction
    ###

    def sum(
        self,
        /,
        *,
        device: Device | None = None,
        requires_grad: bool | None = None,
        axis: Axis | None = None,
        dtype: DTypeLike | None = None,
        out: TensorData | None = None,
        keepdims: bool = True,
        initial: Any = NoValue,
        where: TensorDataBool | bool = True,
    ) -> Tensor:
        """
        Sum of array elements over a given axis.

        Parameters
        ----------
        device : Device | None, optional
            The device to place the result on. If None, inferred from input.
        requires_grad : bool | None, optional
            Whether the result requires gradient computation. If None, inferred from input.
        axis : Axis | None, optional
            Axis or axes along which a sum is performed.
        dtype : DTypeLike | None, optional
            The type of the returned array and of the accumulator in which
            the elements are summed.
        out : TensorData | None, optional
            Alternative output array in which to place the result.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. Default is True.
        initial : Any, optional
            Starting value for the sum.
        where : TensorDataBool | bool, optional
            Elements to include in the sum.

        Returns
        -------
        Tensor
            An array with the same shape as a, with the specified axis removed.
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
        Return the maximum of an array or maximum along an axis.

        Parameters
        ----------
        device : Device | None, optional
            The device to place the result on. If None, inferred from input.
        requires_grad : bool | None, optional
            Whether the result requires gradient computation. If None, inferred from input.
        axis : Axis | None, optional
            Axis or axes along which to operate.
        out : TensorData | None, optional
            Alternative output array in which to place the result.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. Default is False.
        initial : Any, optional
            The minimum value of an output element.
        where : TensorDataBool | bool, optional
            Elements to compare.

        Returns
        -------
        Tensor
            Maximum of a.
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
        Element-wise maximum of array elements.

        Parameters
        ----------
        x2 : TensorLike
            The array holding the elements to compare.
        out : TensorData | None, optional
            The results will be placed in this array.
        device : Device | None, optional
            The device to place the result on. If None, inferred from inputs.
        requires_grad : bool | None, optional
            Whether the result requires gradient computation. If None, inferred from inputs.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            Controls the memory layout of the result.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.

        Returns
        -------
        Tensor
            The maximum of self and x2, element-wise.
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

    ###
    ### Dunder Operations
    ###

    def __add__(self, t: TensorLike) -> Tensor:
        return add(self, t, differentiate=infer_differentiation_status(self, t))

    def __radd__(self, t: TensorLike) -> Tensor:
        return add(t, self, differentiate=infer_differentiation_status(self, t))

    def __iadd__(self, t: TensorLike) -> Tensor:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations are (add) forbidden on differentiable tensors."
            )
        return add(self, t, in_place=True)

    def __sub__(self, t: TensorLike) -> Tensor:
        return subtract(self, t, differentiate=infer_differentiation_status(self, t))

    def __rsub__(self, t: TensorLike) -> Tensor:
        return subtract(t, self, differentiate=infer_differentiation_status(self, t))

    def __isub__(self, t: TensorLike) -> Tensor:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations (sub) are forbidden on differentiable tensors."
            )
        return subtract(self, t, in_place=True)

    def __mul__(self, t: TensorLike) -> Tensor:
        return multiply(self, t, differentiate=infer_differentiation_status(self, t))

    def __rmul__(self, t: TensorLike) -> Tensor:
        return multiply(t, self, differentiate=infer_differentiation_status(self, t))

    def __imul__(self, t: TensorLike) -> Tensor:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations are (mul) forbidden on differentiable tensors."
            )
        return multiply(self, t, in_place=True)

    def __matmul__(self, t: TensorLike) -> Tensor:
        return matmul(self, t, differentiate=infer_differentiation_status(self, t))

    def __rmatmul__(self, t: TensorLike) -> Tensor:
        return matmul(t, self, differentiate=infer_differentiation_status(self, t))

    def __imatmul__(self, t: TensorLike) -> Tensor:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations are (matmul) forbidden on differentiable tensors."
            )
        return matmul(self, t, in_place=True)

    def __truediv__(self, t: TensorLike) -> Tensor:
        return divide(self, t, differentiate=infer_differentiation_status(self, t))

    def __rtruediv__(self, t: TensorLike) -> Tensor:
        return divide(t, self, differentiate=infer_differentiation_status(self, t))

    def __itruediv__(self, t: TensorLike) -> Tensor:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations (div) are forbidden on differentiable tensors."
            )
        return divide(self, t, in_place=True)

    def __pow__(self, t: TensorLike) -> Tensor:
        return power(self, t, differentiate=infer_differentiation_status(self, t))

    def __ipow__(self, t: TensorLike) -> Tensor:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations (pow) are forbidden on differentiable tensors."
            )
        return power(self, t, in_place=True)

    ###
    ### Unary
    ###

    def __eq__(self, value: Any) -> Tensor:  # type: ignore[override]
        return equal(self, value)

    def __ne__(self, value: Any) -> Tensor:  # type: ignore[override]
        return not_equal(self, value)

    def __gt__(self, value: Any) -> Tensor:  # type: ignore[override]
        return greater(self, value)

    def __ge__(self, value: Any) -> Tensor:  # type: ignore[override]
        return greater_equal(self, value)

    def __lt__(self, value: Any) -> Tensor:  # type: ignore[override]
        return less(self, value)

    def __le__(self, value: Any) -> Tensor:  # type: ignore[override]
        return less_equal(self, value)

    def __neg__(self) -> Tensor:
        return negative(self, differentiate=self.requires_grad)

    def __abs__(self) -> Tensor:
        return abs(self, differentiate=self.requires_grad)

    ###
    ### Misc dunder
    ###

    def __getitem__(self, index) -> Any:
        y = Tensor._fast_init(
            self.data[index],
            device=self.device,
            requires_grad=self.requires_grad,
        )
        if y.requires_grad:

            def backward() -> None:
                if y.grad is None:
                    y.zero_grad()
                if self.grad is None:
                    self.zero_grad()
                self.grad[index] += y.grad  # type: ignore
                y.prev = (self,)

            y.backward = backward
        return y

    def __setitem__(self, index, value) -> Any:
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations (indexing) are forbidden on differentiable tensors."
            )
        self.data[index] = value

    def __iter__(self) -> Iterator:
        return iter(self.data)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, device={self.device}{f", name={self.name}" if self.name != "" else ""}, req_grad={self.requires_grad})"

    def __hash__(self) -> int:
        return id(self)
