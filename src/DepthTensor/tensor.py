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
    GradientComputationError,
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
    exp,
    log,
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

from ._core.utils import (
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

###
###
###


def _infer_differentiation_status(a: Tensor, b: TensorLike) -> bool:
    if isinstance(b, Tensor):
        return a.requires_grad or b.requires_grad
    return a.requires_grad


def _check_grad_compatibility(grad: TensorData, tensor: Tensor) -> tuple[bool, str]:
    a = get_device(grad) == tensor.device
    b = grad.shape == tensor.shape
    c = grad.dtype == tensor.dtype

    incompatibility = ""

    if not a:
        incompatibility += "device, "
    if not b:
        incompatibility += "shape, "
    if not c:
        incompatibility += "dtype, "

    incompatibility = incompatibility[:-2]
    return a and b and c, incompatibility


###
###
###


class Tensor:
    """
    A multi-dimensional array (tensor) with automatic differentiation support.

    This class provides a wrapper around NumPy and CuPy arrays, enabling
    CUDA acceleration and automatic gradient computation.

    Attributes
    ----------
    data : TensorData
        The underlying data of the tensor (numpy.ndarray or cupy.ndarray).
    device : Device
        The device where the tensor data resides ('cpu' or 'cuda').
    grad : TensorData | None
        The gradient of the tensor. None if no gradient is computed.
    grad_fn : Callable[[], None] | None
        The function that updates the parent tensor's gradient. None if no
        gradient is computed.
    requires_grad : bool
        Whether the tensor requires gradient computation.
    name : str
        Optional name for the tensor.
    """

    data: TensorData
    device: Device
    grad: TensorData | None
    grad_fn: Callable[[], None] | None
    requires_grad: bool
    name: str

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
            The device to place the tensor on ('cpu' or 'cuda'). If None,
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
        self.grad_fn = None
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
        obj.grad_fn = None
        obj.name = name
        return obj

    def zero_grad(self) -> None:
        """
        Sets the gradients of the tensor to zeros.

        Raises
        ------
        RuntimeError
            If the tensor does not require gradients.
        CuPyNotFound
            If the device is 'cuda' and CuPy is not available.
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

    def one_grad(self) -> None:
        """
        Sets the gradients of the tensor to ones.

        Raises
        ------
        RuntimeError
            If the tensor does not require gradients.
        CuPyNotFound
            If the device is 'cuda' and CuPy is not available.
        """
        if not self.requires_grad:
            raise RuntimeError(
                "Attempted to zero-gradient initialize an undifferentiable tensor."
            )
        if self.grad is None:
            if self.device == "cpu":
                grad = np.ones_like(self.data)
            else:
                if cp is None:
                    raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
                grad = cp.ones_like(self.data)
            self.grad = grad
        else:
            self.grad.fill(1)

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

    def toggle_grad(
        self,
        state: bool | None = None,
        initialize_grad: InitializeGrad = InitializeGrad.NONE,
    ) -> None:
        """
        Toggles whether the tensor requires gradient computation.

        Parameters
        ----------

        state : bool | None, optional
            Whether to toggle the requires_grad flag. Default is None.
            If None, the requires_grad flag is toggled (True -> False, False -> True).

        initialize_grad : InitializeGrad
            Gradient initialization mode. Default is `InitializeGrad.NONE`.


        Raises
        ------
        RuntimeError
            If the tensor's gradient is not compatible with the tensor in terms of shape, device, or dtype.
        """
        if state is None:
            self.requires_grad = not self.requires_grad
        else:
            self.requires_grad = state

        if self.requires_grad:
            # * Toggle from False to True.
            if self.grad is not None:
                comp_status, incomp = _check_grad_compatibility(self.grad, self)
                if not comp_status:
                    raise RuntimeError(
                        f"Incompability between tensor and its gradient in terms of: {incomp}"
                    )

            if initialize_grad == InitializeGrad.ZEROS:
                self.zero_grad()
            elif initialize_grad == InitializeGrad.ONES:
                self.one_grad()
            elif initialize_grad == InitializeGrad.NONE:
                pass
        else:
            # * Toggle from True to False.
            if self.grad is not None:
                self.grad = None
                self.grad_fn = None

    def to_device(
        self, device: Device, in_place: bool = False, clear_prev: bool = True
    ) -> Tensor:
        """
        Moves the tensor to the specified device.

        If `in_place` is `true` and:
        - if `device` is the same as the object's device -> returns the object itself, regardless of `requires_grad`.
        - if `device` is not the same as the object's device -> raises `RuntimeError`.

        If `in_place` is `false` -> returns a new tensor on the specified device.

        Parameters
        ----------
        device : Device
            The target device ('cpu' or 'cuda').
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
            The device of the tensor ('cpu' or 'cuda').
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

    def is_cuda(self) -> bool:
        """
        Check if the tensor is a CUDA tensor.

        Returns
        -------
        bool
            True if the tensor is a CUDA tensor, False otherwise.
        """
        return self.device == "cuda"

    def is_contiguous(self) -> bool:
        """
        Check if the tensor is contiguous in memory.

        Returns
        -------
        bool
            True if the tensor is contiguous, False otherwise.
        """
        if self.is_cpu():
            return self.data.flags["C_CONTIGUOUS"]
        else:
            if cp is None:
                raise CuPyNotFound(CUPY_NOT_FOUND_MSG)
            return self.data.flags.c_contiguous

    def contiguous(self) -> Tensor:
        """
        Creates a clone of the tensor with the data contiguous in memory.

        Returns
        -------
        Tensor
            The tensor with contiguous memory.
        """
        if self.is_contiguous():
            return self

        if self.is_cpu():
            new_data = np.ascontiguousarray(self.data)
        else:
            new_data = cp.ascontiguousarray(self.data)

        y = Tensor._fast_init(
            new_data,
            device=self.device,
            requires_grad=self.requires_grad,
        )

        if y.requires_grad:

            def grad_fn() -> None:
                if y.grad is None:
                    y.zero_grad()
                if self.grad is None:
                    self.zero_grad()
                self.grad += y.grad

            y.prev = (self,)
            y.grad_fn = grad_fn

        return y

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
            If the device is 'cuda' and CuPy is not available.
        """
        y = Tensor._fast_init(
            self.data.transpose(axes),
            device=self.device,
            requires_grad=self.requires_grad,
        )
        if y.requires_grad:

            def grad_fn() -> None:
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
            y.grad_fn = grad_fn
        return y

    def backward(self, grad: TensorData | None = None) -> list[Tensor]:
        """
        Compute the gradients of the tensor with respect to its ancestors.

        This function performs reverse-mode automatic differentiation (backpropagation).
        It traverses the computation graph backwards from the given tensor and
        computes the gradients for all tensors that require gradients.

        Parameters
        ----------
        grad : TensorData | None, optional
            The gradient to seed the backward pass with. If None, it defaults to
            ones with the same shape as `tensor`.

        Returns
        -------
        list[Tensor]
            A list of tensors in the computation graph in topological order (reversed).

        Raises
        ------
        RuntimeError
            If there is a mismatch in gradient device or shape.
        GradientComputationError
            If a tensor in the graph is missing a backward function but requires gradients.
        CuPyNotFound
            If the device is 'cuda' and CuPy is not available.
        """
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(t: Tensor):
            if t in visited:
                return
            visited.add(t)

            for prev in t.prev:
                if isinstance(prev, Tensor):
                    build(prev)
            topo.append(t)

        build(self)

        if grad is not None:
            if self.device != grad.device:
                raise RuntimeError("Tensor's device does not match grad's device.")
            if self.shape != grad.shape:
                raise RuntimeError("Tensor's shape does not match grad's shape.")
            self.grad = grad
        else:
            self.one_grad()

        for t in reversed(topo):
            if t.grad_fn is None:
                if not t.requires_grad:
                    continue
                if len(t.prev) > 0:
                    raise GradientComputationError(
                        f"Tensor ({t})'s backward function is None."
                    )
                continue
            t.grad_fn()
        return topo

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

    def clip(
        self,
        a_min: TensorLike,
        a_max: TensorLike,
        /,
        out: TensorData | None = None,
        *,
        requires_grad: bool = False,
        device: Device = "cpu",
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
        requires_grad : bool, optional
            Whether the result requires gradient computation. Default is False.
        device : Device, optional
            The device to place the result on. Default is 'cpu'.
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
            requires_grad=requires_grad or self.requires_grad,
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
        requires_grad: bool = False,
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
        requires_grad : bool, optional
            If True, the resulting tensor will be differentiable.

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
            requires_grad=requires_grad or self.requires_grad,
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
        requires_grad: bool = False,
    ) -> Tensor:
        """
        Return the non-negative square-root of an array, element-wise.

        Parameters
        ----------
        out : TensorData | None, optional
            A location into which the result is stored. If provided, it must have
            a shape that the inputs broadcast to.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place on the input tensor.
        where : TensorDataBool | bool, optional
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            The order of memory layout to use for the output.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through, otherwise
            the returned array will be forced to be a base-class array.
        requires_grad : bool, optional
            If True, the resulting tensor will be differentiable.

        Returns
        -------
        Tensor
            A new tensor containing the square root of each element.
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
            requires_grad=requires_grad or self.requires_grad,
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
        requires_grad: bool = False,
    ) -> Tensor:
        """
        Return the element-wise square of the input.

        Parameters
        ----------
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place on the input tensor.
        where : TensorDataBool | bool, optional
            Elements to include in the operation.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            The order of memory layout to use for the output.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        requires_grad : bool, optional
            If True, the resulting tensor will be differentiable.

        Returns
        -------
        Tensor
            A new tensor with the squared value of each element.
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
            requires_grad=requires_grad or self.requires_grad,
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
        requires_grad: bool = False,
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
            Whether to perform the operation in-place on the input tensor.
        where : TensorDataBool | bool, optional
            Elements to include in the operation.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            The order of memory layout to use for the output.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        requires_grad : bool, optional
            If True, the resulting tensor will be differentiable.

        Returns
        -------
        Tensor
            The natural logarithm of x, element-wise.
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
            requires_grad=requires_grad or self.requires_grad,
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
        requires_grad: bool = False,
    ) -> Tensor:
        """
        Calculate the exponential of all elements in the input array.

        Parameters
        ----------
        out : TensorData | None, optional
            A location into which the result is stored.
        device : Device | None, optional
            The device to place the result on.
        in_place : bool, optional
            Whether to perform the operation in-place on the input tensor.
        where : TensorDataBool | bool, optional
            Elements to include in the operation.
        casting : Casting, optional
            Controls what kind of data casting may occur.
        order : Order, optional
            The order of memory layout to use for the output.
        dtype : DTypeLike | None, optional
            Overrides the data type of the result.
        subok : bool, optional
            If True, then sub-classes will be passed-through.
        requires_grad : bool, optional
            If True, the resulting tensor will be differentiable.

        Returns
        -------
        Tensor
            Element-wise exponential of the input.
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
            requires_grad=requires_grad or self.requires_grad,
        )

    ###

    ###
    ### Reduction
    ###

    def sum(
        self,
        /,
        *,
        device: Device = "cpu",
        requires_grad: bool = False,
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
        device : Device, optional
            The device to place the result on. Default is 'cpu'.
        requires_grad : bool, optional
            Whether the result requires gradient computation. Default is False.
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
            requires_grad=requires_grad or self.requires_grad,
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
        device: Device = "cpu",
        requires_grad: bool = False,
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
        device : Device, optional
            The device to place the result on. Default is 'cpu'.
        requires_grad : bool, optional
            Whether the result requires gradient computation. Default is False.
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
            requires_grad=requires_grad or self.requires_grad,
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
        device: Device = "cpu",
        requires_grad: bool = False,
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
        device : Device, optional
            The device to place the result on. Default is 'cpu'.
        requires_grad : bool, optional
            Whether the result requires gradient computation. Default is False.
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
            requires_grad=requires_grad or self.requires_grad,
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
        """
        Element-wise addition.

        Parameters
        ----------
        t : TensorLike
            The value to add.

        Returns
        -------
        Tensor
            Result of addition.
        """
        return add(self, t, requires_grad=_infer_differentiation_status(self, t))

    def __radd__(self, t: TensorLike) -> Tensor:
        """
        Element-wise addition (reflected).

        Parameters
        ----------
        t : TensorLike
            The value to add.

        Returns
        -------
        Tensor
            Result of addition.
        """
        return add(t, self, requires_grad=_infer_differentiation_status(self, t))

    def __iadd__(self, t: TensorLike) -> Tensor:
        """
        In-place element-wise addition.

        Parameters
        ----------
        t : TensorLike
            The value to add.

        Returns
        -------
        Tensor
            Self, after update.

        Raises
        ------
        RuntimeError
            If the tensor requires gradients.
        """
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations are (add) forbidden on differentiable tensors."
            )
        return add(self, t, in_place=True)

    def __sub__(self, t: TensorLike) -> Tensor:
        """
        Element-wise subtraction.

        Parameters
        ----------
        t : TensorLike
            The value to subtract.

        Returns
        -------
        Tensor
            Result of subtraction.
        """
        return subtract(self, t, requires_grad=_infer_differentiation_status(self, t))

    def __rsub__(self, t: TensorLike) -> Tensor:
        """
        Element-wise subtraction (reflected).

        Parameters
        ----------
        t : TensorLike
            The value to subtract from.

        Returns
        -------
        Tensor
            Result of subtraction.
        """
        return subtract(t, self, requires_grad=_infer_differentiation_status(self, t))

    def __isub__(self, t: TensorLike) -> Tensor:
        """
        In-place element-wise subtraction.

        Parameters
        ----------
        t : TensorLike
            The value to subtract.

        Returns
        -------
        Tensor
            Self, after update.

        Raises
        ------
        RuntimeError
            If the tensor requires gradients.
        """
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations (sub) are forbidden on differentiable tensors."
            )
        return subtract(self, t, in_place=True)

    def __mul__(self, t: TensorLike) -> Tensor:
        """
        Element-wise multiplication.

        Parameters
        ----------
        t : TensorLike
            The value to multiply.

        Returns
        -------
        Tensor
            Result of multiplication.
        """
        return multiply(self, t, requires_grad=_infer_differentiation_status(self, t))

    def __rmul__(self, t: TensorLike) -> Tensor:
        """
        Element-wise multiplication (reflected).

        Parameters
        ----------
        t : TensorLike
            The value to multiply.

        Returns
        -------
        Tensor
            Result of multiplication.
        """
        return multiply(t, self, requires_grad=_infer_differentiation_status(self, t))

    def __imul__(self, t: TensorLike) -> Tensor:
        """
        In-place element-wise multiplication.

        Parameters
        ----------
        t : TensorLike
            The value to multiply.

        Returns
        -------
        Tensor
            Self, after update.

        Raises
        ------
        RuntimeError
            If the tensor requires gradients.
        """
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations are (mul) forbidden on differentiable tensors."
            )
        return multiply(self, t, in_place=True)

    def __matmul__(self, t: TensorLike) -> Tensor:
        """
        Matrix multiplication.

        Parameters
        ----------
        t : TensorLike
            The value to multiply.

        Returns
        -------
        Tensor
            Result of matrix multiplication.
        """
        return matmul(self, t, requires_grad=_infer_differentiation_status(self, t))

    def __rmatmul__(self, t: TensorLike) -> Tensor:
        """
        Matrix multiplication (reflected).

        Parameters
        ----------
        t : TensorLike
            The value to multiply.

        Returns
        -------
        Tensor
            Result of matrix multiplication.
        """
        return matmul(t, self, requires_grad=_infer_differentiation_status(self, t))

    def __imatmul__(self, t: TensorLike) -> Tensor:
        """
        In-place matrix multiplication.

        Parameters
        ----------
        t : TensorLike
            The value to multiply.

        Returns
        -------
        Tensor
            Self, after update.

        Raises
        ------
        RuntimeError
            If the tensor requires gradients.
        """
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations are (matmul) forbidden on differentiable tensors."
            )
        return matmul(self, t, in_place=True)

    def __truediv__(self, t: TensorLike) -> Tensor:
        """
        Element-wise division.

        Parameters
        ----------
        t : TensorLike
            The divisor.

        Returns
        -------
        Tensor
            Result of division.
        """
        return divide(self, t, requires_grad=_infer_differentiation_status(self, t))

    def __rtruediv__(self, t: TensorLike) -> Tensor:
        """
        Element-wise division (reflected).

        Parameters
        ----------
        t : TensorLike
            The dividend.

        Returns
        -------
        Tensor
            Result of division.
        """
        return divide(t, self, requires_grad=_infer_differentiation_status(self, t))

    def __itruediv__(self, t: TensorLike) -> Tensor:
        """
        In-place element-wise division.

        Parameters
        ----------
        t : TensorLike
            The divisor.

        Returns
        -------
        Tensor
            Self, after update.

        Raises
        ------
        RuntimeError
            If the tensor requires gradients.
        """
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations (div) are forbidden on differentiable tensors."
            )
        return divide(self, t, in_place=True)

    def __pow__(self, t: TensorLike) -> Tensor:
        """
        Element-wise power.

        Parameters
        ----------
        t : TensorLike
            The exponent.

        Returns
        -------
        Tensor
            Result of power.
        """
        return power(self, t, requires_grad=_infer_differentiation_status(self, t))

    def __ipow__(self, t: TensorLike) -> Tensor:
        """
        In-place element-wise power.

        Parameters
        ----------
        t : TensorLike
            The exponent.

        Returns
        -------
        Tensor
            Self, after update.

        Raises
        ------
        RuntimeError
            If the tensor requires gradients.
        """
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations (pow) are forbidden on differentiable tensors."
            )
        return power(self, t, in_place=True)

    ###
    ### Unary
    ###

    def __eq__(self, value: Any) -> Tensor:  # type: ignore[override]
        """
        Element-wise equality comparison.

        Parameters
        ----------
        value : Any
            The value to compare with.

        Returns
        -------
        Tensor
            Boolean tensor of the comparison result.
        """
        return equal(self, value)

    def __ne__(self, value: Any) -> Tensor:  # type: ignore[override]
        """
        Element-wise inequality comparison.

        Parameters
        ----------
        value : Any
            The value to compare with.

        Returns
        -------
        Tensor
            Boolean tensor of the comparison result.
        """
        return not_equal(self, value)

    def __gt__(self, value: Any) -> Tensor:  # type: ignore[override]
        """
        Element-wise greater than comparison.

        Parameters
        ----------
        value : Any
            The value to compare with.

        Returns
        -------
        Tensor
            Boolean tensor of the comparison result.
        """
        return greater(self, value)

    def __ge__(self, value: Any) -> Tensor:  # type: ignore[override]
        """
        Element-wise greater than or equal comparison.

        Parameters
        ----------
        value : Any
            The value to compare with.

        Returns
        -------
        Tensor
            Boolean tensor of the comparison result.
        """
        return greater_equal(self, value)

    def __lt__(self, value: Any) -> Tensor:  # type: ignore[override]
        """
        Element-wise less than comparison.

        Parameters
        ----------
        value : Any
            The value to compare with.

        Returns
        -------
        Tensor
            Boolean tensor of the comparison result.
        """
        return less(self, value)

    def __le__(self, value: Any) -> Tensor:  # type: ignore[override]
        """
        Element-wise less than or equal comparison.

        Parameters
        ----------
        value : Any
            The value to compare with.

        Returns
        -------
        Tensor
            Boolean tensor of the comparison result.
        """
        return less_equal(self, value)

    def __neg__(self) -> Tensor:
        """
        Element-wise negation.

        Returns
        -------
        Tensor
            Negated tensor.
        """
        return negative(self, requires_grad=self.requires_grad)

    ###
    ### Misc dunder
    ###

    def __getitem__(self, index) -> Any:
        """
        Return the item at the given index.

        Parameters
        ----------
        index : int, slice, tuple, or Tensor
            The index or slice to retrieve.

        Returns
        -------
        Any
            The item or tensor at the specified index.
        """
        y = Tensor._fast_init(
            self.data[index],
            device=self.device,
            requires_grad=self.requires_grad,
        )
        if y.requires_grad:

            def grad_fn() -> None:
                if y.grad is None:
                    y.zero_grad()
                if self.grad is None:
                    self.zero_grad()
                self.grad[index] += y.grad  # type: ignore

            y.prev = (self,)
            y.grad_fn = grad_fn
        return y

    def __setitem__(self, index, value) -> Any:
        """
        Set the item at the given index.

        Parameters
        ----------
        index : int, slice, tuple, or Tensor
            The index or slice to set.
        value : Any
            The value to set.

        Raises
        ------
        RuntimeError
            If the tensor requires gradients (in-place modification forbidden).
        """
        if self.requires_grad:
            raise RuntimeError(
                "In-place operations (indexing) are forbidden on differentiable tensors."
            )
        self.data[index] = value

    def __iter__(self) -> Iterator:
        """
        Return an iterator over the tensor.

        Returns
        -------
        Iterator
            An iterator over the tensor's data.
        """
        return iter(self.data)

    def __repr__(self) -> str:
        """
        Return a string representation of the tensor.

        Returns
        -------
        str
            String representation.
        """
        return f"Tensor({self.data}, device={self.device}{f", name={self.name}" if self.name != "" else ""}, req_grad={self.requires_grad})"

    def __hash__(self) -> int:
        """
        Return the hash of the tensor.

        Returns
        -------
        int
            Hash value.
        """
        return id(self)
