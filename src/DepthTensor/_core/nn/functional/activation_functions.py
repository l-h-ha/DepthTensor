from ....typing import TensorType
from .._backend import relu as _relu_op


def relu(x: TensorType, in_place: bool = False) -> TensorType:
    """
    Applies the rectified linear unit function element-wise.

    Parameters
    ----------
    x : Tensor
        The input tensor.
    in_place : bool, optional
        If True, executes the operation in-place. Default is False.

    Returns
    -------
    Tensor
        The output tensor.
    """
    return _relu_op(x, in_place=in_place)
