from ....typing import TensorType
from .._backend import relu as _relu_op

def relu(x: TensorType, in_place: bool = False) -> TensorType:
    """
    Apply ReLU.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    in_place : bool, optional
        Update `x` in place.

    Returns
    -------
    Tensor
        ReLU values. In-place updates return `x`.
    """
    return _relu_op(x, in_place=in_place)
