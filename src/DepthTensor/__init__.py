from .tensor import Tensor
from .typing import (
    Shape,
    Axis,
    Device,
    Order,
    Casting,
    DTypeLike,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    double,
    TensorType,
    TensorData,
    TensorDataBool,
    TensorLike,
)

from ._core.ops.comparison import (
    where,
    not_equal,
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
)
from ._core.ops.creation import (
    zeros_like,
    ones_like,
    zeros,
    ones,
)
from ._core.ops.reduction import sum, max, maximum
from ._core.ops.elementwise import (
    add,
    subtract,
    multiply,
    matmul,
    divide,
    power,
    negative,
    sign,
    abs,
    square,
    log,
    sqrt,
    exp,
    clip,
    mean,
)

from ._core.ops.function import Function
from ._core.random.generator import random, rand, randint, randn

from ._core.exceptions import (
    CuPyNotFound,
    CUPY_NOT_FOUND_MSG,
    DeviceMismatch,
    DEVICE_MISMATCH_MSG,
    OperandMismatch,
    OPERAND_MISMATCH_MSG,
    GradientComputationError,
    GRADIENT_COMPUTATION_ERROR,
)

from ._core.nn.functional.activation_functions import relu
from .enum.initialize_grad import InitializeGrad

__version__ = "2.9.0rc3"

__all__ = [
    # Core Tensor
    "Tensor",
    # Typing
    "Shape",
    "Axis",
    "Device",
    "Order",
    "Casting",
    "DTypeLike",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "double",
    "TensorType",
    "TensorData",
    "TensorDataBool",
    "TensorLike",
    # Comparison Ops
    "where",
    "not_equal",
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    # Creation Ops
    "zeros_like",
    "ones_like",
    "zeros",
    "ones",
    # Reduction Ops
    "sum",
    "max",
    "maximum",
    # Elementwise Ops
    "add",
    "subtract",
    "multiply",
    "matmul",
    "divide",
    "power",
    "negative",
    "sign",
    "abs",
    "square",
    "log",
    "sqrt",
    "exp",
    "clip",
    "mean",
    # Functions
    "Function",
    # Random
    "random",
    "rand",
    "randint",
    "randn",
    # Exceptions
    "CuPyNotFound",
    "CUPY_NOT_FOUND_MSG",
    "DeviceMismatch",
    "DEVICE_MISMATCH_MSG",
    "OperandMismatch",
    "OPERAND_MISMATCH_MSG",
    "GradientComputationError",
    "GRADIENT_COMPUTATION_ERROR",
    # NN
    "relu",
    # Enums
    "InitializeGrad",
]
