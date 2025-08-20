CUPY_NOT_FOUND_MSG = "Module CuPy not found or installed. Please install CuPy."
DEVICE_MISMATCH_MSG = "There is a mismatch in device between two objects."
OPERAND_MISMATCH_MSG = "There is a mismatch in the type between two operands."
GRADIENT_COMPUTATION_ERROR = "An error in gradient computation has occurred."

class CuPyNotFound(RuntimeError):
    ...

class DeviceMismatch(RuntimeError):
    ...

class OperandMismatch(RuntimeError):
    ...

class GradientComputationError(RuntimeError):
    ...