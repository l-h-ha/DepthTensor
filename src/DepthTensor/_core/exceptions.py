CUPY_NOT_FOUND_MSG = "Module CuPy not found or installed. Please install CuPy."
DEVICE_MISMATCH_MSG = "There is a mismatch in device between two objects."
GRADIENT_COMPUTATION_ERROR = "An error in gradient computation has occurred."

class CuPyNotFound(RuntimeError):
    ...

class DeviceMismatch(RuntimeError):
    ...

class GradientComputationError(RuntimeError):
    ...