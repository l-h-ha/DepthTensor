CUPY_NOT_FOUND_MSG = "Install CuPy for CUDA."

class CuPyNotFound(RuntimeError): ...

DEVICE_MISMATCH_MSG = "Device mismatch."

class DeviceMismatch(RuntimeError): ...

OPERAND_MISMATCH_MSG = "Operand type mismatch."

class OperandMismatch(RuntimeError): ...

GRADIENT_COMPUTATION_ERROR = "Gradient computation failed."

class GradientComputationError(RuntimeError): ...
