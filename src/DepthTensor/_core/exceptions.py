CUPY_NOT_FOUND_MSG = "Module CuPy not found or installed. Please install CuPy."
DEVICE_MISMATCH_MSG = "There is a mismatch in device between two objects."

class CuPyNotFound(RuntimeError):
    ...

class DeviceMismatch(RuntimeError):
    ...