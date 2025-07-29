from .exceptions import CuPyNotFoundError
import numpy as np

try:
    import cupy as cp
except:
    cp = None

###
###
###

EXPECTED_OBJ_TYPE = "Expected {obj} of type: {expect}, got: {got}."
PROVIDED_OBJ_INVALID = "Provided {obj} is invalid, expected {expect}, got: {got}."
CUPY_NOT_FOUND = "Module CuPy not found or installed."
EXPECTED_OBJ_DEVICE = "Expected {obj} of device: {expect}, got: {got}."

###
###
###

def get_device(obj):
    if isinstance(obj, Tensor):
        return obj.device
    if isinstance(obj, (np.ndarray, np.floating, np.integer)):
        return "cpu"
    if cp is not None:
        if isinstance(obj, (cp.ndarray, cp.floating, cp.integer)):
            return "gpu"
    raise RuntimeError(EXPECTED_OBJ_TYPE.format(
        obj="object",
        expect="numpy.ndarray, or cupy.ndarray",
        got=type(obj),
    ))

def is_same_device(a, b, convert_if_necessary = False, raise_if_fail = False):
    if convert_if_necessary:
        a, b = convert_to_array(a), convert_to_array(b)
    if not get_device(a) == get_device(b):
        if raise_if_fail:
            raise RuntimeError("Provided objects differ in device.")
        return False
    return True

def is_cpu(obj, raise_if_fail = False):
    device = get_device(obj)
    if device == "cpu":
        return True
    if raise_if_fail:
        raise RuntimeError(EXPECTED_OBJ_DEVICE.format(
            obj="object",
            expect="cpu",
            got=device
        ))
    return False

def is_gpu(obj, raise_if_fail = False):
    device = get_device(obj)
    if device == "gpu":
        return True
    if raise_if_fail:
        raise RuntimeError(EXPECTED_OBJ_DEVICE.format(
            obj="object",
            expect="gpu",
            got=device
        ))
    return False


def validate_device_str(string, raise_if_fail = False):
    if string in ["cpu", "gpu"]:
        return True
    if raise_if_fail:
        raise RuntimeError(PROVIDED_OBJ_INVALID.format(
            obj="device",
            expect="cpu or gpu",
            got=string
        ))
    return False

def convert_to_array(obj, dtype = None, device = None):
    if device is None:
        device = "cpu"
    validate_device_str(device)
    if isinstance(obj, (np.ndarray, np.floating, np.integer)):
        if obj.dtype != dtype:
            obj = obj.astype(dtype=dtype)
        if device == "gpu":
            if cp is not None:
                obj = cp.asarray(obj)
            else:
                raise CuPyNotFoundError(CUPY_NOT_FOUND)
        return obj
    if cp is not None:
        if isinstance(obj, (cp.ndarray, cp.floating, cp.integer)):
            if obj.dtype != dtype:
                obj = obj.astype(dtype=dtype)
            if device == "cpu":
                obj = cp.asnumpy(obj)
            return obj
    if isinstance(obj, (int, float, list, tuple)):
        if device == "cpu":
            return np.asarray(obj, dtype=dtype)
        if cp is not None:
            return cp.asarray(obj, dtype=dtype)
    raise RuntimeError(EXPECTED_OBJ_TYPE.format(
        obj="object",
        expect="int, float, list, tuple, numpy.ndarray, or cupy.ndarray",
        got=type(obj),
    ))

###
###
###

def type_ok_for_operation(obj, raise_if_fail = False):
    if isinstance(obj, (int, float, list, tuple, np.ndarray, np.floating, np.integer, Tensor)):
        return True
    if cp is not None:
        if isinstance(obj, (cp.ndarray, cp.floating, cp.integer)):
            return True
    if raise_if_fail:
        raise RuntimeError(f"Expected objects of type: int, float, list, tuple, Tensor, numpy.ndarray, cupy.ndarray, got: {type(obj)}")
    return False

def double_operand_operation(a, b, callback, in_place = False):
    type_ok_for_operation(a, raise_if_fail=True)
    type_ok_for_operation(b, raise_if_fail=True)

    if isinstance(a, Tensor):
        if isinstance(b, Tensor):
            is_same_device(a, b, raise_if_fail=True)
            if is_cpu(a):
                if in_place:
                    a.data = callback(a.data, b.data, device="cpu")
                    return a
                return Tensor(data=callback(a.data, b.data, device="cpu"), device="cpu")
            if cp is not None and is_gpu(a):
                if in_place:
                    a.data = callback(a.data, b.data, device="gpu")
                    return a
                return Tensor(data=callback(a.data, b.data, device="gpu"), device="gpu")
        if isinstance(b, (np.ndarray, np.floating, np.integer)):
            is_cpu(a, raise_if_fail=True)
            if in_place:
                a.data = callback(a.data, b, device="cpu")
                return a
            return Tensor(data=callback(a.data, b, device="cpu"), device="cpu")
        if cp is not None:
            if isinstance(b, (cp.ndarray, cp.floating, cp.integer)):
                is_gpu(a, raise_if_fail=True)
                if in_place:
                    a.data = callback(a.data, b, device="gpu")
                    return a
                return Tensor(data=callback(a.data, b, device="gpu"), device="gpu")
        if isinstance(b, (int, float, list, tuple)):
            if is_cpu(a):
                if in_place:
                    a.data = callback(a.data, b, device="cpu")
                    return a
                return Tensor(data=callback(a.data, b, device="cpu"), device="cpu")
            if cp is not None and is_gpu(a):
                if in_place:
                    a.data = callback(a.data, b, device="gpu")
                    return a
                return Tensor(data=callback(a.data, b, device="gpu"), device="gpu")
        raise RuntimeError(f"Expected the second operand of type: int, float, list, tuple, Tensor, numpy.ndarray, cupy.ndarray, got: {type(b)}")
    if isinstance(b, Tensor):
        if isinstance(a, Tensor):
            is_same_device(b, a, raise_if_fail=True)
            if is_cpu(b):
                if in_place:
                    b.data = callback(b.data, a.data, device="cpu")
                    return b
                return Tensor(data=callback(b.data, a.data, device="cpu"), device="cpu")
            if cp is not None and is_gpu(b):
                if in_place:
                    b.data = callback(b.data, a.data, device="gpu")
                    return b
                return Tensor(data=callback(b.data, a.data, device="gpu"), device="gpu")
        if isinstance(a, (np.ndarray, np.floating, np.integer)):
            is_cpu(b, raise_if_fail=True)
            if in_place:
                b.data = callback(b.data, a, device="cpu")
                return b
            return Tensor(data=callback(b.data, a, device="cpu"), device="cpu")
        if cp is not None:
            if isinstance(a, (cp.ndarray, cp.floating, cp.integer)):
                is_gpu(b, raise_if_fail=True)
                if in_place:
                    b.data = callback(b.data, a, device="gpu")
                    return b
                return Tensor(data=callback(b.data, a, device="gpu"), device="gpu")
        if isinstance(a, (int, float, list, tuple)):
            if is_cpu(b):
                if in_place:
                    b.data = callback(b.data, a, device="cpu")
                    return b
                return Tensor(data=callback(b.data, a, device="cpu"), device="cpu")
            if cp is not None and is_gpu(b):
                if in_place:
                    b.data = callback(b.data, a, device="gpu")
                    return a
                return Tensor(data=callback(b.data, a, device="gpu"), device="gpu")
        raise RuntimeError(f"Expected the second operand of type: int, float, list, tuple, Tensor, numpy.ndarray, cupy.ndarray, got: {type(b)}")

def single_operand_operation(a, callback, in_place = False):
    #* a must be a Tensor
    if isinstance(a, Tensor):
        if in_place:
            if is_cpu(a):
                a.data = callback(a.data)
                return a
            a.data = callback(a.data)
            return a
        if is_cpu(a):
            return Tensor(data=callback(a.data), device="cpu")
        return Tensor(data=callback(a.data), device="gpu")
    raise RuntimeError(f"Expected a tensor, got: {type(a)}")

###
###
###

def add(a, b, dtype = None, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.add(data_a, data_b, dtype=dtype)
        if cp is not None:
            return cp.add(data_a, data_b, dtype=dtype)
    return double_operand_operation(a, b, callback=callback, in_place=in_place)

def subtract(a, b, dtype = None, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.subtract(data_a, data_b, dtype=dtype)
        if cp is not None:
            return cp.subtract(data_a, data_b, dtype=dtype)
    return double_operand_operation(a, b, callback=callback, in_place=in_place)

def multiply(a, b, dtype = None, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.multiply(data_a, data_b, dtype=dtype)
        if cp is not None:
            return cp.multiply(data_a, data_b, dtype=dtype)
    return double_operand_operation(a, b, callback=callback, in_place=in_place)

def divide(a, b, dtype = None, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.divide(data_a, data_b, dtype=dtype)
        if cp is not None:
            return cp.divide(data_a, data_b, dtype=dtype)
    return double_operand_operation(a, b, callback=callback, in_place=in_place)

def power(a, b, dtype = None, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.power(data_a, data_b, dtype=dtype)
        if cp is not None:
            return cp.power(data_a, data_b, dtype=dtype)
    return double_operand_operation(a, b, callback=callback, in_place=in_place)

def dot(a, b, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.dot(data_a, data_b)
        if cp is not None:
            return cp.dot(data_a, data_b)
    return double_operand_operation(a, b, callback=callback, in_place=in_place)

def matmul(a, b, dtype = None, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.matmul(data_a, data_b, dtype=dtype)
        if cp is not None:
            return cp.matmul(data_a, data_b, dtype=dtype)
    return double_operand_operation(a, b, callback=callback, in_place=in_place)

def mean(a, axis = None, dtype = None, keepdims = False, in_place = False):
    def callback(data):
        if is_cpu(data):
            return np.mean(data, axis, dtype=dtype, keepdims=keepdims)
        if cp is not None:
            return cp.mean(data, axis, dtype=dtype, keepdims=keepdims)
    return single_operand_operation(a, callback, in_place=in_place)

def square(a, dtype = None, in_place = False):
    def callback(data):
        if is_cpu(data):
            return np.square(data, dtype=dtype)
        if cp is not None:
            return cp.square(data, dtype=dtype)
    return single_operand_operation(a, callback, in_place=in_place)

###
###
###

class Tensor():
    def __init__(self, data, dtype = None, device = None) -> None:
        data = convert_to_array(data, dtype=dtype, device=device)
        self.data = data
        self.device = device
        self.dtype = data.dtype

    ###
    ###
    ###

    @property
    def shape(self):
        return self.data.shape
    @property
    def size(self):
        return self.data.size
    @property
    def ndim(self):
        return self.data.ndim
    
    ###
    ###
    ###

    def add(self, other, dtype = None, in_place = False):
        return add(self, other, dtype=dtype, in_place=in_place)
    
    def subtract(self, other, dtype = None, in_place = False):
        return subtract(self, other, dtype=dtype, in_place=in_place)
    
    def multiply(self, other, dtype = None, in_place = False):
        return multiply(self, other, dtype=dtype, in_place=in_place)
    
    def divide(self, other, dtype = None, in_place = False):
        return divide(self, other, dtype=dtype, in_place=in_place)
    
    def power(self, other, dtype = None, in_place = False):
        return power(self, other, dtype=dtype, in_place=in_place)
    
    def dot(self, other, in_place = False):
        return dot(self, other, in_place=in_place)
    
    def matmul(self, other, dtype = None, in_place = False):
        return matmul(self, other, dtype=dtype, in_place=in_place)
    
    def mean(self, axis = None, dtype = None, keepdims = False, in_place = False):
        return mean(self, axis=axis, dtype=dtype, keepdims=keepdims, in_place=in_place)
    
    def square(self, dtype = None, in_place = False):
        return square(self, dtype=dtype, in_place=in_place)
    
    ###
    ###
    ###

    def __add__(self, other):
        return add(self, other)
    def __radd__(self, other):
        return add(other, self)
    def __iadd__(self, other):
        return add(self, other, in_place=True)
    
    def __sub__(self, other):
        return subtract(self, other)
    def __rsub__(self, other):
        return subtract(other, self)
    def __isub__(self, other):
        return subtract(self, other, in_place=True)
    
    def __mul__(self, other):
        return multiply(self, other)
    def __rmul__(self, other):
        return multiply(other, self)
    def __imul__(self, other):
        return multiply(self, other, in_place=True)
    
    def __truediv__(self, other):
        return divide(self, other)
    def __rtruediv__(self, other):
        return divide(other, self)
    def __itruediv__(self, other):
        return divide(self, other, in_place=True)
    
    def __pow__(self, other):
        return power(self, other)
    def __ipow__(self, other):
        return power(self, other, in_place=True)
    
    def __matmul__(self, other):
        return matmul(self, other)
    def __rmatmul__(self, other):
        return matmul(other, self)
    def __imatmul__(self, other):
        return matmul(self, other, in_place=True)

    ###
    ###
    ###

    def __repr__(self):
        return f"Tensor({self.data})"