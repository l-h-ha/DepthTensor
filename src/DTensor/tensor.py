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
    if isinstance(obj, (np.ndarray, np.floating, np.integer, Tensor)):
        return True
    if cp is not None:
        if isinstance(obj, (cp.ndarray, cp.floating, cp.integer)):
            return True
    if raise_if_fail:
        raise RuntimeError(f"Expected objects of type: int, float, list, tuple, Tensor, numpy.ndarray, cupy.ndarray, got: {type(obj)}")
    return False

def double_operand_operation(a, b, callback, backward_callback, in_place = False):
    type_ok_for_operation(a, raise_if_fail=True)
    type_ok_for_operation(b, raise_if_fail=True)
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        is_same_device(a, b, raise_if_fail=True)
        if is_cpu(a):
            if in_place:
                a.data = callback(a.data, b.data, device="cpu")
                return a
            result = Tensor(data=callback(a.data, b.data, device="cpu"), device="cpu", requires_grad=a.requires_grad or b.requires_grad, prev=(a, b))
            if result.requires_grad:
                backward_callback(result, a, b)
            return result
        if cp is not None and is_gpu(a):
            if in_place:
                a.data = callback(a.data, b.data, device="gpu")
                return a
            result = Tensor(data=callback(a.data, b.data, device="gpu"), device="gpu", requires_grad=a.requires_grad or b.requires_grad, prev=(a, b))
            if result.requires_grad:
                backward_callback(result, a, b)
            return result

def single_operand_operation(a, callback, backward_callback, in_place = False):
    #* a must be a Tensor
    if isinstance(a, Tensor):
        if in_place:
            if is_cpu(a):
                a.data = callback(a.data)
                return a
            a.data = callback(a.data)
            return a
        if is_cpu(a):
            result = Tensor(data=callback(a.data), device="cpu", requires_grad=a.requires_grad, prev=(a,))
            if result.requires_grad:
                backward_callback(result, a)
            return result
        result = Tensor(data=callback(a.data), device="gpu", requires_grad=a.requires_grad, prev=(a,))
        if result.requires_grad:
            backward_callback(result, a)
        return result
    raise RuntimeError(f"Expected a tensor, got: {type(a)}")

###
###
###

def zeros_like(obj, dtype = None):
    if isinstance(obj, (list, tuple, np.ndarray)):
        return np.zeros_like(obj, dtype=dtype)
    if cp is not None and isinstance(obj, cp.ndarray):
        return cp.zeros_like(obj, dtype=dtype)
    if isinstance(obj, Tensor):
        if is_cpu(obj):
            return np.zeros_like(obj.data, dtype=dtype)
        if cp is not None:
            return cp.zeros_like(obj.data, dtype=dtype)

###
###
###

def sum_to_shape(result, target_shape, device):
    """
    Reverses broadcasting to the un-broadcasted shape.

    When a variable was broadcasted in order to be compatible with the other, e.g. [1.0] + [1.0, 2.0, 3.0], differentiating 
    the result w.r.t. the broadcasted variable such that the gradient matches the variable's gradient requires collapsing 
    the result's shape down to the variable's.

    Let's say:
    Scalar A, vector B (1x3)

    C = A + B (A is broadcasted into a 1x3 vector)

    In order to calculate A's gradients, per the chain rule, we have to differentiate C w.r.t. A, which gives you a vector 
    with the same shape as C's, even though the gradient's shape must match A's.

    Mathematically, since A influences every components of C, to get the gradient, we would have to sum every connections from
    A to C, which this function generalizes for every cases.
    """

    result_shape = result.shape
    if result_shape == target_shape:
        return result
    
    gained_dims = len(result_shape) - len(target_shape)
    if gained_dims > 0:
        #* We sum for gained dimensions.
        gained_axes = tuple([i for i in range(gained_dims)])
        
        if device == "cpu":
            result = np.sum(result, axis=gained_axes)
        elif device == "gpu":
            if cp is None:
                raise CuPyNotFoundError(CUPY_NOT_FOUND)
            result = cp.sum(result, axis=gained_axes)

    #* Just collapsing theg gained dimensions would not be enough, collapsing stretched dimensions is required too.
    stretched_axes = []
    print(target_shape)
    for i, d in enumerate(target_shape):
        if result.ndim == 0:
            continue
        if d == 1 and result.shape[i] > 1:
            stretched_axes.append(i)
    if len(stretched_axes) > 0:
        if device == "cpu":
            result = np.sum(result, axis=tuple(stretched_axes), keepdims=True)
        elif device == "gpu":
            if cp is None:
                raise CuPyNotFoundError(CUPY_NOT_FOUND)
            result = cp.sum(result, axis=tuple(stretched_axes), keepdims=True)
    return result

def add(a, b, dtype = None, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.add(data_a, data_b, dtype=dtype)
        if cp is not None:
            return cp.add(data_a, data_b, dtype=dtype)
    def backward_callback(result, a, b):
        def backward():
            if a.requires_grad:
                a.grad += sum_to_shape(result.grad * 1, a.grad.shape, device=result.device)
            if b.requires_grad:
                b.grad += sum_to_shape(result.grad * 1, b.grad.shape, device=result.device)
        result.backward = backward
    return double_operand_operation(a, b, callback=callback, backward_callback=backward_callback, in_place=in_place)

def subtract(a, b, dtype = None, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.subtract(data_a, data_b, dtype=dtype)
        if cp is not None:
            return cp.subtract(data_a, data_b, dtype=dtype)
    def backward_callback(result, a, b):
        def backward():
            if a.requires_grad:
                a.grad += sum_to_shape(result.grad * 1, a.grad.shape, device=result.device)
            if b.requires_grad:
                b.grad += sum_to_shape(result.grad * -1, b.grad.shape, device=result.device)
        result.backward = backward
    return double_operand_operation(a, b, callback=callback, backward_callback=backward_callback, in_place=in_place)

def multiply(a, b, dtype = None, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.multiply(data_a, data_b, dtype=dtype)
        if cp is not None:
            return cp.multiply(data_a, data_b, dtype=dtype)
    def backward_callback(result, a, b):
        def backward():
            if a.requires_grad:
                a.grad += sum_to_shape(result.grad * b.data, a.grad.shape, device=result.device)
            if b.requires_grad:
                b.grad += sum_to_shape(result.grad * a.data, b.grad.shape, device=result.device)
        result.backward = backward
    return double_operand_operation(a, b, callback=callback, backward_callback=backward_callback, in_place=in_place)

def divide(a, b, dtype = None, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.divide(data_a, data_b, dtype=dtype)
        if cp is not None:
            return cp.divide(data_a, data_b, dtype=dtype)
    def backward_callback(result, a, b):
        def backward():
            if a.requires_grad:
                a.grad += sum_to_shape(result.grad / b.data, a.grad.shape, device=result.device)
            if b.requires_grad:
                b.grad += sum_to_shape(result.grad * (a.data * -b.data**-2), b.grad.shape, device=result.device)
        result.backward = backward
    return double_operand_operation(a, b, callback=callback, backward_callback=backward_callback, in_place=in_place)

def power(a, b, dtype = None, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.power(data_a, data_b, dtype=dtype)
        if cp is not None:
            return cp.power(data_a, data_b, dtype=dtype)
    def backward_callback(result, a, b):
        def backward():
            if a.requires_grad:
                a.grad += sum_to_shape(result.grad * (b.data * a.data**(b.data - 1)), a.grad.shape, device=result.device)
            if b.requires_grad:
                b.grad += sum_to_shape(result.grad * -a.data, b.grad.shape, device=result.device)
        result.backward = backward
    return double_operand_operation(a, b, callback=callback, backward_callback=backward_callback, in_place=in_place)

def dot(a, b, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.dot(data_a, data_b)
        if cp is not None:
            return cp.dot(data_a, data_b)
    def backward_callback(result, a, b):
        def backward() -> None:
            if a.ndim == 0 or b.ndim == 0:
                if a.requires_grad:
                    a.grad += sum_to_shape(result.grad * b.data, a.grad.shape, result.device)
                if b.requires_grad:
                    b.grad += sum_to_shape(result.grad * a.data, b.grad.shape, result.device)
            #* Vec-vec
            elif a.ndim == 1 and b.ndim == 1:
                if a.requires_grad:
                    a.grad += sum_to_shape(result.grad * b.data, a.grad.shape, result.device)
                if b.requires_grad:
                    b.grad += sum_to_shape(result.grad * a.data, b.grad.shape, result.device)
            #* Mat-vec
            elif a.ndim == 2 and b.ndim == 1:
                if a.requires_grad:
                    if a.device == "gpu":
                        if cp is None:
                            raise CuPyNotFoundError(CUPY_NOT_FOUND)
                        a.grad += cp.outer(result.grad, b.data)
                    elif a.device == "cpu":
                        a.grad += np.outer(result.grad, b.data)
                if b.requires_grad:
                    b.grad += a.data.T @ result.grad
            #* Vec-mat
            elif a.ndim == 1 and b.ndim == 2:
                if a.requires_grad:
                    a.grad += result.grad @ b.data.T
                if b.requires_grad:
                    if b.device == "gpu":
                        if cp is None:
                            raise CuPyNotFoundError(CUPY_NOT_FOUND)
                        b.grad += cp.outer(a.data, result.grad)
                    elif b.device == "cpu":
                        b.grad += np.outer(a.data, result.grad)
            #* Mat-mat / ten-ten
            elif a.ndim > 1 and b.ndim > 1:
                if a.requires_grad:
                    a.grad += result.grad @ b.data.swapaxes(-2, -1)
                if b.requires_grad:
                    b.grad += a.data.swapaxes(-2, -1) @ result.grad
            else:
                raise RuntimeError("An unexpected error has occurred.")
        result.backward = backward
    return double_operand_operation(a, b, callback=callback, backward_callback=backward_callback, in_place=in_place)

def matmul(a, b, dtype = None, in_place = False):
    def callback(data_a, data_b, device):
        if device == "cpu":
            return np.matmul(data_a, data_b, dtype=dtype)
        if cp is not None:
            return cp.matmul(data_a, data_b, dtype=dtype)
    def backward_callback(result, a, b):
        def backward():
            if a.ndim == 1 and b.ndim == 1:
                if a.requires_grad:
                    a.grad += sum_to_shape(result.grad * b.data, a.grad.shape, result.device)
                if b.requires_grad:
                    b.grad += sum_to_shape(result.grad * a.data, b.grad.shape, result.device)
            #* Mat-vec
            elif a.ndim == 2 and b.ndim == 1:
                if a.requires_grad:
                    if a.device == "gpu":
                        if cp is None:
                            raise CuPyNotFoundError(CUPY_NOT_FOUND)
                        a.grad += cp.outer(result.grad, b.data)
                    elif a.device == "cpu":
                        a.grad += np.outer(result.grad, b.data)
                if b.requires_grad:
                    b.grad += a.data.T @ result.grad
            #* Vec-mat
            elif a.ndim == 1 and b.ndim == 2:
                if a.requires_grad:
                    a.grad += result.grad @ b.data.T
                if b.requires_grad:
                    if b.device == "gpu":
                        if cp is None:
                            raise CuPyNotFoundError(CUPY_NOT_FOUND)
                        b.grad += cp.outer(a.data, result.grad)
                    elif b.device == "cpu":
                        b.grad += np.outer(a.data, result.grad)
            #* Mat-mat / ten-ten
            elif a.ndim > 1 and b.ndim > 1:
                if a.requires_grad:
                    a.grad += result.grad @ b.data.swapaxes(-2, -1)
                if b.requires_grad:
                    b.grad += a.data.swapaxes(-2, -1) @ result.grad
            else:
                raise RuntimeError("An unexpected error has occurred.")
        result.backward = backward
    return double_operand_operation(a, b, callback=callback, backward_callback=backward_callback, in_place=in_place)

def mean(a, axis = None, dtype = None, keepdims = False, in_place = False):
    def callback(data):
        if is_cpu(data):
            return np.mean(data, axis, dtype=dtype, keepdims=keepdims)
        if cp is not None:
            return cp.mean(data, axis, dtype=dtype, keepdims=keepdims)
    def backward_callback(result, a):
        def backward():
            if a.requires_grad:
                size = 1
                if isinstance(axis, int):
                    size = a.shape[axis]
                elif isinstance(axis, tuple):
                    if is_cpu(result):
                        size = np.prod([a.shape[axis] for axis in axis])
                    if cp is not None and is_gpu(result):
                        size = cp.prod([a.shape[axis] for axis in axis])
                a.grad += sum_to_shape(result.grad / size, a.grad.shape, result.device)
        result.backward = backward
    return single_operand_operation(a, callback, backward_callback=backward_callback, in_place=in_place)

def square(a, dtype = None, in_place = False):
    def callback(data):
        if is_cpu(data):
            return np.square(data, dtype=dtype)
        if cp is not None:
            return cp.square(data, dtype=dtype)
    def backward_callback(result, a):
        def backward():
            if a.requires_grad:
                a.grad += sum_to_shape(result.grad * 2, a.grad.shape, result.device)
        result.backward = backward
    return single_operand_operation(a, callback, backward_callback=backward_callback, in_place=in_place)

###
###
###

class Tensor():
    def __init__(self, data, dtype = None, device = None, prev = (), requires_grad = False) -> None:
        data = convert_to_array(data, dtype=dtype, device=device)
        self.data = data
        self.device = device
        self.dtype = data.dtype

        self.requires_grad = requires_grad
        self.prev = prev
        self.backward = None
        self.grad = zeros_like(self.data, dtype=self.dtype)

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