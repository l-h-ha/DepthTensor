from src import DepthTensor as DTensor

a = DTensor.random.randn(5, 3, requires_grad=True)
b = a.mean(keepdims=True)
DTensor.differentiate(b)
print(a)
print(b)
print(a.grad)
