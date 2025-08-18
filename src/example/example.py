from ..DepthTensor import Tensor, differentiate

a = Tensor(2., requires_grad=True)
b = Tensor(3., requires_grad=True)
c = Tensor.add(a, b)
differentiate(c)
print(a.grad)
