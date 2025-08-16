from ..DepthTensor import Tensor

a = Tensor(2., requires_grad=True)
b = Tensor(3., requires_grad=True)
c = Tensor.add(a, b)
c.grad = Tensor(1., requires_grad=True).data
if c.backward is not None:
    c.backward()
print(c)
print(a.grad)