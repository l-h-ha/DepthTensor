from depthtensor import Tensor, relu

a = Tensor([-10, 2.0, 3.0], requires_grad=True)
b = relu(a)
b.backward()
print(a.grad)
