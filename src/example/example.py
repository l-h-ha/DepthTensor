from ..DTensor import CPUtensor, differentiate

a = CPUtensor(1.0, requires_grad=True)
b = CPUtensor(2.0, requires_grad=True)
c = a + b
differentiate(c)
print(a.grad)