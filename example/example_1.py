"""
Minimize the Rosenbrock function.
"""

import depthtensor as dt

x = dt.Tensor([1.2], device="cpu", requires_grad=True)
y = dt.Tensor([1.2], device="cpu", requires_grad=True)

a, b = dt.Tensor([1], device="cpu"), dt.Tensor([100], device="cpu")

lr = 0.001

for i in range(500):
    loss = (a - x) ** 2 + b * (y - x**2) ** 2
    loss.backward()

    x.data -= lr * x.grad
    y.data -= lr * y.grad

    x.zero_grad()
    y.zero_grad()

    if i % 10 == 0:
        print(loss.item())

print(f"({x.data}, {y.data})")
