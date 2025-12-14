from src import depthtensor as dt

# Initialize tensors
x = dt.Tensor([1.2], device="gpu", requires_grad=True)
y = dt.Tensor([1.2], device="gpu", requires_grad=True)

a, b = dt.Tensor([1], device="gpu"), dt.Tensor([100], device="gpu")

# 2. Optimization Loop
lr = 0.001
for i in range(500):
    # Rosenbrock: f(x,y) = (a-x)^2 + b(y-x^2)^2
    loss = (a - x) ** 2 + b * (y - x**2) ** 2

    # Backpward pass
    dt.differentiate(loss)

    # Gradient Descent
    x.data -= lr * x.grad  # type: ignore
    y.data -= lr * y.grad  # type: ignore

    # Zero grads
    x.zeros_grad()
    y.zeros_grad()

    if i % 10 == 0:
        print(loss.item())

print(f"Converged: ({x.data}, {y.data})")
# Target: (1.0, 1.0)
