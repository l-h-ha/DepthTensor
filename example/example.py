from src import DepthTensor as DTensor

# 1. Setup
a = DTensor.Tensor([2.0], requires_grad=True)

# 2. Forward Pass (a is 2.0)
b = a * a  # b = 4.0. Derivative should be 2*a = 4.0

# 3. The Sabotage (In-place modification)
a += 1.0  # a is now 3.0

# 4. Backward Pass
DTensor.differentiate(b)

# 5. The Moment of Truth
print(f"Expected Gradient: 4.0 (because a was 2.0 during forward)")
print(f"Your Gradient:     {a.grad}")
