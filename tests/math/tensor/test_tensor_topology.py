import depthtensor as dt

def test_backward_adds_gradients_for_repeated_operand(device, assert_allclose):
    x = dt.Tensor([2.0], device=device, requires_grad=True)
    z = x * x

    loss = z.sum()
    loss.backward()

    assert_allclose(x.grad, [4.0])

def test_gradient_accumulation_across_calls(device, assert_allclose):
    x = dt.Tensor([2.0], device=device, requires_grad=True)

    z1 = x * 3.0
    z1.sum().backward()
    assert_allclose(x.grad, [3.0])

    z2 = x * 4.0
    z2.sum().backward()

    assert_allclose(x.grad, [7.0])

    x.zero_grad()
    assert_allclose(x.grad, [0.0])

def test_gradient_accumulation_on_reused_graph(device, assert_allclose):
    x = dt.Tensor([2.0], device=device, requires_grad=True)
    intermediate = x * 2.0
    loss = intermediate.sum()

    loss.backward()
    assert_allclose(x.grad, [2.0])

    loss.backward()
    assert_allclose(x.grad, [4.0])
    assert_allclose(intermediate.grad, [2.0])
