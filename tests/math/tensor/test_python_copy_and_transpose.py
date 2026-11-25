import copy
import pickle

import numpy as np
import pytest

import depthtensor as dt

def _pickle_roundtrip(tensor):
    return pickle.loads(pickle.dumps(tensor))

@pytest.mark.parametrize("restore", [copy.copy, copy.deepcopy, _pickle_roundtrip])
def test_python_copy_and_pickle_keep_leaf_out_protection(restore, device, assert_allclose):
    source = dt.Tensor([1.0, 2.0], device=device, name="parameter", requires_grad=True)
    source.one_grad()
    restored = restore(source)
    del source

    assert restored.name == "parameter"
    assert restored.requires_grad
    assert_allclose(restored.grad, [1.0, 1.0])
    with pytest.raises(RuntimeError, match="out buffer.*differentiable Tensor"):
        dt.add([7.0, 8.0], 1.0, out=restored.data, device=device)
    assert_allclose(restored.data, [1.0, 2.0])

    restored.zero_grad()
    restored.square().sum().backward()
    assert_allclose(restored.grad, [2.0, 4.0])

@pytest.mark.parametrize("copy_graph", [copy.copy, copy.deepcopy, pickle.dumps])
def test_python_copy_and_pickle_reject_computation_history(copy_graph):
    x = dt.Tensor([3.0], requires_grad=True)
    y = x.square()
    y.backward()
    x.zero_grad()

    with pytest.raises(TypeError, match=r"computation history.*Tensor\.copy"):
        copy_graph(y)

    np.testing.assert_array_equal(x.grad, [0.0])
    np.testing.assert_array_equal(y.grad, [1.0])

def test_restore_legacy_scalar_leaf():
    source = dt.Tensor(1.0)
    state = source.__dict__.copy()
    state["data"] = np.float64(1.0)
    del state["_version_counter"]

    restored = object.__new__(dt.Tensor)
    restored.__setstate__(state)
    restored[...] = 3.0

    assert isinstance(restored.data, np.ndarray)
    assert restored.shape == ()
    assert restored.item() == 3.0
    restored.toggle_grad(True)
    with pytest.raises(RuntimeError, match="out buffer.*differentiable Tensor"):
        dt.add(4.0, 5.0, out=restored.data)

def test_transpose_omitted_axes_reverses_dimensions_and_gradients(device, assert_allclose):
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = dt.Tensor(data, device=device, requires_grad=True)
    y = x.transpose()
    seed = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    assert_allclose(y, data.T)
    y.backward(dt.Tensor(seed, device=device))
    assert_allclose(x.grad, seed.T)
