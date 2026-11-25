import numpy as np
import pytest

import depthtensor as dt

@pytest.fixture(params=[
    pytest.param((dt.add, (1.0,), {}, [2.0, 3.0]), id="add"),
    pytest.param((dt.square, (), {}, [1.0, 4.0]), id="square"),
    pytest.param((dt.clip, (2.5, 4.0), {}, [2.5, 2.5]), id="clip"),
    pytest.param((dt.sum, (), {"axis": (), "initial": 1}, [2.0, 3.0]), id="sum"),
    pytest.param((dt.mean, (), {"axis": ()}, [1.0, 2.0]), id="mean"),
    pytest.param((dt.max, (), {"axis": (), "initial": 3}, [3.0, 3.0]), id="max"),
    pytest.param((dt.maximum, (3.0,), {}, [3.0, 3.0]), id="maximum"),
    pytest.param((dt.less, (1.0,), {}, [0.0, 0.0]), id="less"),
])
def output_operation(request):
    return request.param

def test_tracked_input_rejects_overlapping_out_view(device, output_operation, assert_allclose):
    op, args, kwargs, expected = output_operation
    x = dt.Tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    input_view = x[:-1]

    with pytest.raises(RuntimeError, match="out buffer overlaps"):
        op(input_view, *args, out=x.data[1:], **kwargs)

    assert_allclose(x, [1.0, 2.0, 3.0])

    independent_out = input_view.data.copy()
    result = op(input_view, *args, out=independent_out, **kwargs)
    assert result.data is independent_out
    assert_allclose(result, expected)

def test_untracked_out_updates_saved_operand_versions(device, output_operation, assert_allclose):
    op, args, kwargs, expected = output_operation
    source = dt.Tensor([1.0, 2.0], device=device)
    tracked = dt.Tensor([2.0, 2.0], device=device, requires_grad=True)
    loss = (tracked * source).sum()

    result = op(source, *args, out=source.data, **kwargs)

    assert result.data is source.data
    assert_allclose(source, expected)

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        loss.backward()

def test_out_cannot_mutate_an_untracked_operand_of_a_tracked_operation(device, assert_allclose):
    tracked = dt.Tensor([2.0, 2.0], device=device, requires_grad=True)
    source = dt.Tensor([1.0, 1.0], device=device)

    with pytest.raises(RuntimeError, match="out buffer overlaps"):
        dt.maximum(tracked, source, out=source.data, requires_grad=False)

    assert_allclose(source, [1.0, 1.0])

def test_out_advances_shared_version_once_and_tracks_result_mutations(device):
    source = dt.Tensor([1.0, 2.0], device=device)
    source_view = source[:]
    version = source._version

    result = dt.maximum(source, source_view, out=source.data)

    assert source._version == source_view._version == result._version == version + 1
    result[0] = 3.0
    assert source._version == source_view._version == result._version == version + 2

def test_in_place_with_aliased_out_advances_version_once(device, assert_allclose):
    source = dt.Tensor([1.0, 2.0], device=device)
    source_view = source[:]
    version = source._version

    result = dt.add(source, 2.0, out=source.data, in_place=True)

    assert result is source
    assert source._version == source_view._version == version + 1
    assert_allclose(source_view, [3.0, 4.0])

def test_out_rejects_separate_tracked_tensor(device, assert_allclose):
    source = dt.Tensor([2.0, 3.0], device=device, requires_grad=True)
    loss = source.square().sum()

    with pytest.raises(RuntimeError, match="out buffer.*differentiable Tensor"):
        dt.add([3.0, 4.0], 4.0, device=device, out=source.data[:])

    assert_allclose(source, [2.0, 3.0])
    loss.backward()
    assert_allclose(source.grad, [4.0, 6.0])

def test_numpy_out_tuple_cannot_mutate_tracked_tensor(assert_allclose):
    source = dt.Tensor([2.0, 3.0], device="cpu", requires_grad=True)

    with pytest.raises(RuntimeError, match="out buffer.*differentiable Tensor"):
        dt.add(3.0, 4.0, out=(source.data,))

    assert_allclose(source, [2.0, 3.0])

def test_tracked_operation_accepts_disjoint_interleaved_out(device, assert_allclose):
    storage = dt.Tensor([1.0, -9.0, 2.0, -9.0], device=device).data
    source = dt.Tensor(storage[::2], requires_grad=True)
    destination = dt.Tensor(storage[1::2])

    result = dt.square(source, out=destination.data)
    result.sum().backward()

    assert result.data is destination.data
    assert source._version == 0
    assert destination._version == result._version == 1
    assert_allclose(storage, [1.0, 1.0, 2.0, 4.0])
    assert_allclose(source.grad, [2.0, 4.0])

def test_tracked_operation_rejects_partly_overlapping_mixed_itemsize_out(device, assert_allclose):
    storage = dt.Tensor([1.0, -9.0, 2.0, -9.0], dtype=np.float64, device=device).data
    source = dt.Tensor(storage[::2], requires_grad=True)
    overlapping_bytes = storage.view(np.float32)[1::4]

    with pytest.raises(RuntimeError, match="out buffer overlaps"):
        dt.square(source, out=overlapping_bytes)

    assert_allclose(storage, [1.0, -9.0, 2.0, -9.0])

def test_out_updates_version_of_separate_untracked_tensor(device, assert_allclose):
    destination = dt.Tensor([2.0, 3.0], device=device)
    tracked = dt.Tensor([1.0, 1.0], device=device, requires_grad=True)
    loss = (tracked * destination).sum()
    version = destination._version

    result = dt.add(tracked, 4.0, out=destination.data)

    assert result.data is destination.data
    assert destination._version == result._version == version + 1
    assert_allclose(destination, [5.0, 5.0])

    with pytest.raises(RuntimeError, match="modified by an in-place operation"):
        loss.backward()

    result.sum().backward()
    assert_allclose(tracked.grad, [1.0, 1.0])

def test_tracked_operation_rejects_out_overlapping_array_operand(device, assert_allclose):
    tracked = dt.Tensor([2.0, 3.0], device=device, requires_grad=True)
    constant = dt.Tensor([4.0, 5.0], device=device).data

    with pytest.raises(RuntimeError, match="out buffer overlaps"):
        dt.multiply(tracked, constant, out=constant)

    assert_allclose(constant, [4.0, 5.0])
