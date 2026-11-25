import pytest
import numpy as np
import depthtensor as dt

def _get_numpy_data(t, device):
    return t.data if device == "cpu" else t.data.get()

def test_rand_bounds_and_shape(device):
    t = dt.rand(2, 3, device=device, requires_grad=True)

    assert t.shape == (2, 3)
    assert t.requires_grad is True

    data = _get_numpy_data(t, device)
    assert np.all(data >= 0.0), "rand generated values < 0"
    assert np.all(data < 1.0), "rand generated values >= 1"

def test_rand_rejects_invalid_dtype():
    with pytest.raises(
        RuntimeError, match="dtype must be float32 or float64"
    ):
        dt.rand(2, 2, dtype=np.int32, device="cpu")

def test_randn_shape_and_type(device):
    t = dt.randn(4, 4, device=device, requires_grad=True)

    assert t.shape == (4, 4)
    assert t.requires_grad is True
    assert t.dtype in (np.float32, np.float64)

def test_randn_rejects_invalid_dtype():
    with pytest.raises(
        RuntimeError, match="dtype must be float32 or float64"
    ):
        dt.randn(2, dtype=np.int32, device="cpu")

def test_randint_bounds(device):
    t = dt.randint(low=-5, high=5, size=(10, 10), device=device)

    assert t.shape == (10, 10)
    assert np.issubdtype(t.dtype, np.integer)

    data = _get_numpy_data(t, device)
    assert np.all(data >= -5), "randint generated values below low bound"
    assert np.all(data < 5), "randint generated values above or equal to high bound"

def test_uniform_bounds(device):
    t = dt.uniform(low=10.0, high=20.0, size=(5, 5), device=device, requires_grad=True)

    assert t.shape == (5, 5)
    assert t.requires_grad is True

    data = _get_numpy_data(t, device)
    assert np.all(data >= 10.0), "uniform generated values below low bound"
    assert np.all(data < 20.0), "uniform generated values above or equal to high bound"

def test_uniform_default_scalar(device):
    sample = dt.uniform(device=device, requires_grad=True)

    assert sample.shape == ()
    assert sample.device == device
    assert sample.dtype == np.float32
    assert sample.requires_grad
    sample.backward()
    assert sample.grad.item() == 1.0

@pytest.mark.parametrize("draw", [dt.rand, dt.randn])
@pytest.mark.parametrize("dtype", ["float32", np.dtype("float64")])
def test_cpu_random_accepts_dtype_like(draw, dtype):
    sample = draw(4, dtype=dtype, device="cpu")

    assert sample.shape == (4,)
    assert sample.dtype == np.dtype(dtype)

def test_cpu_random_repeats_with_same_seed():
    def draw_samples():
        return (
            dt.rand(16, dtype="float32").data,
            dt.randn(16, dtype=np.dtype("float64")).data,
            dt.uniform(size=(16,)).data,
            dt.randint(0, 16, size=(16,)).data,
        )

    state = dt.random.get_state()
    try:
        dt.random.seed(17)
        first = draw_samples()

        dt.random.seed(17)
        second = draw_samples()
    finally:
        dt.random.set_state(state)

    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)

@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_uniform_adjacent_bounds_exclude_high(device, dtype):
    low = dtype(1.0)
    high = np.nextafter(low, dtype(2.0))

    sample = dt.uniform(float(low), float(high), size=(16, 16), dtype=dtype, device=device)

    assert sample.shape == (16, 16)
    assert sample.dtype == dtype
    np.testing.assert_array_equal(_get_numpy_data(sample, device), low)

def test_uniform_rounds_low_into_interval(device):
    low = 1.0 + np.finfo(np.float32).eps.item() / 4
    only_value = np.nextafter(np.float32(1.0), np.float32(2.0))
    high = float(only_value) + np.finfo(np.float32).eps.item() / 4

    sample = dt.uniform(low, high, size=128, dtype=np.float32, device=device)

    np.testing.assert_array_equal(_get_numpy_data(sample, device), only_value)

@pytest.mark.parametrize("low, high", [(1.00000001, 1.00000002), (2.0, 1.0)])
def test_uniform_rejects_interval_without_representable_values(device, low, high):
    with pytest.raises(ValueError, match="no values representable in float32"):
        dt.uniform(low, high, dtype=np.float32, device=device)

def test_uniform_equal_bounds_return_constant(device):
    sample = dt.uniform(1.1, 1.1, size=(2, 2), dtype=np.float32, device=device)

    assert sample.shape == (2, 2)
    assert sample.dtype == np.float32
    np.testing.assert_array_equal(_get_numpy_data(sample, device), np.float32(1.1))

@pytest.mark.parametrize(
    "dtype, low, high",
    [(np.float16, -2.0, 3.0), (np.float32, -3e38, 3e38), (np.float64, -1e308, 1e308)],
)
def test_uniform_samples_wide_and_narrow_dtypes(device, dtype, low, high):
    sample = dt.uniform(low, high, size=64, dtype=dtype, device=device)
    data = _get_numpy_data(sample, device).astype(np.float64)

    assert sample.dtype == dtype
    assert np.all(data >= low)
    assert np.all(data < high)
    assert np.unique(data).size > 1

@pytest.mark.parametrize("size", [None, (64, 3)])
def test_uniform_wide_bounds_broadcast_with_other_intervals(device, size):
    xp = np if device == "cpu" else pytest.importorskip("cupy")
    limit = np.finfo(np.float64).max
    low = xp.asarray([[-limit, 3.0, 5.0]])
    high = xp.asarray([limit, 4.0, 5.0])

    sample = dt.uniform(low, high, size=size, device=device, dtype=np.float64)
    data = _get_numpy_data(sample, device)

    assert sample.shape == ((1, 3) if size is None else size)
    assert np.all(data >= [-limit, 3.0, 5.0])
    assert np.all(data[..., :2] < [limit, 4.0])
    assert np.all(data[..., 2] == 5.0)

    if size is not None:
        assert np.ptp(data[..., 0] / limit) > 0.5

@pytest.mark.parametrize("draw, args", [(dt.rand, (4,)), (dt.randn, (4,)), (dt.randint, (4,)), (dt.uniform, ())])
def test_random_rejects_complex_dtype(device, draw, args):
    with pytest.raises(RuntimeError, match="Complex dtypes are unsupported"):
        draw(*args, dtype=np.complex64, device=device)

@pytest.mark.parametrize("draw", [dt.randint, dt.uniform])
def test_random_rejects_complex_bounds(device, draw):
    with pytest.raises(RuntimeError, match="Complex dtypes are unsupported"):
        draw(1, 2 + 1j, device=device)

def test_cpu_uniform_float16_bounds_preserve_random_stream():
    state = dt.random.get_state()
    try:
        dt.random.seed(0)
        sample = dt.uniform(size=50_000, dtype=np.float16)
        following_draw = dt.random.random_sample()
    finally:
        dt.random.set_state(state)

    reference = np.random.RandomState(0)
    rounded = reference.uniform(size=50_000).astype(np.float16)
    np.minimum(rounded, np.nextafter(np.float16(1.0), np.float16(0.0)), out=rounded)

    assert np.all(sample.data < 1.0)
    np.testing.assert_array_equal(sample.data, rounded)
    assert following_draw == reference.random_sample()
