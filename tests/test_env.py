def test_devices(device):
    assert device in ["cpu", "cuda"]