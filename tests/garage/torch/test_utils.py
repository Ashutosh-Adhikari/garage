"""This is a script to test the PyTorch utility functions."""
import numpy as np
import torch

import garage.torch.utils as tu


class TestTorchUtils:
    """Test for Torch utility functions."""

    def test_set_gpu_mode(self):
        """Test set_gpu_mode."""
        tu.set_gpu_mode(True, 0)
        assert tu.device == torch.device('cuda:0')

        tu.set_gpu_mode(False)
        assert tu.device == torch.device('cpu')

    def test_torch_utils(self):
        """Test other utility functions."""
        np_array = np.random.rand(5)
        test_torch_tensor = tu.from_numpy(np_array)
        assert all([a == b for a, b in zip(np_array, test_torch_tensor)])

        test_np_array = tu.to_numpy(test_torch_tensor)
        assert np_array.all() == test_np_array.all()

        zeros = torch.zeros(10)
        test_torch_tensor = tu.zeros(10)
        assert torch.all(torch.eq(zeros, test_torch_tensor))

        tu.set_gpu_mode(True, 0)
        ones = torch.ones(10).to('cuda:0')
        test_torch_tensor = tu.ones(10)
        assert torch.all(torch.eq(ones, test_torch_tensor))
