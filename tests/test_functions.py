import pytest
import torch
import numpy as np
from minisom_gpu.functions import BatchedFunctions



class TestFunctions:
    @pytest.fixture
    def data(self):
        return torch.tensor(np.array([
            [1,7],
            [3,9],
            [8,5]
        ]), dtype=torch.float32)
    
    
    @pytest.mark.parametrize('device', [('cpu'), ('cuda')])
    def test_mean(self, data, device):
        obj = BatchedFunctions(data, batch_size=2, device=device)
        ret = obj.mean()
        assert ret[0] == 4.
        assert ret[1] == 7.

