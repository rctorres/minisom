import torch
from torch.utils.data import TensorDataset, DataLoader

class BatchedFunctions:
    def __init__(self, data: torch.tensor, batch_size: int=32, device='cpu', shuffle: bool=False):
        self.data = data
        self.dataset = TensorDataset(data)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle)


    def mean(self) -> torch.tensor:
        N,M = self.data.shape
        ret = torch.zeros(M, dtype=torch.float32, device=self.device)
        for x in self.data_loader:
            x = x[0].to(self.device)
            ret += x.sum(dim=0)
        return ret / N


    def cov(self) -> torch.tensor:
        mean = self.mean()
        N = self.data.shape[0]
        M = self.data.shape[1]
        ret = torch.zeros(M,M, dtype=torch.float32, device=self.device)
        partial_ret = torch.zeros(M,M, dtype=torch.float32, device=self.device)
        for x in self.data_loader:
            x = x[0].to(self.device) - mean
            torch.mm(x.T, x, out=partial_ret)
            ret += partial_ret
        del partial_ret
        return ret / (N-1)
