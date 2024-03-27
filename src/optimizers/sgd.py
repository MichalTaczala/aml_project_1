import torch

from .base import Base


class SGD(Base):
    def step(self, grad: torch.Tensor, param=None) -> torch.Tensor:
        # print("GRAD", grad)
        return self.lr * grad
