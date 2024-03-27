from abc import ABC

import torch


class Base(ABC):
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def beta_1_grad(
        self, X: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean(X.T * (y - y_hat), dim=1)

    def beta_0_grad(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return torch.mean(y - y_hat)

    def grad(
        self, X: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.beta_0_grad(y, y_hat), self.beta_1_grad(X, y, y_hat)

    def step(self, grad: torch.Tensor, param_name=None) -> torch.Tensor:
        raise NotImplementedError

    def backprop(self, X: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor):
        self.model.beta0 += self.step(self.beta_0_grad(y, y_hat), "beta0")
        self.model.beta1 += self.step(self.beta_1_grad(X, y, y_hat), "beta1")
