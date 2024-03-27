import torch

from .base import Base


class ADAM(Base):
    def __init__(self, model, lr, beta1=0.999, beta2=0.9, epsilon=1e-8):
        super().__init__(model, lr)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = {"beta0": 0, "beta1": torch.zeros_like(self.model.beta1)}
        self.v = {"beta0": 0, "beta1": torch.zeros_like(self.model.beta1)}
        self.t = 0

    def step(self, grad, param_name):
        self.t += 1
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (
            grad**2
        )

        m_hat = self.m[param_name] / (1 - self.beta1**self.t)
        v_hat = self.v[param_name] / (1 - self.beta2**self.t)

        update = self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        return update
