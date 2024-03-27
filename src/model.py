import torch
import torch.nn.functional as F


class LogisticRegression:
    def __init__(self, num_features: int):
        self.num_features = num_features

        self.beta0 = torch.zeros(1)
        self.beta1 = torch.zeros(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.matmul(x, self.beta1) + self.beta0)

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(y_hat, y.float())
