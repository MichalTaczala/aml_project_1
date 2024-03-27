import torch

class IWLS(Base):
    def __init__(self, model, iterations: int = 10):
        super().__init__(model)
        self.iterations = iterations

    def backprop(self, X: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor):
        super().backprop(X, y, y_hat)

        # Iteratively reweight and solve the least squares problem
        for _ in range(self.iterations):
            # Compute the weights for the current iteration
            weights = y_hat * (1 - y_hat)

            # Avoid division by zero
            weights = torch.clamp(weights, min=1e-5)

            # Construct a diagonal matrix of weights
            W = torch.diag(weights)

            # Augment X with ones for the intercept term
            X_augmented = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)

            # Solve the weighted least squares problem
            # We use a regularized version to ensure numerical stability
            try:
                beta = torch.linalg.solve(
                    X_augmented.T @ W @ X_augmented + 1e-5 * torch.eye(X_augmented.shape[1]), 
                    X_augmented.T @ W @ y
                )
            except RuntimeError as e:
                print("An error occurred during IWLS optimization:", e)
                break

            # Update model parameters
            self.model.beta0, self.model.beta1 = beta[0], beta[1:]

            # Update predictions for the next iteration
            y_hat = self.model.forward(X)

