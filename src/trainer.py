from dataloader import DataloaderModule
from model import LogisticRegression
from optimizers.base import Base
from sklearn.metrics import balanced_accuracy_score
import wandb


class Trainer:
    def __init__(
            self,
            model: LogisticRegression,
            dataloader: DataloaderModule,
            optimizer: Base,
            log_wandb: bool = False,
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.log_wandb = log_wandb

    def train(self, epochs: int) -> tuple[list[float], list[float], list[float], list[float]]:
        train_dataloader = self.dataloader.train_dataloader()
        losses_step = []
        losses_epoch = []
        accuracy_step = []
        accuracy_epoch = []
        step = 0

        for epoch in range(epochs):
            losses = []
            accuracies = []
            for x, y in train_dataloader:
                y_hat = self.model.forward(x)
                loss = self.model.loss(y_hat, y).item()
                self.optimizer.backprop(x, y, y_hat)
                accuracy = balanced_accuracy_score(y, (y_hat > 0.5).long())

                losses.append(loss)
                losses_step.append(loss)
                accuracy_step.append(accuracy)
                accuracies.append(accuracy)

                if self.log_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "step": step,
                            "train/accuracy_step": accuracy,
                            "train/loss_step": loss,
                        }
                    )

                step += 1

            loss_epoch = sum(losses) / len(losses)
            acc_epoch = sum(accuracies) / len(accuracies)
            losses_epoch.append(loss_epoch)
            accuracy_epoch.append(acc_epoch)

            if self.log_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "step": step,
                        "train/accuracy_epoch": acc_epoch,
                        "train/loss_epoch": loss_epoch,
                    }
                )

        return losses_step, losses_epoch, accuracy_step, accuracy_epoch

    def test(self) -> tuple[float, float]:
        losses = []
        accuracies = []
        test_dataloader = self.dataloader.test_dataloader()

        for x, y in test_dataloader:
            y_hat = self.model.forward(x)
            accuracies.append(balanced_accuracy_score(y, (y_hat > 0.5).long()))
            losses.append(self.model.loss(y_hat, y).item())

        loss = sum(losses) / len(losses)
        accuracy = sum(accuracies) / len(accuracies)
        if self.log_wandb:
            wandb.log({"test/accuracy": accuracy, "test/loss": loss})
        return loss, accuracy
