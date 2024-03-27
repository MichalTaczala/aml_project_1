from ucimlrepo import fetch_ucirepo

import torch

from base import PreprocessData


class Fertility(PreprocessData):
    def __init__(self, data_dir: str):
        super().__init__(
            data_dir,
            "Fertility",
            """This dataset contains information about fertility""",
        )

    def load_and_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        fertility = fetch_ucirepo(id=244)
        X = fertility.data.features
        y = fertility.data.targets
        y["diagnosis"] = y["diagnosis"].map({"N": 1, "O": 0})
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.int)
        return X_tensor, y_tensor


if __name__ == "__main__":
    apple_quality = Fertility("data")
    apple_quality.load_and_transform()
    apple_quality.upload_data()
