import torch

from .base import PreprocessData


class AppleQuality(PreprocessData):
    def __init__(self, data_dir: str):
        super().__init__(
            data_dir,
            "Apple Quality",
            """This dataset contains information about various 
            attributes of a set of fruits, providing insights into their characteristics."""
            ,
        )

    def load_and_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass


if __name__ == "__main__":
    apple_quality = AppleQuality("")
    apple_quality.upload_data()
