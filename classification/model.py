import torch
import lightning as L
from torch import nn
from torch.utils.data import DataLoader

from .dataset import EnhancedImageDataset


class LitModel(L.LightningModule):
    def __init__(
            self,
            model,
            num_epochs: int = 10,
            batch_size: int = 2,
            optimizer=torch.optim.Adam,
            lr: float = 1e-4,
            loss_func=nn.CrossEntropyLoss,
            val_interval: int = 1,
            logger=None,
    ) -> None:
        super().__init__()

        self.model = model

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.loss_func = loss_func
        self.val_interval = val_interval

        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, feature_vector, labels = batch
        outputs = self.model(x, feature_vector)
        labels = labels.float().unsqueeze(1)  # for BCE
        loss = self.loss_func(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, feature_vector, labels = batch
        outputs = self.model(x, feature_vector)
        labels = labels.float().unsqueeze(1)  # ensure labels have the same shape as outputs for BCE
        loss = self.loss_func(outputs, labels)  # Compute loss
        probs = torch.sigmoid(outputs)
        preds = probs >= 0.5  # Convert model outputs (logits) to binary predictions (0 or 1)
        return loss, preds, probs

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        inputs, feature_vector, labels = batch
        outputs = self.model(inputs, feature_vector)
        # labels = labels.float().unsqueeze(1)
        probs = torch.sigmoid(outputs)
        preds = probs >= 0.5  # Convert probabilities to binary predictions (0 or 1)
        return preds, probs, labels

    @staticmethod
    def _get_dataloader(imgs, labels, feature_vector, batch_size, transforms=None,
                        shuffle=False, reader='NumpyReader', pin_memory=True, num_workers=2, **kwargs):
        # Define the datasets and dataloaders using the new dataset
        dataset = EnhancedImageDataset(
            imgs, labels, feature_vector,
            reader=reader,
            transform=transforms,
            **kwargs,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def train_dataloader(self, imgs, labels, feature_vector, num_workers=2,
                         spatial_transforms=None, intensity_transforms=None) -> DataLoader:
        # Return your dataloader for training
        return self._get_dataloader(
            imgs, labels, feature_vector,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=True,
            spatial_transformation=spatial_transforms,
            intensity_transformation=intensity_transforms
        )

    def val_dataloader(self, imgs, labels, feature_vector, num_workers=2, transforms=None) -> DataLoader:
        # Return your dataloader for validation
        return self._get_dataloader(
            imgs, labels, feature_vector,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=False,
            transforms=transforms
        )

    def test_dataloader(self, imgs, labels, feature_vector) -> DataLoader:
        return self._get_dataloader(
            imgs, labels, feature_vector,
            batch_size=1,
            transforms=None
        )
