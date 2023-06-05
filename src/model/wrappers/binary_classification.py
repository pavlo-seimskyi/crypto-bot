import numpy as np
import torch

from src.model.wrappers.base import ModelWrapper


class BinaryModelWrapper(ModelWrapper):
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        loss_fn: torch.nn.Module,
        dataset_builder: torch.utils.data.Dataset,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_decay_step: int = None,
        lr_decay_multiplier: float = None,
    ):
        """
        Model wrapper for binary classification.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model.
        batch_size : int, optional
            Batch size, by default 512.
        loss_fn : torch.nn.Module, optional
            Loss function, by default torch.nn.BCELoss()
        dataset_builder : torch.utils.data.Dataset
            Dataset builder, one of `src.model.datasets`.
        lr : float, optional
            Learning rate, by default 1e-4.
        weight_decay : float, optional
            Weight decay, by default 1e-4.
        lr_decay_step : int, optional
            Learning rate decay step, by default 10.
        lr_decay_multiplier : float, optional
            Learning rate will multiply by this number every `lr_decay_step`-th epoch.
        """
        self.default_model = model
        self.model = model
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.dataset_builder = dataset_builder
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_step = lr_decay_step
        self.lr_decay_multiplier = lr_decay_multiplier

        self.set_optimizer()
        self.set_scheduler()

    def reset_model(self):
        self.model = self.default_model
        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler()

    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def set_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.lr_decay_step,
            gamma=self.lr_decay_multiplier,
        )

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor = None,
        y_valid: torch.Tensor = None,
        n_epochs: int = 10,
    ):
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(x_train, y_train)
            summary = f"Epoch: {epoch + 1} | Train Loss: {train_loss:.3f}"
            if x_valid is not None and y_valid is not None:
                valid_loss = self.evaluate(x_valid, y_valid)
                summary += f" | Valid Loss: {valid_loss:.3f}"
            print(summary)

    def train_epoch(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        dataloader = self.build_dataloader(x, y)
        losses = []
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            batch_pred = self.model(inputs)
            loss = self.loss_fn(batch_pred, targets)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()
        self.scheduler.step()
        return np.mean(losses)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        y_pred = self.predict(x)
        # Truncate `y_true` to match `y_pred`
        dl_valid = self.build_dataloader(x, y)
        y_true = dl_valid.dataset.y
        return self.loss_fn(y_pred, y_true).item()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        dataloader = self.build_dataloader(x)
        y_pred = torch.Tensor([]).to(self.device)
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                batch_pred = self.model(inputs)
            y_pred = torch.cat((y_pred, batch_pred), dim=0)
        return y_pred

    def build_dataloader(self, x: torch.Tensor, y: torch.Tensor = None):
        dataset = self.dataset_builder(x, y)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )
