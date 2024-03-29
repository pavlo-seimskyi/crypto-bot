from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

from src.model.preprocessors.abstract import Preprocessor
from src.model.wrappers.abstract import ModelWrapper


class BinaryModelWrapper(ModelWrapper):
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        loss_fn: torch.nn.Module,
        dataset_builder: Dataset,
        dataset_builder_kwargs: Dict[str, Any] = {},
        preprocessors: List[Preprocessor] = [],
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_decay_step: int = None,
        lr_decay_multiplier: float = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
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
            Learning rate will multiply by this number
            every `lr_decay_step`-th epoch. By default 0.1.
        device : torch.device, optional
            Device to run the model on. By default,
            selects CUDA if a GPU is available and CPU otherwise.
        """
        self.device = device
        self.model = model.to(self.device)
        self.default_model = deepcopy(self.model)

        self.batch_size = batch_size
        self.loss_fn = loss_fn.to(self.device)
        self.dataset_builder = dataset_builder
        self.dataset_builder_kwargs = dataset_builder_kwargs
        self.preprocessors = preprocessors

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_step = lr_decay_step
        self.lr_decay_multiplier = lr_decay_multiplier
        self.set_optimizer()
        self.set_scheduler()

        self.train_losses = []
        self.valid_losses = []

    def reset_model(self):
        self.model = deepcopy(self.default_model)
        self.set_optimizer()
        self.set_scheduler()

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
        plot_losses: bool = True,
    ):
        self.reset_losses()
        for preprocessor in self.preprocessors:
            x_train = preprocessor.fit_transform(x_train)
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(x_train, y_train)
            self.train_losses.append(train_loss)
            summary = f"Epoch: {epoch + 1} | Train Loss: {train_loss:.3f}"
            if x_valid is not None and y_valid is not None:
                valid_loss = self.evaluate(x_valid, y_valid)
                self.valid_losses.append(valid_loss)
                summary += f" | Valid Loss: {valid_loss:.3f}"
            print(summary)
        if plot_losses:
            self.plot_losses()

    def reset_losses(self):
        self.train_losses = []
        self.valid_losses = []

    def train_epoch(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        dataloader = self.build_dataloader(x, y)
        losses = []
        for inputs, targets in dataloader:
            self.optimizer.zero_grad()
            batch_pred = self.model(inputs)
            loss = self.loss_fn(batch_pred, targets)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()
        self.scheduler.step()
        return np.mean(losses)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self.predict(x)
        # Truncate `y_true` to match `y_pred`
        y_true = self.build_dataloader(x, y).dataset.y
        loss = self.loss_fn(y_pred, y_true).item()
        return loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        for preprocessor in self.preprocessors:
            x = preprocessor.transform(x)
        x = x.to(self.device)
        dataloader = self.build_dataloader(x)
        y_pred = torch.Tensor([]).to(self.device)
        for inputs in dataloader:
            with torch.no_grad():
                batch_pred = self.model(inputs)
            y_pred = torch.cat((y_pred, batch_pred), dim=0)
        return y_pred

    def build_dataloader(self, x: torch.Tensor, y: torch.Tensor = None):
        dataset = self.dataset_builder(x, y, **self.dataset_builder_kwargs)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def plot_losses(self):
        plt.plot(self.train_losses, label="Train Loss")
        if len(self.valid_losses) > 0:
            plt.plot(self.valid_losses, label="Valid Loss")
        plt.legend()
        plt.show()

    def find_optimal_lr(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        beta: float = 0.98,
    ) -> Tuple[List[float], List[float]]:
        """
        Estimate relationship between learning rates vs. losses.
        Plot the results. The optimal learning rate is the one
        that is in the middle of the sharpest downward slope.
        The model parameters and optimizer state will be reset.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor
            Target data.
        start_lr : float, optional
            Starting learning rate, by default 1e-7.
        end_lr : float, optional
            Ending learning rate, by default 1.0.
        beta : float, optional
            Smoothing parameter, by default 0.98.

        Returns
        -------
        Tuple[List[float], List[float]]
            List of learning rates and losses.
        """
        self.reset_model()
        x, y = x.to(self.device), y.to(self.device)
        dataloader = self.build_dataloader(x, y)
        lrs, losses = self.get_lrs_vs_losses(
            dataloader, start_lr=start_lr, end_lr=end_lr, beta=beta
        )
        lr_steep, lr_min = self.get_steepest_and_min_loss_lrs(lrs, losses)
        self.plot_lrs_vs_losses(lrs, losses, lr_steep, lr_min)
        self.reset_model()
        return lrs, losses

    def get_lrs_vs_losses(
        self,
        dataloader: DataLoader,
        start_lr: float,
        end_lr: float,
        beta: float,
    ) -> Tuple[List[float], List[float]]:
        n_iterations = len(dataloader) - 1
        mult = (end_lr / start_lr) ** (1 / n_iterations)
        lr = start_lr
        self.optimizer.param_groups[0]["lr"] = lr
        avg_loss = 0
        best_loss = 0
        batch_num = 0
        losses = []
        lrs = []

        for data, target in dataloader:
            batch_num += 1
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Record the learning rate and loss
            losses.append(smoothed_loss)
            lrs.append(lr)
            # Backward pass
            loss.backward()
            self.optimizer.step()
            # Update the learning rate
            lr *= mult
            self.optimizer.param_groups[0]["lr"] = lr
            if batch_num >= n_iterations:
                break

        return lrs, losses

    def get_steepest_and_min_loss_lrs(
        self, lrs: List[float], losses: List[float]
    ) -> Tuple[float, float]:
        """
        Get the learning rate with the steepest decrease (most negative gradient)
        and the learning rate with the minimum loss.

        Returns
        -------
        Tuple[float, float]
            Learning rate with the steepest decrease and learning rate with the
        """
        lrs = np.array(lrs)
        gradient_losses = np.gradient(losses)
        steepest_lr = lrs[np.argmin(gradient_losses)]
        min_loss_lr = lrs[np.argmin(losses)]
        return steepest_lr, min_loss_lr

    def plot_lrs_vs_losses(
        self,
        lrs: List[float],
        losses: List[float],
        lr_steep: float,
        lr_min: float,
    ) -> None:
        plt.plot(lrs, losses)
        plt.axvline(
            x=lr_steep,
            color="red",
            ls="--",
            label=f"Steepest ({lr_steep:.1e})",
        )
        plt.axvline(
            x=lr_min,
            color="green",
            ls="--",
            label=f"Min. loss ({lr_min:.1e})",
        )
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.xscale("log")
        plt.title("Learning rate vs. loss")
        plt.legend()
        plt.show()
