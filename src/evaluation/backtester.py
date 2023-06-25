from typing import Callable

import torch

from src.model.wrappers.abstract import ModelWrapper


class Backtester:
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        wrapper: ModelWrapper,
        evaluation_fn: Callable,
        gap_proportion: float,
        valid_proportion: float,
        n_splits: int,
        n_epochs: int = 10,
    ):
        self.x = x
        self.y = y
        self.wrapper = wrapper
        self.evaluation_fn = evaluation_fn
        self.train_proportion = float(1 - gap_proportion - valid_proportion)
        self.gap_proportion = gap_proportion
        self.valid_proportion = valid_proportion
        self.n_splits = n_splits
        self.n_epochs = n_epochs
        self.y_true = None
        self.y_pred_proba = None
        self.validate()

    @property
    def split_length(self):
        split_lengths_sum = 1 + ((self.n_splits - 1) * self.valid_proportion)
        return int(len(self.x) // split_lengths_sum)

    @property
    def step_size(self):
        return int(self.split_length * self.valid_proportion)

    @property
    def train_length(self):
        return int(self.train_proportion * self.split_length)

    @property
    def gap_length(self):
        return int(self.gap_proportion * self.split_length)

    @property
    def valid_length(self):
        return int(self.valid_proportion * self.split_length)

    def run(self):
        self.y_true = []
        self.y_pred_proba = []

        for i in range(self.n_splits):
            self.wrapper.reset_model()

            train_start = self.step_size * i
            train_end = train_start + self.train_length
            valid_start = train_end + self.gap_length
            valid_end = train_start + self.split_length

            print(f"Split {i+1}")
            print(
                f"Train start: {train_start} | "
                f"Train end: {train_end} | "
                f"Valid start: {valid_start} | "
                f"Valid end: {valid_end}"
            )

            x_train = self.x[train_start:train_end]
            y_train = self.y[train_start:train_end]
            x_valid = self.x[valid_start:valid_end]
            y_true = self.y[valid_start:valid_end]

            # Calculate predictions
            self.wrapper.fit(
                x_train, y_train, x_valid, y_true, n_epochs=self.n_epochs
            )
            y_pred_proba = self.wrapper.predict(x_valid)
            y_pred = torch.round(y_pred_proba)

            # Truncate `y_valid` to be same as `y_pred`
            valid_dl = self.wrapper.build_dataloader(x_valid, y_true)
            y_true = valid_dl.dataset.y

            # Evaluate current split
            self.evaluation_fn(y_true, y_pred)

            # Store predictions
            self.y_true.extend(y_true)
            self.y_pred_proba.extend(y_pred_proba)

        self.y_true = torch.Tensor(self.y_true)
        self.y_pred_proba = torch.Tensor(self.y_pred_proba)
        # Evaluate all splits
        print("FINAL RESULTS")
        self.evaluation_fn(self.y_true, self.y_pred_proba)

    def validate(self):
        # Validate proportions
        assert self.gap_proportion + self.valid_proportion < 1, (
            "Gap proportion and valid proportion must sum to less than 1. "
            f"Got {self.gap_proportion} & {self.valid_proportion}."
        )
        all_proportions = (
            self.gap_proportion + self.valid_proportion + self.train_proportion
        )
        assert all_proportions == 1, (
            "Gap, valid, and train proportions must sum to 1. "
            f"Got {all_proportions}."
        )
        # Validate lengths
        assert (
            self.train_length > 0
        ), f"Train length must be greater than 0. Got {self.train_length}."
        assert (
            self.valid_length > 0
        ), f"Valid length must be greater than 0. Got {self.valid_length}."
        assert self.gap_length >= 0, (
            "Gap length must be greater than or equal to 0. "
            f"Got {self.gap_length}."
        )
        # Validate other attributes
        assert (
            self.n_splits > 0
        ), f"Number of splits must be greater than 0. Got {self.n_splits}."
        assert (
            self.n_epochs > 0
        ), f"Number of epochs must be greater than 0. Got {self.n_epochs}."
