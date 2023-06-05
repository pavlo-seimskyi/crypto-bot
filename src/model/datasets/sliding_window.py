from typing import Optional

import torch


class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        seq_len: int = 1,
        y_position: Optional[int] = None,
    ):
        """
        Creates a sequence of sliding windows from the original X array, and maps each
            window to a target on a specified position within the window.

        Parameters
        ----------
        x : torch.Tensor
            Tensor with features of shape (n_samples, n_features)
        y : torch.Tensor, optional
            Tensor with targets of shape (n_samples, 1)
        seq_len : int
            Window length
        y_position : int, optional
            Position within the window that corresponds to the target.
            By default, the target is mapped to the last window element.
            Can be anything between `0` and `seq_len - 1`.
        """
        super(SlidingWindowDataset, self).__init__()
        self.seq_len = seq_len
        self.x = self.apply_sliding_window_to_x(x)
        self.y_available = y is not None
        if self.y_available:
            self.y_position = (
                y_position if y_position is not None else seq_len - 1
            )
            self.y = self.get_y_per_window(y)
            self.validate_y()

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index: int):
        if self.y_available:
            return self.x[index], self.y[index]
        else:
            return self.x[index]

    def apply_sliding_window_to_x(self, x: torch.Tensor) -> torch.Tensor:
        """Transforming an array into a sequence of sliding windows of size:
        (batch, seq_len, n_features)."""
        return x.unfold(dimension=0, size=self.seq_len, step=1).transpose(2, 1)

    def get_y_per_window(self, y: torch.Tensor) -> torch.Tensor:
        start = self.y_position
        end = -self.seq_len + start + 1 if start + 1 != self.seq_len else None
        return y[start:end]

    def validate_y(self):
        assert self.x.size(0) == self.y.size(
            0
        ), "X and y must have equal shape."
        assert (
            self.y_position < self.seq_len
        ), f"Y position must be between 0 and sequence length - 1."
