"""
Application interface.
"""

import abc
import pathlib
import typing

import torch
import torch.utils.data as torchdata

from torch import nn


class Application(abc.ABC):
    """
    Application interface.
    """

    def __init__(
        self,
        dataset: torchdata.Dataset,
        model: nn.Module,
        default_model_file_path: pathlib.Path,
    ):
        self.dataset = dataset
        self.model = model

        self._default_model_file_path = default_model_file_path

    def save(
        self,
        model_file_path: pathlib.Path | None = None,
    ) -> None:
        """
        Save the model.
        """
        model_file_path = self._default_model_file_path if model_file_path is None else model_file_path
        torch.save(self.model.state_dict(), model_file_path)

    def load(
        self,
        model_file_path: pathlib.Path | None = None,
    ) -> None:
        """
        Load the model.
        """
        model_file_path = self._default_model_file_path if model_file_path is None else model_file_path
        self.model.load_state_dict(torch.load(model_file_path, weights_only=True))

    @abc.abstractmethod
    def train(
        self,
        device: str,
    ) -> typing.Generator[float, None, None]:
        """
        Train the model.
        """

    @abc.abstractmethod
    def eval(
        self,
        device: str,
    ) -> dict[str, float]:
        """
        Evaluate the model.
        """
