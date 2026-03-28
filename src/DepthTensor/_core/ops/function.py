from abc import abstractmethod
from typing import Any
from ...typing import TensorType


class Function:
    @abstractmethod
    def link(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self, *args: Any, requires_grad: bool = False, **kwargs: Any
    ) -> TensorType:
        raise NotImplementedError
