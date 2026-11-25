from abc import abstractmethod
from contextlib import AbstractContextManager, ExitStack
from typing import Any, Callable
from ...typing import TensorType

class _ChainedBackward:
    previous_backward: Callable[[], None] | None

    def _use_saved_output(self) -> AbstractContextManager[None]:
        raise NotImplementedError

    def __call__(self) -> None:
        backward: Callable[[], None] | None = self

        with ExitStack() as stack:
            while isinstance(backward, _ChainedBackward):
                stack.enter_context(backward._use_saved_output())
                backward = backward.previous_backward

            if backward is not None:
                backward()

class Function:
    @abstractmethod
    def link(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self, *args: Any, requires_grad: bool = False, **kwargs: Any
    ) -> TensorType:
        raise NotImplementedError
