from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Deque, Generic, TypeVar


T = TypeVar("T")


class FeedbackBuffer(Generic[T]):
    """Bounded history for delayed server broadcasts.

    ``delay=0`` returns the latest pushed feedback. If the requested history is
    not available yet, the oldest stored feedback is returned so early slots are
    well-defined.
    """

    def __init__(self, max_delay: int) -> None:
        self.max_delay = max(0, int(max_delay))
        self.buffer: Deque[T] = deque(maxlen=self.max_delay + 1)

    def clear(self) -> None:
        self.buffer.clear()

    def push(self, feedback: T) -> None:
        self.buffer.append(deepcopy(feedback))

    def get(self, delay: int) -> T:
        if not self.buffer:
            raise RuntimeError("FeedbackBuffer is empty")
        delay = max(0, int(delay))
        if len(self.buffer) <= delay:
            return self.buffer[0]
        return self.buffer[-1 - delay]
