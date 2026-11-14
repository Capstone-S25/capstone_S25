from queue import Queue
from typing import TypeVar

DEFAULT_FRAME_QUEUE_SIZE = 8

T = TypeVar("T")


class DropOldestQueue(Queue[T]):
    """Keep the newest waiting items, dropping the oldest when the queue is full

    An item already being processed is not removed or interrupted.
    """

    def __init__(self, maxsize: int = DEFAULT_FRAME_QUEUE_SIZE) -> None:
        """
        Args:
            maxsize : Maximum number of waiting items; defaults to 8
        """
        if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize <= 0:
            raise ValueError("Queue size must be a positive integer")
        super().__init__(maxsize=maxsize)
        self._dropped_count = 0

    @property
    def dropped_count(self) -> int:
        """Get the number of items dropped because the queue was full"""
        with self.mutex:
            return self._dropped_count

    def put(self, item: T, block: bool = True, timeout: float | None = None) -> None:
        """Add an item, dropping the oldest waiting item if the queue is full

        Args:
            item    : New item to add to the queue
            block   : Accepted for Queue compatibility; ignored because adding
                      an item never waits for space
            timeout : Accepted for Queue compatibility; ignored because adding
                      an item never waits for space
        """
        with self.not_full:
            if self._qsize() >= self.maxsize:
                self._get()
                self.unfinished_tasks -= 1
                self._dropped_count += 1
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()
