from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class VideoSource(ABC):
    """Interface for extensible frame providers."""

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

