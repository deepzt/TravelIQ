from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    DEFAULT_CLASS_BASE: float = 3.5
    DEFAULT_BUDGET: float = 200.0
    DEFAULT_MIN_RATING: float = 3.0
    DEFAULT_LIMIT: int = 10
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    def __post_init__(self) -> None:
        host = os.getenv("TRAVEL_API_HOST")
        if host:
            self.API_HOST = host
        port = os.getenv("TRAVEL_API_PORT")
        if port:
            try:
                self.API_PORT = int(port)
            except ValueError:
                pass


DEFAULT_CONFIG = Config()
