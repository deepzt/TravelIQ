from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    """Return the project root (two levels up from this file: src/ -> project/)."""
    return Path(__file__).parent.parent


def parse_locality(address_str: Any) -> str | None:
    """Extract the 'locality' field from a stringified address dict."""
    if address_str is None:
        return None
    try:
        import pandas as pd
        if pd.isna(address_str):
            return None
    except Exception:
        pass
    try:
        d = ast.literal_eval(str(address_str))
        if isinstance(d, dict):
            return d.get("locality")
    except Exception:
        return None
    return None


def safe_numeric(val: Any, default: float | None = None) -> float | None:
    """Coerce val to float, returning default on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default
