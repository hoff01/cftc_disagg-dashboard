"""Helpers to manage local Plotly assets for dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests

PLOTLY_VERSION = "3.3.1"
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{PLOTLY_VERSION}.min.js"


def ensure_plotly_asset(target_dir: Path, *, version: Optional[str] = None) -> Path:
    """Ensure plotly.min.js exists in target_dir."""

    target_dir.mkdir(parents=True, exist_ok=True)
    ver = version or PLOTLY_VERSION
    filename = f"plotly-{ver}.min.js"
    target_path = target_dir / filename
    if target_path.exists():
        return target_path
    url = f"https://cdn.plot.ly/plotly-{ver}.min.js"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    target_path.write_text(response.text, encoding="utf-8")
    return target_path


__all__ = ["ensure_plotly_asset", "PLOTLY_VERSION", "PLOTLY_CDN"]
