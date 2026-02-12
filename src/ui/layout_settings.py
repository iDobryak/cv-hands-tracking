from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class LayoutSettings:
    window_width: int = 1460
    window_height: int = 940
    top_splitter_sizes: list[int] | None = None
    vertical_splitter_sizes: list[int] | None = None
    hands_splitter_sizes: list[int] | None = None
    invert_camera: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutSettings":
        return cls(
            window_width=int(data.get("window_width", 1460)),
            window_height=int(data.get("window_height", 940)),
            top_splitter_sizes=_parse_sizes(data.get("top_splitter_sizes")),
            vertical_splitter_sizes=_parse_sizes(data.get("vertical_splitter_sizes")),
            hands_splitter_sizes=_parse_sizes(data.get("hands_splitter_sizes")),
            invert_camera=bool(data.get("invert_camera", False)),
        )


def load_layout_settings(path: Path) -> LayoutSettings:
    if not path.exists():
        return LayoutSettings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return LayoutSettings()
        return LayoutSettings.from_dict(raw)
    except Exception:
        return LayoutSettings()


def save_layout_settings(path: Path, settings: LayoutSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_sizes(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    sizes: list[int] = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            return None
        if number <= 0:
            return None
        sizes.append(number)
    return sizes if sizes else None
