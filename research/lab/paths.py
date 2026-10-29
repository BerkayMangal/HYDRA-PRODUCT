"""Keep mutable journals outside source files and allow portable workspaces."""

import json
import os
from pathlib import Path

ROOT = (
    Path(
        os.environ.get(
            "HYDRA_RESEARCH_DIR",
            Path(__file__).resolve().parents[2] / "data" / "research",
        )
    )
    .expanduser()
    .resolve()
)


def write_json(path, value):
    """Publish complete JSON atomically so dashboard readers never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
