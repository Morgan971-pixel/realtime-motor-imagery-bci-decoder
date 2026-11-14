"""Where recordings and trained models may be written: never inside this repository."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def outside_repo(path) -> Path:
    """The resolved path, or ValueError if it is inside the repository. Symlinks are resolved,
    and the nearest existing folder is compared with the repository by file identity, so a
    differently capitalised path on a case-insensitive disk (macOS) is caught too."""
    resolved = Path(path).expanduser().resolve()
    existing = resolved
    while not existing.exists():
        existing = existing.parent
    if any(os.path.samefile(folder, REPO) for folder in (existing, *existing.parents)):
        raise ValueError(f"{path}: recordings and trained models stay outside the repository ({REPO})")
    return resolved
