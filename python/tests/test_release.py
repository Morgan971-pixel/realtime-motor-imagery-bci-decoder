"""Checks for the public repository: the licence is in place, and no tracked file carries a
private machine path, a personal email address, or the name of the private recordings export."""
import re
import subprocess
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# built from pieces so this file does not match its own patterns
PRIVATE = re.compile("|".join([r"/" + "Users/", r"@" + r"(gmail|icloud|outlook|yahoo)\.com", "drive" + "-download"]),
                     re.IGNORECASE)


def test_mit_licence_with_the_copyright_holder():
    text = (REPO / "LICENSE").read_text()
    assert text.startswith("MIT License\n\nCopyright (c) 2026 Morgan971-pixel\n")
    assert "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND" in text
    cargo = tomllib.loads((REPO / "rust" / "mi_core" / "Cargo.toml").read_text())
    assert cargo["package"]["license"] == "MIT"


def test_no_tracked_file_has_a_private_path_or_address():
    files = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True).stdout.split(b"\0")
    hits = []
    for name in filter(None, files):
        path = REPO / name.decode()
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits += [f"{name.decode()}: {m.group(0)}" for m in PRIVATE.finditer(text)]
    assert hits == []
