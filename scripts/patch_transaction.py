"""Stage vendor patches and restore earlier files if installation fails."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from shutil import copy2, copymode
from tempfile import TemporaryDirectory


def helper_needs_update(destination: Path, data: bytes) -> bool:
    try:
        return destination.read_bytes() != data
    except FileNotFoundError:
        return True


def install_files(updates: dict[Path, str | bytes]) -> None:
    """Prepare replacement files and backups before replacing any destination."""
    with ExitStack() as staging:
        prepared: list[tuple[Path, Path, Path | None]] = []
        for destination, data in updates.items():
            if destination.is_symlink():
                destination = destination.resolve()
            directory = Path(staging.enter_context(
                TemporaryDirectory(prefix=".patch-", dir=destination.parent)
            ))
            replacement = directory / "replacement"
            if isinstance(data, str):
                replacement.write_text(data)
            else:
                replacement.write_bytes(data)
            backup = None
            if destination.exists():
                backup = directory / "original"
                copy2(destination, backup)
                copymode(destination, replacement)
            prepared.append((destination, replacement, backup))

        installed: list[tuple[Path, Path | None]] = []
        try:
            for destination, replacement, backup in prepared:
                replacement.replace(destination)
                installed.append((destination, backup))
        except OSError as error:
            for destination, backup in reversed(installed):
                try:
                    if backup is None:
                        destination.unlink()
                    else:
                        backup.replace(destination)
                except OSError as rollback_error:
                    error.add_note(f"Could not restore {destination}: {rollback_error}")
            raise
