"""Importable add-on install, remove and run steps.

These functions are used by the CLI (`scripts/addon.py`), the studio API and the MCP
server. They raise on error so each caller can present the failure in its own way.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from image_to_3dlab import __version__, processes
from image_to_3dlab.addons import (
    Listing,
    Manifest,
    build_command,
    install_plan,
    output_dir,
    parse_inputs,
    read_manifest,
    read_registry,
    run_record,
    tier_from_install,
    validate_addon_id,
)
from image_to_3dlab.blender import find_blender
from image_to_3dlab.provenance import (
    read_model_record,
    sha256_file,
    write_derived_sidecar,
    write_provenance,
)

# The list of add-ons: built from the entry files in the public list repo and committed there on every merge.
REGISTRY_URL = "https://raw.githubusercontent.com/AssetFurnace/assetfurnace-addons/main/index.json"
USER_AGENT = f"image-to-3dlab/{__version__} (+https://github.com/Bingeljell/image-to-3dlab)"
INSTALL_JSON = ".addon-install.json"


def _addon_dir(addon_id: str, addons_dir: Path) -> Path:
    return addons_dir / addon_id


def _git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), check=check,
                          capture_output=True, text=True)


def _clone(repo_url: str, commit: str, dest: Path, cwd: Path) -> None:
    """Clone a repo and check out the pinned commit, verifying HEAD matches."""
    if dest.exists():
        _remove_tree(dest)
    _git("clone", repo_url, str(dest), cwd=cwd)
    _git("checkout", commit, cwd=dest)
    result = _git("rev-parse", "HEAD", cwd=dest)
    head = result.stdout.strip()
    if head != commit:
        _remove_tree(dest, quiet=True)
        raise RuntimeError(
            f"HEAD mismatch: checked out {head}, expected pinned commit {commit}"
        )


def _install_json_path(dest: Path) -> Path:
    return dest / INSTALL_JSON


def _write_install_json(dest: Path, listing: Listing, tier: str) -> None:
    record = {
        "id": listing.id,
        "repo": listing.repo,
        "commit": listing.commit,
        "tier": tier,
        "installed_at": datetime.now(UTC).isoformat(),
    }
    _install_json_path(dest).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _read_install_json(dest: Path) -> dict[str, Any]:
    path = _install_json_path(dest)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc


def _fetch_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def load_registry(index: Path | str | None) -> list[Listing]:
    """Read the add-on list from a file or the network."""
    if index is not None:
        data = json.loads(Path(index).read_text(encoding="utf-8"))
    else:
        data = _fetch_json(REGISTRY_URL)
    return read_registry(data)


def resolve_install_target(
    addon_id: str | None,
    unlisted: str | None,
    index: Path | str | None,
) -> tuple[Listing, str]:
    """Return the listing to install and its tier (Unknown when it is not on the list).

    Raises ValueError for bad ids, bad URLs, or unlisted names.
    """
    if unlisted is not None:
        match = re.fullmatch(r"(.+)@([0-9a-f]{40})", unlisted)
        if not match:
            raise ValueError("--unlisted must be <git url>@<commit>")
        repo_url, commit = match.group(1), match.group(2)
        derived_id = Path(repo_url).stem.replace(".git", "")
        validate_addon_id(derived_id)
        listing = Listing(
            id=derived_id,
            name=derived_id,
            repo=repo_url,
            commit=commit,
            author={"name": "unknown", "url": ""},
            summary="",
            type="tool",
        )
        return listing, "unknown"

    if not addon_id:
        raise ValueError("add requires an id or --unlisted")
    validate_addon_id(addon_id)
    registry = load_registry(index)
    matches = [entry for entry in registry if entry.id == addon_id]
    if not matches:
        raise ValueError(
            f"{addon_id} is not on the add-on list. "
            "Use --unlisted <git url>@<commit> if you still want it (it will be marked Unknown)."
        )
    return matches[0], matches[0].tier


def _tool_by_id(manifest: Manifest, tool_id: str) -> Any:
    for tool in manifest.tools:
        if tool.id == tool_id:
            return tool
    raise ValueError(f"tool {tool_id!r} not found in add-on {manifest.id}")


def _default_run_stem() -> str:
    """A per-second timestamp so repeated runs do not collide by default."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def prepare_install(
    listing: Listing,
    addons_dir: Path,
    cwd: Path,
    *,
    id_from_manifest: bool = False,
) -> tuple[Manifest, Path]:
    """Clone the listing's repo to a temp folder and verify its manifest.

    A listed add-on's id must match its manifest. An unlisted one has no list entry to
    match (its id was only guessed from the repo name), so `id_from_manifest` skips the check.

    Returns the manifest and the temp path. The caller must move the temp path into
    place (or let the TemporaryDirectory clean it up).
    """
    temp_parent = addons_dir
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_dest = tempfile.mkdtemp(prefix=f".{listing.id}-install-", dir=temp_parent)
    temp_path = Path(temp_dest)
    try:
        _clone(listing.repo, listing.commit, temp_path, cwd=cwd)
        manifest = read_manifest(temp_path / "addon.json")
        if manifest.id != listing.id and not id_from_manifest:
            raise ValueError(
                f"manifest id {manifest.id!r} does not match listing id {listing.id!r}"
            )
        return manifest, temp_path
    except Exception:
        _remove_tree(temp_path, quiet=True)
        raise


def apply_install(
    listing: Listing,
    manifest: Manifest,
    temp_dest: Path,
    addons_dir: Path,
    tier: str,
) -> Path:
    """Move the prepared temp folder into addons_dir and write the install record."""
    dest = _addon_dir(listing.id, addons_dir)
    if dest.exists():
        _remove_tree(dest)
    shutil.move(str(temp_dest), str(dest))
    _write_install_json(dest, listing, tier=tier)
    return dest


def run_addon_setup(addon_dir: Path) -> None:
    """Run the add-on's setup command in its own folder."""
    manifest = read_manifest(addon_dir / "addon.json")
    if manifest.setup:
        result = subprocess.run(manifest.setup, cwd=str(addon_dir), check=False)
        if result.returncode != 0:
            raise RuntimeError("setup failed")


def install_addon(
    addon_id: str | None = None,
    unlisted: str | None = None,
    index: Path | str | None = None,
    confirm: bool = False,
    addons_dir: Path | None = None,
    repo_dir: Path | None = None,
) -> dict[str, Any]:
    """Plan or perform an add-on install.

    With `confirm=False` (the default) this clones the repo into a temporary folder,
    verifies the manifest, and returns the install plan. Nothing is written to
    `addons_dir`.

    With `confirm=True` the temporary folder is moved into `addons_dir`, the install
    record is written, and setup is run.

    Raises ValueError or RuntimeError on failure.
    """
    addons_dir = Path(addons_dir) if addons_dir is not None else Path("addons")
    repo_dir = Path(repo_dir) if repo_dir is not None else Path.cwd()
    listing, tier = resolve_install_target(addon_id, unlisted, index)
    manifest, temp_dest = prepare_install(
        listing, addons_dir, cwd=repo_dir, id_from_manifest=tier == "unknown"
    )
    if tier == "unknown":
        validate_addon_id(manifest.id)
        listing = replace(listing, id=manifest.id, name=manifest.name)
    plan = install_plan(listing, manifest, tier=tier)
    if not confirm:
        _remove_tree(temp_dest, quiet=True)
        return {"plan": plan, "listing": listing, "manifest": manifest}

    try:
        dest = apply_install(listing, manifest, temp_dest, addons_dir, tier=tier)
        run_addon_setup(dest)
        return {"installed": True, "id": listing.id, "dir": dest}
    except Exception:
        _remove_tree(temp_dest, quiet=True)
        raise


def _remove_tree(path: Path, *, quiet: bool = False) -> None:
    """Remove a cloned add-on folder, read-only git objects included (Windows keeps them read-only, so a plain
    `shutil.rmtree` stopped a reinstall and `ignore_errors` left every temporary clone behind in `addons/`).
    `quiet` swallows what still fails, for clean-up on a path that is already failing."""
    try:
        shutil.rmtree(path, onerror=_remove_readonly)
    except OSError:
        if not quiet:
            raise


def _remove_readonly(func, path, _exc_info):
    """Windows leaves cloned git objects read-only; make them removable."""
    import stat

    try:
        os.chmod(path, stat.S_IWRITE)
    except OSError:
        pass
    func(path)


def remove_addon(addon_id: str, addons_dir: Path | None = None) -> None:
    """Remove an installed add-on."""
    validate_addon_id(addon_id)
    addons_dir = Path(addons_dir) if addons_dir is not None else Path("addons")
    dest = _addon_dir(addon_id, addons_dir)
    if not dest.exists():
        raise ValueError(f"{addon_id} is not installed")
    _remove_tree(dest)


# How long a cancelled tool gets to stop by itself before it is stopped for certain.
CANCEL_GRACE_SECONDS = 10.0


def _run_subprocess(
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    logger: Callable[[str], None] | None,
    cancel_event: threading.Event | None = None,
    on_start: Callable[[subprocess.Popen], None] | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, optionally streaming each line to `logger`.

    From a terminal (no logger, nothing to cancel it) the tool runs in the foreground, where
    Ctrl-C reaches it. Otherwise it leads its own process group, `on_start` is told the
    process, and a cancel stops the group: the tool and everything it started.
    """
    if logger is None and cancel_event is None and on_start is None:
        return subprocess.run(argv, cwd=str(cwd), env=env, check=False)
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        text=True,
        **processes.stoppable_popen_kwargs(),
    )
    assert proc.stdout is not None
    done = threading.Event()

    def watch_for_cancel() -> None:
        # Stop a silent process too: the reading loop below only wakes on output.
        while not done.is_set():
            if cancel_event.wait(timeout=0.2):
                # The group, not the tool alone: what it started (Blender's helpers) would run
                # on, and keep the output open so the loop below never ends.
                processes.terminate_group(proc.pid)
                # Asked first, so a tool can tidy up. One that carries on would stay
                # "cancelling" for good, holding the place every other big job waits for.
                if not done.wait(timeout=CANCEL_GRACE_SECONDS):
                    processes.kill_group(proc.pid)
                return

    try:
        if on_start is not None:
            on_start(proc)
        if cancel_event is not None:
            threading.Thread(target=watch_for_cancel, daemon=True).start()
        for line in proc.stdout:
            if logger is not None:
                logger(line.rstrip("\n"))
        code = proc.wait()
    except BaseException:
        # A line that could not be logged or read: nobody reads the output now, so the tool
        # would block on it or run on unwatched. It goes with the run.
        processes.kill_group(proc.pid)
        # A group that cannot be stopped must not hold the error back: wait the grace, then go on.
        try:
            proc.wait(timeout=CANCEL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        proc.stdout.close()
        raise
    finally:
        done.set()      # after the wait: a tool that closes its output and carries on is still watched
    return subprocess.CompletedProcess(argv, returncode=code)


def result_is_fresh(path: Path, started: float) -> bool:
    """True if `path` exists and was written at or after `started`."""
    resolved = path.resolve()
    return resolved.exists() and resolved.stat().st_mtime >= started


def run_addon(
    addon_id: str,
    tool_id: str,
    inputs: dict[str, str],
    *,
    run_stem: str | None = None,
    output_root: Path | None = None,
    addons_dir: Path | None = None,
    repo_dir: Path | None = None,
    blender: str | Path | None = None,
    logger: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    on_start: Callable[[subprocess.Popen], None] | None = None,
) -> Path:
    """Run one tool from an installed add-on and return the result file path.

    `on_start` is given the tool's process once it runs, so the caller can stop it.

    Raises ValueError or RuntimeError on failure.
    """
    validate_addon_id(addon_id)
    addons_dir = Path(addons_dir) if addons_dir is not None else Path("addons")
    repo_dir = Path(repo_dir) if repo_dir is not None else Path.cwd()
    output_root = Path(output_root) if output_root is not None else Path("output")

    addon_dir = _addon_dir(addon_id, addons_dir)
    if not addon_dir.exists():
        raise ValueError(f"{addon_id} is not installed")
    manifest = read_manifest(addon_dir / "addon.json")
    tool = _tool_by_id(manifest, tool_id)

    install_info = _read_install_json(addon_dir)
    listing = Listing(
        id=install_info["id"],
        name=manifest.name,
        repo=install_info["repo"],
        commit=install_info["commit"],
        author=manifest.author,
        summary=manifest.summary,
        type=manifest.type,
    )
    tier = tier_from_install(install_info)

    parsed_inputs = parse_inputs(inputs, tool, cwd=repo_dir)
    stem = run_stem or _default_run_stem()
    out_dir = output_root / output_dir(addon_id, stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()
    blender_path = blender or find_blender() or "blender"
    argv = build_command(tool, parsed_inputs, out_dir=out_dir, blender=blender_path)

    started = datetime.now(UTC).timestamp()
    # PWD too: Blender on Linux finds a relative script (`-P crate.py`) from $PWD, which still names the folder we started in
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PWD": str(addon_dir.resolve())}
    result = _run_subprocess(argv, addon_dir, env, logger, cancel_event=cancel_event,
                             on_start=on_start)
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("cancelled")
    if result.returncode != 0:
        raise RuntimeError("run failed")

    result_file = (out_dir / f"{stem}.{tool.makes}").resolve()
    if not result_is_fresh(result_file, started):
        raise RuntimeError(f"expected result {result_file.name} not written")

    record = run_record(listing, manifest, tool, parsed_inputs, tier=tier)
    write_add_on_record(result_file, record, tool, parsed_inputs)
    return result_file


def write_add_on_record(result_file: Path, record: dict[str, Any], tool: Any,
                        parsed_inputs: dict[str, Any]) -> None:
    """Write the licence record for an add-on's result.

    A model that went in keeps its licence: the result inherits it (and any restriction, such as
    a model not licensed in some countries), through the same ancestry the core uses elsewhere.
    A model with no record counts as unverified, never as clear.
    """
    sources = [Path(parsed_inputs[spec.id]) for spec in tool.inputs
               if spec.type == "model" and spec.id in parsed_inputs]
    if sources:
        derived_path = write_derived_sidecar(
            sources[0], result_file, operation=f"add-on:{record['addon']['id']}:{tool.id}",
            component_records=[r for src in sources[1:]
                               if (r := read_model_record(src)) is not None],
        )
        derived = json.loads(derived_path.read_text(encoding="utf-8"))
        derived_output = derived.get("output", {})
        record = {**derived, **{k: v for k, v in record.items() if k != "schema_version"}}
        record["output"] = {**derived_output, "source": "add-on"}
    else:
        record["output"] = {"path": str(result_file), "sha256": sha256_file(result_file),
                            "source": "add-on"}
    write_provenance(result_file.with_suffix(".provenance.json"), record, atomic=True)


def list_installed(addons_dir: Path | None = None) -> list[str]:
    """Ids of every add-on currently installed."""
    addons_dir = Path(addons_dir) if addons_dir is not None else Path("addons")
    if not addons_dir.is_dir():
        return []
    return [p.name for p in addons_dir.iterdir() if p.is_dir()]
