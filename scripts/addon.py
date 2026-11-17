#!/usr/bin/env python3
"""Install and run add-ons: someone else's repo with an `addon.json`.

    ./lab addon list
    ./lab addon add vehicle-kit
    ./lab addon add --unreviewed https://github.com/example/pedestal.git@<commit>
    ./lab addon remove vehicle-kit
    ./lab addon run vehicle-kit build --input preset=bus_12m

Add-ons are listed in the registry at `https://assetfurnace.com/add-ons/index.json`.
Only listed names can be installed; anything else needs `--unreviewed` and a warning.
Each add-on is cloned at a pinned commit, checked out, verified, and run in its own
folder with `HF_HUB_OFFLINE=1`. The core writes the `.provenance.json` sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import __version__
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
    validate_addon_id,
)
from image_to_3dlab.blender import find_blender
from image_to_3dlab.provenance import write_provenance

REGISTRY_URL = "https://assetfurnace.com/add-ons/index.json"
ADDONS_DIR = Path(os.environ.get("ADDONS_DIR", REPO / "addons"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", REPO / "output"))
USER_AGENT = f"image-to-3dlab/{__version__} (+https://github.com/Bingeljell/image-to-3dlab)"
INSTALL_JSON = ".addon-install.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List reviewed add-ons.")
    list_cmd.add_argument("--index", help="local registry file for tests or review")
    list_cmd.add_argument("--installed", action="store_true",
                          help="Only show add-ons already installed.")

    add_cmd = sub.add_parser("add", help="Install an add-on after showing its plan.")
    add_cmd.add_argument("--index", help="local registry file for tests or review")
    add_cmd.add_argument("id", nargs="?", help="id of a listed add-on")
    add_cmd.add_argument("--unreviewed", help="git url@commit, not on the registry")
    add_cmd.add_argument("--yes", action="store_true",
                         help="do not ask for confirmation (scripts only)")

    remove_cmd = sub.add_parser("remove", help="Remove an installed add-on.")
    remove_cmd.add_argument("id", help="add-on id")

    run_cmd = sub.add_parser("run", help="Run a tool from an installed add-on.")
    run_cmd.add_argument("id", help="add-on id")
    run_cmd.add_argument("tool", help="tool id from the add-on manifest")
    run_cmd.add_argument("--input", action="append", default=[],
                         help="key=value input for the tool (repeatable)")
    run_cmd.add_argument("--output-dir", type=Path, help="override the output folder")
    run_cmd.add_argument("--run-stem", default=None,
                         help="name for this run and its result file (default: timestamp)")
    run_cmd.add_argument("--blender", help="override the Blender path")
    run_cmd.add_argument("--yes", action="store_true",
                         help="do not ask for confirmation")
    return parser


def _fetch_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_registry(index: Path | str | None) -> list[Listing]:
    if index is not None:
        data = json.loads(Path(index).read_text(encoding="utf-8"))
    else:
        data = _fetch_json(REGISTRY_URL)
    return read_registry(data)


def _addon_dir(addon_id: str) -> Path:
    return ADDONS_DIR / addon_id


def _git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), check=check,
                          capture_output=True, text=True)


def _clone(repo_url: str, commit: str, dest: Path) -> None:
    """Clone a repo and check out the pinned commit, verifying HEAD matches."""
    if dest.exists():
        shutil.rmtree(dest)
    _git("clone", repo_url, str(dest), cwd=REPO)
    _git("checkout", commit, cwd=dest)
    result = _git("rev-parse", "HEAD", cwd=dest)
    head = result.stdout.strip()
    if head != commit:
        shutil.rmtree(dest, ignore_errors=True)
        raise SystemExit(
            f"HEAD mismatch: checked out {head}, expected pinned commit {commit}"
        )


def _install_json_path(dest: Path) -> Path:
    return dest / INSTALL_JSON


def _write_install_json(dest: Path, listing: Listing, reviewed: bool) -> None:
    record = {
        "id": listing.id,
        "repo": listing.repo,
        "commit": listing.commit,
        "reviewed": reviewed,
        "installed_at": datetime.now(UTC).isoformat(),
    }
    _install_json_path(dest).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _read_install_json(dest: Path) -> dict[str, Any]:
    path = _install_json_path(dest)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc


def _print_plan(plan: dict) -> None:
    print(f"Author: {plan['author']}")
    print(f"Licence: {plan['licence']}")
    if plan["models"]:
        print("Models it loads:")
        for model in plan["models"]:
            print(f"  - {model['name']} ({model['licence']}, {model['size']})")
    print(f"Setup: {' '.join(plan['setup'])}")
    print(plan["warning"])


def _ask_yes(prompt: str = "Continue?") -> bool:
    if not sys.stdin.isatty():
        return False
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def _default_run_stem() -> str:
    """A per-second timestamp so repeated runs do not collide by default."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _collect_inputs(items: list[str]) -> dict[str, str]:
    """Split --input key=value strings without coercing their types."""
    inputs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--input must be key=value: {item!r}")
        key, value = item.split("=", 1)
        inputs[key] = value
    return inputs


def _tool_by_id(manifest: Manifest, tool_id: str) -> Any:
    for tool in manifest.tools:
        if tool.id == tool_id:
            return tool
    raise ValueError(f"tool {tool_id!r} not found in add-on {manifest.id}")


def _all_files(directory: Path) -> set[Path]:
    """Every file under directory, resolved."""
    return {p.resolve() for p in directory.rglob("*") if p.is_file()}


def cmd_list(args: argparse.Namespace) -> int:
    registry = _load_registry(args.index)
    installed = {p.name for p in ADDONS_DIR.iterdir() if p.is_dir()} if ADDONS_DIR.is_dir() else set()
    for listing in registry:
        if args.installed and listing.id not in installed:
            continue
        marker = " (installed)" if listing.id in installed else ""
        print(f"{listing.id}{marker}: {listing.name} by {listing.author['name']} [{listing.type}]")
        if listing.summary:
            print(f"  {listing.summary}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    reviewed = True
    if args.unreviewed:
        match = re.fullmatch(r"(.+)@([0-9a-f]{40})", args.unreviewed)
        if not match:
            print("error: --unreviewed must be <git url>@<commit>", file=sys.stderr)
            return 2
        repo_url, commit = match.group(1), match.group(2)
        addon_id = Path(repo_url).stem.replace(".git", "")
        try:
            validate_addon_id(addon_id)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        listing = Listing(
            id=addon_id,
            name=addon_id,
            repo=repo_url,
            commit=commit,
            author={"name": "unreviewed", "url": ""},
            summary="",
            type="tool",
        )
        reviewed = False
        print(f"Warning: {addon_id} is not on the reviewed registry.")
    else:
        if not args.id:
            print("error: add requires an id or --unreviewed", file=sys.stderr)
            return 2
        try:
            validate_addon_id(args.id)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        registry = _load_registry(args.index)
        matches = [entry for entry in registry if entry.id == args.id]
        if not matches:
            print(f"error: {args.id} is not on the reviewed registry. "
                  f"Use --unreviewed <git url>@<commit> if you still want it.",
                  file=sys.stderr)
            return 1
        listing = matches[0]

    dest = _addon_dir(listing.id)
    # Clone into a temp sibling folder so an existing install survives a "no".
    temp_parent = ADDONS_DIR
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{listing.id}-install-", dir=temp_parent) as tmp:
        temp_dest = Path(tmp)
        _clone(listing.repo, listing.commit, temp_dest)
        manifest = read_manifest(temp_dest / "addon.json")
        if manifest.id != listing.id:
            print(f"error: manifest id {manifest.id!r} does not match listing id {listing.id!r}",
                  file=sys.stderr)
            return 1
        plan = install_plan(listing, manifest, reviewed=reviewed)
        _print_plan(plan)
        if not args.yes and not _ask_yes("Install?"):
            print("Nothing installed.")
            return 1
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(temp_dest), str(dest))
    _write_install_json(dest, listing, reviewed=reviewed)
    if manifest.setup:
        print(f"Running setup: {' '.join(manifest.setup)}")
        result = subprocess.run(manifest.setup, cwd=str(dest), check=False)
        if result.returncode != 0:
            shutil.rmtree(dest, ignore_errors=True)
            print("Setup failed.", file=sys.stderr)
            return 1
    print(f"Installed {listing.id} at {dest}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    try:
        validate_addon_id(args.id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    dest = _addon_dir(args.id)
    if not dest.exists():
        print(f"error: {args.id} is not installed", file=sys.stderr)
        return 1
    shutil.rmtree(dest)
    print(f"Removed {args.id}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        validate_addon_id(args.id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    dest = _addon_dir(args.id)
    if not dest.exists():
        print(f"error: {args.id} is not installed", file=sys.stderr)
        return 1
    manifest = read_manifest(dest / "addon.json")
    tool = _tool_by_id(manifest, args.tool)

    install_info = _read_install_json(dest)
    listing = Listing(
        id=install_info["id"],
        name=manifest.name,
        repo=install_info["repo"],
        commit=install_info["commit"],
        author=manifest.author,
        summary=manifest.summary,
        type=manifest.type,
    )
    reviewed = bool(install_info.get("reviewed", False))

    raw_inputs = _collect_inputs(args.input)
    inputs = parse_inputs(raw_inputs, tool, cwd=Path.cwd())
    run_stem = args.run_stem or _default_run_stem()
    out_dir = (args.output_dir or OUTPUT_DIR / output_dir(args.id, run_stem))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()
    blender = args.blender or find_blender() or "blender"
    argv = build_command(tool, inputs, out_dir=out_dir, blender=blender)

    if not args.yes:
        print(f"Will run: {' '.join(argv)}")
        if not _ask_yes("Run?"):
            print("Not running.")
            return 1

    started = datetime.now(UTC).timestamp()
    before = _all_files(out_dir)
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(argv, cwd=str(dest), env=env, check=False)
    if result.returncode != 0:
        print("Run failed.", file=sys.stderr)
        return 1

    after = _all_files(out_dir)
    created = after - before

    result_file = (out_dir / f"{run_stem}.{tool.makes}").resolve()
    if (
        result_file in created
        or result_file.exists()
        and result_file.stat().st_mtime > started
    ):
        pass
    else:
        print(f"error: expected result {result_file.name} not written", file=sys.stderr)
        return 1

    record = run_record(listing, manifest, tool, inputs, reviewed=reviewed)
    record["output"] = {"path": str(result_file), "source": "add-on"}
    write_provenance(result_file.with_suffix(".provenance.json"), record)
    print(result_file)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "add":
        return cmd_add(args)
    if args.command == "remove":
        return cmd_remove(args)
    if args.command == "run":
        return cmd_run(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
