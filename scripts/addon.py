#!/usr/bin/env python3
"""Install and run add-ons: someone else's repo with an `addon.json`.

    ./lab addon list
    ./lab addon add vehicle-kit
    ./lab addon add --unreviewed https://github.com/example/pedestal.git@<commit>
    ./lab addon remove vehicle-kit
    ./lab addon run vehicle-kit build --input preset=bus_12m

Add-ons are listed at `https://assetfurnace.com/add-ons/index.json`.
Only listed names can be installed; anything else needs `--unreviewed` and a warning.
Each add-on is cloned at a pinned commit, checked out, verified, and run in its own
folder with `HF_HUB_OFFLINE=1`. The core writes the `.provenance.json` sidecar.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab.addon_install import (
    install_addon,
    list_installed,
    load_registry,
    remove_addon,
    run_addon,
)

ADDONS_DIR = Path(os.environ.get("ADDONS_DIR", REPO / "addons"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", REPO / "output"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List reviewed add-ons.")
    list_cmd.add_argument("--index", help="local list file for tests or review")
    list_cmd.add_argument("--installed", action="store_true",
                          help="Only show add-ons already installed.")

    add_cmd = sub.add_parser("add", help="Install an add-on after showing its plan.")
    add_cmd.add_argument("--index", help="local list file for tests or review")
    add_cmd.add_argument("id", nargs="?", help="id of a listed add-on")
    add_cmd.add_argument("--unreviewed", help="git url@commit, not on the list")
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


def _ask_yes(prompt: str = "Continue?") -> bool:
    if not sys.stdin.isatty():
        return False
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def _collect_inputs(items: list[str]) -> dict[str, str]:
    """Split --input key=value strings without coercing their types."""
    inputs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--input must be key=value: {item!r}")
        key, value = item.split("=", 1)
        inputs[key] = value
    return inputs


def _print_plan(plan: dict) -> None:
    print(f"Author: {plan['author']}")
    print(f"Licence: {plan['licence']}")
    if plan["models"]:
        print("Models it loads:")
        for model in plan["models"]:
            print(f"  - {model['name']} ({model['licence']}, {model['size']})")
    print(f"Setup: {' '.join(plan['setup'])}")
    print(plan["warning"])


def _link_line(listing) -> str:
    parts = []
    if listing.author.get("url"):
        parts.append(f"author: {listing.author['url']}")
    if listing.links.get("repo"):
        parts.append(f"repo: {listing.links['repo']}")
    if listing.links.get("site"):
        parts.append(f"site: {listing.links['site']}")
    return "  ".join(parts)


def cmd_list(args: argparse.Namespace) -> int:
    try:
        registry = load_registry(args.index)
    except Exception as exc:
        print(f"error: could not load the add-on list: {exc}", file=sys.stderr)
        return 1
    installed = set(list_installed(ADDONS_DIR))
    for listing in registry:
        if args.installed and listing.id not in installed:
            continue
        marker = " (installed)" if listing.id in installed else ""
        rating = ""
        if listing.rating is not None:
            count = f" ({listing.ratings_count})" if listing.ratings_count else ""
            rating = f" ★ {listing.rating}{count}"
        print(f"{listing.id}{marker}: {listing.name} by {listing.author['name']}{rating} [{listing.type}]")
        if listing.summary:
            print(f"  {listing.summary}")
        link_line = _link_line(listing)
        if link_line:
            print(f"  {link_line}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    try:
        result = install_addon(
            addon_id=args.id,
            unreviewed=args.unreviewed,
            index=args.index,
            confirm=False,
            addons_dir=ADDONS_DIR,
            repo_dir=REPO,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    plan = result["plan"]
    _print_plan(plan)
    if not args.yes and not _ask_yes("Install?"):
        print("Nothing installed.")
        return 1

    try:
        install_addon(
            addon_id=args.id,
            unreviewed=args.unreviewed,
            index=args.index,
            confirm=True,
            addons_dir=ADDONS_DIR,
            repo_dir=REPO,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    dest = ADDONS_DIR / result["listing"].id
    print(f"Installed {result['listing'].id} at {dest}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    try:
        remove_addon(args.id, ADDONS_DIR)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Removed {args.id}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    raw_inputs = _collect_inputs(args.input)
    try:
        result_file = run_addon(
            addon_id=args.id,
            tool_id=args.tool,
            inputs=raw_inputs,
            run_stem=args.run_stem,
            output_root=args.output_dir or OUTPUT_DIR,
            addons_dir=ADDONS_DIR,
            repo_dir=REPO,
            blender=args.blender,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
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
