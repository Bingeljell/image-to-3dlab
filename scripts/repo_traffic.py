"""Save a repo's GitHub traffic and its one-line install counts into one CSV, a row per day.

GitHub keeps views and clones for 14 days only, so anything not saved is gone. Run this
weekly (or more often: re-running is safe, a day already saved is just refreshed) and the
CSV becomes the long history GitHub never keeps.

Install counts come from the landing site's private ``/api/installs`` (see the
assetfurnace repo, ``lib/installs.js``). They are the closest thing we have to a user
count: clones include bots and re-clones, and the app itself sends nothing home. Without
a token file the install columns are simply left empty.

    python scripts/repo_traffic.py                      # this repo → journal/traffic.csv
    python scripts/repo_traffic.py --repo owner/name --out traffic.csv --no-installs
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

FIELDS = ["date", "views", "unique_views", "clones", "unique_clones", "installs_sh", "installs_ps1"]
DEFAULT_INSTALLS_URL = "https://assetfurnace.com/api/installs"
DEFAULT_TOKEN_FILE = Path.home() / ".config" / "assetfurnace" / "stats_token"


def rows_from_github(views: dict, clones: dict) -> dict[str, dict]:
    """GitHub's /traffic/views and /traffic/clones payloads → {date: fields}."""
    rows: dict[str, dict] = {}
    for payload, key, total, unique in ((views, "views", "views", "unique_views"),
                                        (clones, "clones", "clones", "unique_clones")):
        for day in payload.get(key, []):
            row = rows.setdefault(day["timestamp"][:10], {})
            row[total] = day["count"]
            row[unique] = day["uniques"]
    return rows


def rows_from_installs(counts: dict) -> dict[str, dict]:
    """The site's {date: {sh, ps1}} → {date: fields}."""
    return {day: {"installs_sh": c.get("sh", 0), "installs_ps1": c.get("ps1", 0)}
            for day, c in counts.items()}


def merge(saved: dict[str, dict], *fresh: dict[str, dict]) -> dict[str, dict]:
    """Fresh numbers win field by field; fields a fresh source did not report keep their
    saved value. GitHub's newest day is partial, so a later run must be able to correct it."""
    out = {day: dict(row) for day, row in saved.items()}
    for source in fresh:
        for day, row in source.items():
            out.setdefault(day, {}).update(row)
    return out


def read_csv(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {r["date"]: {k: int(v) for k, v in r.items() if k != "date" and v != ""}
                for r in csv.DictReader(f)}


def write_csv(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for day in sorted(rows):
            w.writerow({"date": day, **rows[day]})


def last_days(rows: dict[str, dict], n: int = 7) -> dict[str, int]:
    """Totals over the newest ``n`` days saved, for the one-line summary."""
    days = sorted(rows)[-n:]
    return {k: sum(rows[d].get(k, 0) for d in days) for k in FIELDS[1:]}


def gh_api(path: str) -> dict:
    return json.loads(subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True).stdout)


def current_repo() -> str:
    out = subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                         check=True, capture_output=True, text=True)
    return out.stdout.strip()


def fetch_installs(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"authorization": f"Bearer {token}",
                                               "user-agent": "repo_traffic.py"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", help="owner/name (default: the repo you are in)")
    ap.add_argument("--out", type=Path, default=Path("journal/traffic.csv"))
    ap.add_argument("--installs-url", default=DEFAULT_INSTALLS_URL)
    ap.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    ap.add_argument("--no-installs", action="store_true", help="GitHub numbers only")
    args = ap.parse_args(argv)

    repo = args.repo or current_repo()
    fresh = [rows_from_github(gh_api(f"repos/{repo}/traffic/views"),
                              gh_api(f"repos/{repo}/traffic/clones"))]
    if not args.no_installs:
        if args.token_file.exists():
            try:
                fresh.append(rows_from_installs(fetch_installs(args.installs_url,
                                                               args.token_file.read_text().strip())))
            except OSError as e:  # the site being down must not cost us GitHub's 14 days
                print(f"installs skipped: {e}", file=sys.stderr)
        else:
            print(f"installs skipped: no token at {args.token_file}", file=sys.stderr)

    rows = merge(read_csv(args.out), *fresh)
    write_csv(args.out, rows)
    week = last_days(rows)
    print(f"{args.out}: {len(rows)} days saved. Last 7: {week['unique_views']} visitors, "
          f"{week['unique_clones']} cloners (daily uniques, summed), {week['installs_sh'] + week['installs_ps1']} installs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
