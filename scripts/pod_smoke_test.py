#!/usr/bin/env python3
"""Run the NVIDIA one-shot on a fresh RunPod pod, through the viewer's own web API.

    python scripts/pod_smoke_test.py --ref feat/hunyuan-nvidia
    python scripts/pod_smoke_test.py --routes trellis,pixal3d --image my.png
    python scripts/pod_smoke_test.py --routes hunyuan-cuda      # one route, no Blender

What a new user does, with nobody fixing anything mid-run: install with the one-line
installer, start the lab, sign in to Hugging Face, set up every route, generate one model
on each, install Blender, and Finish the Pixal3D model (Pixel Match included). Each step
is timed; models, records and a summary land in output/pod-smoke-<time>/.

Money and downloads are only spent after one "yes": the script names the GPU, its hourly
price cap, every route and its download size first (AGENTS.md). The pod is rented on
Secure cloud unless --cloud community, checked for download speed (a slow host turns a
40-minute test into three hours), and **always deleted at the end**, pass or fail. RunPod
also terminates it by itself after --max-hours, in case this machine dies first.

Needs `runpodctl` with an API key (`runpodctl doctor`) and an SSH key RunPod knows
(`runpodctl ssh add-key`). The Hugging Face token is read from HF_TOKEN or asked for,
and only ever sent to the pod's own viewer.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))

import backend_catalog  # noqa: E402

# Ampere like the 3090 the routes were proven on; the 4090 is the fallback.
GPUS = ("NVIDIA A40", "NVIDIA GeForce RTX 4090")
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
ROUTES = ("trellis", "hunyuan-cuda", "pixal3d")
PORT = 8777
# The installer of the ref under test: main's may not know flags the branch added.
INSTALL_URL = "https://raw.githubusercontent.com/Bingeljell/image-to-3dlab/{ref}/install.sh"
# Big public files on the hosts setup downloads from most: Hugging Face and GitHub
# releases (flash-attn's wheel). The slower one sets the pace.
SPEED_URLS = (
    "https://huggingface.co/facebook/dinov2-large/resolve/main/model.safetensors",
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/"
    "flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
)
MIN_MB_PER_S = 20.0
TERMINAL = {"done", "error", "cancelled"}
DEFAULT_IMAGE = REPO / "output" / "pod-test-2026-10-01" / "input-vanguard_plain.jpg"


# ---- pure helpers (tested) -----------------------------------------------------------

def terminate_after(hours: float, now: datetime | None = None) -> str:
    """RunPod's own kill switch, as the UTC timestamp `--terminate-after` takes."""
    now = now or datetime.now(timezone.utc)
    return (now + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


# Where the Hunyuan licence does not reach: the EU, the UK and South Korea.
BLOCKED_COUNTRIES = frozenset({
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
    "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    "GB", "KR",
})


def create_command(gpu: str, name: str, hours: float, cloud: str = "secure",
                   country: str = "US") -> list[str]:
    """Community pods only get a reachable SSH port with a public IP, which --wait needs.

    Pinned to one country because the run fetches Hunyuan weights, which are not licensed
    in the EU, the UK or South Korea, and an unpinned pod can land in any of them."""
    community = ["--public-ip"] if cloud == "community" else []
    return ["runpodctl", "pod", "create", "--cloud-type", cloud.upper(), *community,
            "--country-code", country, "--gpu-id", gpu,
            # Hosts set the driver. Below CUDA 12.9 (driver 575) Pixal3D compiles for 15
            # minutes instead of downloading its prebuilt in one.
            "--min-cuda-version", "12.9",
            "--image", IMAGE, "--name", name, "--container-disk-in-gb", "120",
            "--volume-in-gb", "60", "--ports", f"{PORT}/http,22/tcp",
            "--terminate-after", terminate_after(hours), "--wait", "-o", "json"]


def pod_price(pod: dict) -> float | None:
    """What RunPod says it will bill, which is not always the listed price."""
    for key in ("costPerHr", "adjustedCostPerHr", "cost"):
        value = pod.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def ssh_target(info: dict) -> tuple[str, int, str | None]:
    """(host, port, key file) from `runpodctl ssh info`, whose command reads
    `ssh root@1.2.3.4 -p 12345 -i ~/.ssh/key`. Found by shape, not key name."""
    command = next((v for v in info.values()
                    if isinstance(v, str) and v.startswith("ssh ") and "@" in v), None)
    if command is None:
        raise RuntimeError(f"no ssh command in runpodctl's answer: {info}")
    words = shlex.split(command)
    host = next(w for w in words if "@" in w)
    port = int(words[words.index("-p") + 1]) if "-p" in words else 22
    key = words[words.index("-i") + 1] if "-i" in words else None
    return host, port, key


def matching_private_key(registered: dict, ssh_dir: Path) -> Path | None:
    """The local private key whose public half RunPod has on file (`runpodctl ssh
    list-keys`). Without it ssh tries the default key, which RunPod may never have seen."""
    known = {" ".join(k.get("key", "").split()[:2]) for k in registered.get("keys", [])}
    for pub in sorted(ssh_dir.glob("*.pub")):
        if " ".join(pub.read_text().split()[:2]) in known and pub.with_suffix("").is_file():
            return pub.with_suffix("")
    return None


def speed_command(url: str, seconds: int = 20) -> str:
    return (f"curl -sL -o /dev/null --max-time {seconds} -w '%{{speed_download}}' "
            f"{shlex.quote(url)} || true")


def mb_per_s(curl_output: str) -> float:
    try:
        return float(curl_output.strip().split()[-1]) / 1e6
    except (ValueError, IndexError):
        return 0.0


def install_command(ref: str) -> str:
    return (f"curl -fsSL {INSTALL_URL.format(ref=ref)} | bash -s -- --ref {shlex.quote(ref)} "
            "--yes --no-start")


def lab_command(pod_id: str) -> str:
    """./lab listens on all interfaces when it sees RUNPOD_POD_ID, which an SSH session
    does not always inherit from the container, so it is passed explicitly.

    `;`, not `&&`: with `&&` the trailing `&` backgrounds the whole list, whose stdout is
    still the SSH channel, so ssh never returns."""
    return (f"cd ~/image-to-3dlab; RUNPOD_POD_ID={shlex.quote(pod_id)} "
            "nohup ./lab > lab.log 2>&1 < /dev/null &")


# Blender for the hand install, when the viewer's own install fails. Finish is what the
# run is for; the installer is a means, so its failure is recorded and then got around.
BLENDER_FALLBACK = "4.2.23"


def blender_fallback_command(version: str = BLENDER_FALLBACK) -> str:
    """Plain curl + tar into ~/blender-lts, where the lab looks. Exits non-zero on any
    failure (pipefail), so a half-unpacked Blender is never taken for a working one."""
    name = f"blender-{version}-linux-x64"
    url = f"https://download.blender.org/release/Blender{version.rsplit('.', 1)[0]}/{name}.tar.xz"
    return (f"set -eo pipefail; cd ~ && curl -fsSL {shlex.quote(url)} | tar -xJ "
            f"&& rm -rf blender-lts && mv {name} blender-lts "
            "&& ~/blender-lts/blender --background --version | head -1")


def viewer_url(pod_id: str) -> str:
    return f"https://{pod_id}-{PORT}.proxy.runpod.net"


def multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
                     f"\r\n\r\n{value}\r\n".encode())
    for name, (filename, data) in files.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                     f'filename="{filename}"\r\nContent-Type: application/octet-stream'
                     "\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def is_glb(data: bytes) -> bool:
    return len(data) > 1024 and data[:4] == b"glTF"


def sse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            try:
                events.append(json.loads(line[5:].strip()))
            except json.JSONDecodeError:
                pass
    return events


def worth_waiting(status: int, payload: dict) -> bool:
    """A 409 that says "wait" (another job or setup still running) passes; one that will
    never pass, like "Finish needs Blender", must fail now rather than in 10 minutes."""
    return status == 409 and "wait" in str(payload.get("error", "")).lower()


def setup_run_outcome(events: list[dict]) -> tuple[str, str]:
    """(status, the last lines it printed) from a setup run's event stream. The stream
    ends with a `setup_done` event; one that never arrived is "unfinished"."""
    done = next((e for e in reversed(events) if e.get("phase") == "setup_done"), None)
    lines = [str(e.get("message", "")) for e in events if e.get("phase") == "setup"]
    return (done.get("status", "error") if done else "unfinished"), "\n".join(lines[-15:])


def announcement(routes: list[str], gpus: list[str], cap: float, hours: float,
                 cloud: str = "secure") -> str:
    lines = [f"This rents one RunPod pod ({cloud} cloud) and downloads model weights onto it:",
             f"  GPU: first available of {', '.join(gpus)}, at most ${cap:.2f}/hr",
             f"  deleted at the end; RunPod terminates it anyway after {hours:g} h"]
    total = 0.0
    for route in routes:
        backend = backend_catalog.resolve(route)
        size = backend.bytes_expected / 1e9
        total += size
        lines.append(f"  {backend.label}: ~{size:.1f} GB of weights")
    lines.append(f"  total ~{total:.0f} GB, plus Blender 4.2 LTS (~0.4 GB)")
    return "\n".join(lines)


# ---- the pod -------------------------------------------------------------------------

def runpodctl(*args: str) -> dict:
    out = subprocess.run(["runpodctl", *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"runpodctl {' '.join(args)}: {out.stderr.strip() or out.stdout}")
    return json.loads(out.stdout or "{}")


class Pod:
    def __init__(self, pod: dict):
        self.id = pod["id"]
        self.host, self.port, self.key = ssh_target(runpodctl("ssh", "info", self.id))
        if not self.key or not Path(self.key).expanduser().is_file():
            self.key = matching_private_key(runpodctl("ssh", "list-keys"), Path.home() / ".ssh")

    def ssh(self, command: str, timeout: float = 1800) -> str:
        args = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30", "-p", str(self.port)]
        if self.key:
            args += ["-i", str(Path(self.key).expanduser())]
        out = subprocess.run([*args, self.host, command], capture_output=True, text=True,
                             timeout=timeout)
        if out.returncode != 0:
            raise RuntimeError(f"on the pod: {command[:80]}\n{out.stderr[-2000:]}")
        return out.stdout


def rent(gpus: list[str], cap: float, hours: float, cloud: str = "secure",
         country: str = "US") -> dict:
    name = f"pod-smoke-{datetime.now():%m%d-%H%M}"
    for gpu in gpus:
        print(f"Renting {gpu} in {country}...", flush=True)
        try:
            pod = runpodctl(*create_command(gpu, name, hours, cloud, country)[1:])
        except RuntimeError as exc:
            print(f"  not available: {str(exc).splitlines()[0][:160]}")
            continue
        price = pod_price(pod)
        if price is None or price > cap:
            delete(pod["id"])
            raise SystemExit(f"{gpu} bills ${price}/hr, over the ${cap:.2f} cap. Deleted.")
        print(f"  pod {pod['id']} at ${price:.2f}/hr")
        return {**pod, "rented_gpu": gpu}
    raise SystemExit("No GPU from the list was available. Nothing is running.")


def delete(pod_id: str) -> None:
    for _ in range(3):
        try:
            runpodctl("pod", "delete", pod_id)
            print(f"Deleted pod {pod_id}.")
            return
        except RuntimeError as exc:
            print(f"  delete failed, retrying: {exc}")
            time.sleep(10)
    print(f"!! Could not delete pod {pod_id}. Delete it by hand: runpodctl pod delete {pod_id}")


# ---- the viewer API ------------------------------------------------------------------

# Named honestly: Cloudflare refuses Python-urllib's anonymous default, not us.
USER_AGENT = "image-to-3dlab pod smoke test (+https://github.com/Bingeljell/image-to-3dlab)"


class Viewer:
    def __init__(self, base: str):
        self.base = base

    def call(self, method: str, path: str, body: bytes | None = None,
             content_type: str = "application/json", timeout: float = 120) -> tuple[int, bytes]:
        request = urllib.request.Request(self.base + path, data=body, method=method,
                                         headers={"Content-Type": content_type,
                                                  "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def json(self, method: str, path: str, payload=None, **kw) -> tuple[int, dict]:
        body = json.dumps(payload).encode() if payload is not None else kw.pop("body", None)
        status, data = self.call(method, path, body, **kw)
        try:
            return status, json.loads(data or b"{}")
        except json.JSONDecodeError:
            return status, {"raw": data[:300].decode(errors="replace")}

    def start(self, path: str, body: bytes | None = None, ctype: str = "application/json",
              patience: float = 600) -> dict:
        """POST that waits out a 409 (another job or setup still finishing)."""
        deadline = time.time() + patience
        while True:
            status, payload = self.json("POST", path, body=body, content_type=ctype)
            if status in (200, 202):
                return payload
            if not worth_waiting(status, payload) or time.time() > deadline:
                raise RuntimeError(f"POST {path}: HTTP {status} {payload}")
            time.sleep(10)

    def wait(self, path: str, limit: float, poll: float = 15) -> dict:
        """Poll until the job ends. A dropped connection is just a missed poll: over a
        90-minute run through Cloudflare, one blip must not fail a 30-minute setup."""
        deadline = time.time() + limit
        last = ""
        while time.time() < deadline:
            try:
                status, payload = self.json("GET", path)
            except OSError:
                status, payload = 0, {}
            if status == 200:
                event = payload.get("last_event") or {}
                note = str(event.get("message") or event.get("phase") or "")[:100]
                if note and note != last:
                    print(f"    {note}", flush=True)
                    last = note
                if payload.get("status") in TERMINAL:
                    return payload
            time.sleep(poll)
        raise RuntimeError(f"{path} still running after {limit / 60:.0f} min")


def interrupt_on_hangup() -> None:
    """Closing the terminal (SIGHUP) or a kill (SIGTERM) would skip `finally` and leave
    the pod billing until RunPod's own deadline. As Ctrl-C, they still delete it."""
    def handler(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, handler)


def step(results: list[dict], name: str, func, *args):
    print(f"\n== {name}", flush=True)
    began = time.time()
    try:
        detail = func(*args)
        ok = True
    except Exception as exc:  # noqa: BLE001 - every failure is a result, then we go on
        detail, ok = str(exc), False
        print(f"   FAILED: {detail[:600]}", flush=True)
    minutes = (time.time() - began) / 60
    results.append({"step": name, "ok": ok, "minutes": round(minutes, 1),
                    "detail": detail if isinstance(detail, (str, dict)) else str(detail)})
    print(f"   {'ok' if ok else 'FAILED'} in {minutes:.1f} min", flush=True)
    return ok, detail


def run_smoke(pod: Pod, args, token: str, out: Path, results: list[dict]) -> list[dict]:
    """Appends to `results` as it goes, so a Ctrl-C keeps the steps that finished."""

    def speed():
        speeds = [mb_per_s(pod.ssh(speed_command(url), 60)) for url in SPEED_URLS]
        print(f"   download speed {min(speeds):.0f} MB/s")
        if min(speeds) < MIN_MB_PER_S:
            raise RuntimeError(f"{min(speeds):.0f} MB/s is under {MIN_MB_PER_S:.0f}")
        return f"{min(speeds):.0f} MB/s"

    ok, _ = step(results, "download speed", speed)
    if not ok:
        return results
    ok, _ = step(results, "install", lambda: pod.ssh(install_command(args.ref), 1800)[-500:])
    if not ok:
        return results
    viewer = Viewer(viewer_url(pod.id))

    def start_lab():
        pod.ssh(lab_command(pod.id), 60)
        last = None
        for _ in range(40):
            try:
                last, _ = viewer.call("GET", "/api/catalog", timeout=20)
            except OSError as exc:
                last = exc
            if last == 200:
                return viewer.base + "/viewer/studio.html"
            time.sleep(6)
        raise RuntimeError(f"viewer never answered {viewer.base} (last: {last}): "
                           + pod.ssh("tail -20 ~/image-to-3dlab/lab.log"))

    ok, url = step(results, "start the lab", start_lab)
    if not ok:
        return results
    print(f"   watch along: {url}")

    def sign_in():
        status, payload = viewer.json("POST", "/api/hf/sign-in", {"token": token})
        if status != 200 or not payload.get("signed_in"):
            raise RuntimeError(f"HTTP {status}: {payload.get('error', payload)}")
        missing = [r["repo"] for r in payload.get("repos", []) if r["access"] != "yes"]
        return f"signed in as {payload.get('user')}; no access to: {missing or 'none'}"

    step(results, "Hugging Face sign-in", sign_in)

    for route in args.routes:
        def setup(route=route):
            viewer.start(f"/api/setup/{route}/download")
            final = viewer.wait(f"/api/setup/{route}/status", 90 * 60)
            if final["status"] != "done":
                raise RuntimeError(final.get("log_tail", "")[-1500:])
            return "done"
        step(results, f"set up {route}", setup)

    def blender():
        _, caps = viewer.json("GET", "/api/finish/capabilities")
        if caps.get("blender"):
            return f"already there: {caps['blender']}"
        started = viewer.start("/api/blender/install")
        # The event stream stays open until the installer exits, so this read is the wait.
        _, raw = viewer.call("GET", started["events_url"], timeout=20 * 60)
        status, tail = setup_run_outcome(sse_events(raw.decode(errors="replace")))
        if status != "done":
            raise RuntimeError(f"installer {status}:\n{tail}")
        _, caps = viewer.json("GET", "/api/finish/capabilities")
        if not caps.get("blender"):
            raise RuntimeError(caps.get("blender_problem") or "installed, but Finish cannot see it")
        return f"Blender {caps.get('blender_version')}"

    # Only Finish needs Blender, and only the Pixal3D model is finished.
    if "pixal3d" in args.routes:
        ok, _ = step(results, "install Blender", blender)
        if not ok:
            # Recorded as failed above, so the run is NOT CLEAN; Finish still gets tested.
            step(results, "install Blender by hand (fallback)",
                 lambda: pod.ssh(f"bash -c {shlex.quote(blender_fallback_command())}", 900))

    image = args.image.read_bytes()
    models: dict[str, bytes] = {}
    for route in args.routes:
        def generate(route=route):
            body, ctype = multipart({"settings": json.dumps({"backend": route})},
                                    {"image": (args.image.name, image)})
            job = viewer.start("/api/generate", body, ctype, patience=1800)
            job_id = job.get("job_id") or job.get("id")
            final = viewer.wait(f"/api/generate/{job_id}/status", 60 * 60)
            if final["status"] != "done":
                raise RuntimeError(str(final.get("last_event"))[:1500])
            _, glb = viewer.call("GET", f"/api/generate/{job_id}/result.glb", timeout=600)
            if not is_glb(glb):
                raise RuntimeError("result.glb is not a GLB")
            _, record = viewer.call("GET", f"/api/generate/{job_id}/manifest.json")
            (out / f"{route}.glb").write_bytes(glb)
            (out / f"{route}.json").write_bytes(record)
            models[route] = glb
            return f"{len(glb) / 1e6:.1f} MB"
        step(results, f"generate {route}", generate)

    if "pixal3d" in models:
        def finish():
            body, ctype = multipart({"settings": "{}"},
                                    {"asset": ("pixal3d.glb", models["pixal3d"]),
                                     "image": (args.image.name, image)})
            job = viewer.start("/api/finish", body, ctype)
            job_id = job.get("job_id") or job.get("id")
            final = viewer.wait(f"/api/finish/{job_id}/status", 60 * 60)
            if final["status"] != "done":
                raise RuntimeError(str(final.get("last_event"))[:1500])
            _, events = viewer.call("GET", f"/api/finish/{job_id}/events", timeout=60)
            stages = next((e.get("stages") for e in sse_events(events.decode(errors="replace"))
                           if e.get("stages")), [])
            if "photo" not in stages:
                raise RuntimeError(f"Pixel Match did not run; stages were {stages}")
            _, glb = viewer.call("GET", f"/api/finish/{job_id}/result.glb", timeout=600)
            (out / "pixal3d-finished.glb").write_bytes(glb)
            return f"stages {stages}; {len(glb) / 1e6:.1f} MB"
        step(results, "Finish the Pixal3D model (Pixel Match)", finish)

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="main", help="branch or tag the installer takes")
    parser.add_argument("--routes", default=",".join(ROUTES),
                        help=f"comma-separated (default: {','.join(ROUTES)})")
    parser.add_argument("--gpu", action="append", help="GPU id to try, in order (repeatable)")
    parser.add_argument("--cloud", choices=("secure", "community"), default="secure",
                        help="community is cheaper (a 3090 for ~$0.22/hr) but rarer with "
                             "a public IP")
    parser.add_argument("--country", default="US", type=str.upper,
                        help="where the pod may run (US). Hunyuan's weights are not licensed "
                             "in the EU, the UK or South Korea, so never pick one of those")
    parser.add_argument("--max-price", type=float, default=0.80, help="$/hr cap (0.80)")
    parser.add_argument("--max-hours", type=float, default=3.0,
                        help="RunPod terminates the pod after this, whatever happens (3)")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="input picture")
    parser.add_argument("--keep", action="store_true", help="do not delete the pod at the end")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = parser.parse_args(argv)
    args.routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    unknown = [r for r in args.routes if r not in ROUTES]
    if unknown:
        parser.error(f"unknown route: {', '.join(unknown)}")
    if args.country in BLOCKED_COUNTRIES and "hunyuan-cuda" in args.routes:
        parser.error(f"Hunyuan's weights are not licensed in {args.country}; pick another "
                     "--country or drop hunyuan-cuda from --routes")
    if not args.image.is_file():
        parser.error(f"no input picture at {args.image}; pass --image")
    gpus = args.gpu or list(GPUS)

    print(announcement(args.routes, gpus, args.max_price, args.max_hours, args.cloud))
    if not args.yes and input("Rent the pod and download all of that? [y/N] ").strip().lower() \
            not in {"y", "yes"}:
        print("Nothing rented.")
        return 1
    token = (os.environ.get("HF_TOKEN")
             or getpass.getpass("Hugging Face Read token (not shown): ")).strip()

    out = REPO / "output" / f"pod-smoke-{datetime.now():%Y%m%d-%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    interrupt_on_hangup()
    pod_json = rent(gpus, args.max_price, args.max_hours, args.cloud, args.country)
    results: list[dict] = []
    try:
        run_smoke(Pod(pod_json), args, token, out, results)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if args.keep:
            print(f"Kept pod {pod_json['id']}; RunPod terminates it after {args.max_hours:g} h.")
        else:
            delete(pod_json["id"])
        hours = (time.time() - started) / 3600
        price = pod_price(pod_json) or 0.0
        summary = {"ref": args.ref, "gpu": pod_json.get("machine", {}).get("gpuDisplayName")
                   or pod_json.get("gpuTypeId") or pod_json.get("rented_gpu"),
                   "price_per_hr": price,
                   "hours": round(hours, 2), "cost": round(hours * price, 2), "steps": results}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n{'step':45} {'result':8} min")
        for r in results:
            print(f"{r['step']:45} {'ok' if r['ok'] else 'FAILED':8} {r['minutes']}")
        print(f"~${summary['cost']:.2f} over {hours:.1f} h. Results in {out.relative_to(REPO)}")
    clean = bool(results) and all(r["ok"] for r in results)
    print("ONE-SHOT: CLEAN" if clean else "ONE-SHOT: NOT CLEAN")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
