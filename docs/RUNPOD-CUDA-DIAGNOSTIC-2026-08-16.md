# RunPod CUDA diagnostic pod — setup notes (2026-08-16)

**Why this exists:** our Mac port runs TRELLIS.2 on MPS/SDPA, and when a texture/material
result looks wrong (see the Bloomglass darkening investigation in
[STATE-OF-REPO-2026-08-16.md](STATE-OF-REPO-2026-08-16.md)) the only way to tell "our port's
bug" from "the model just does this" is a real CUDA run of the **unmodified upstream**
TRELLIS.2 repo on the exact same input image and seed. This doc is the runbook for standing
one up on RunPod, plus every snag hit doing it the first time — read it before repeating the
exercise instead of rediscovering these.

## Tooling / auth setup

1. Install the Claude Code plugin: `claude plugin marketplace add
   runpod/runpod-plugins-official` then `claude plugin install runpod@runpod`, `/reload-plugins`,
   `/mcp` → **runpod** → **Sign in with Runpod**.
2. **That OAuth only authenticates the MCP tools** — `runpodctl` (needed for SSH key
   registration, which the MCP tools don't expose) and `flash` stay unauthenticated. Get a real
   key from https://console.runpod.io/user/settings and either `export RUNPOD_API_KEY=...` or
   write it to `~/.runpod/config.toml` as `apikey = '...'`.
3. Install `runpodctl`. `curl -sSL https://cli.runpod.net | bash` wants sudo; on macOS use
   `brew install runpod/runpodctl/runpodctl` instead.
4. Register an SSH key **before creating any pod** — Runpod injects registered keys into
   `~/.ssh/authorized_keys` at boot only; a key added after boot needs a restart to take effect.
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/<name> -N ''
   runpodctl ssh add-key --key-file ~/.ssh/<name>.pub
   runpodctl ssh list-keys   # confirm
   ```

## Creating the pod

- Community cloud stock status ("High"/"Low") is not a reliable predictor of actual
  availability. In one session, an RTX 4090 (stock: High, EU-RO-1) and an RTX A6000 (stock:
  Low) both failed with `"no longer any instances available"` / `"does not have the resources
  to deploy"`; an RTX 3090 with no data-center pinned succeeded immediately. **If create fails
  on availability, retry with a different GPU model and/or drop `--data-center-ids`.**
- `--container-disk-in-gb` **defaults to 20GB** — far too small. The CUDA 13 venv alone (torch
  2.11 + all the `nvidia-*` split packages) is ~6GB before any model weights; TRELLIS.2-4B is
  ~14GB and DINOv3 conditioning adds more. **Use 80GB+.**
- Community cloud SSH needs `--cloud-type COMMUNITY --public-ip` together (the flag is
  community-only).
- Worked example (RTX 3090, 24GB, $0.22/hr, no DC pin):
  ```bash
  runpodctl pod create --name trellis-cuda-diff \
    --template-id runpod-torch-v280 --gpu-id "NVIDIA GeForce RTX 3090" \
    --cloud-type COMMUNITY --public-ip \
    --container-disk-in-gb 80 \
    --ports "22/tcp" \
    --ssh --terminate-after <iso8601, a few hours out — cost guard> \
    --wait --wait-timeout 6m
  ```
- **Always set `--terminate-after` as a cost guard**, and remove the pod explicitly
  (`runpodctl pod remove <id>`) the moment you're done rather than relying on it — don't leave
  a meter running on a throwaway diagnostic box.
- **Check the host is actually healthy before doing the full setup.** `nvidia-smi` alone,
  right after create, is cheap and catches two real failure modes hit live: (1) a driver too
  old for what you need (`torch==2.11.0+cu130` needs a driver new enough for CUDA 13 — one
  community host had driver 570.195.03/CUDA 12.8 and `torch.cuda.is_available()` returned
  `False`), and (2) a **stuck GPU** — one community host reported `100% GPU-Util` with `No
  running processes found` and `2MiB` used, and `torch.cuda.init()` failed with `CUDA unknown
  error` consistently (not transient) — a genuine bad machine draw, not something to debug
  further. **Community Cloud is documented as "variable reliability"** (`runpod-usage`
  skill's `gpu-selection.md`) — cheaper, but hit two bad hosts in a row live in this session.
  **Secure Cloud worked cleanly on the first try** after that (higher price, but "high
  redundancy, stable public IPs" per the same doc). If community keeps handing back bad
  hosts, switch to secure rather than repeatedly re-rolling community.
- Landing on the exact same IP twice in a row on community cloud (even after explicitly
  removing the pod and recreating) can happen — community capacity for a given GPU model in a
  region can be genuinely thin, and the scheduler may keep handing back the same (bad) host.
  If that happens, try pinning a different `--data-center-ids`, or switch cloud type.

**Standing default: try community first, gate on a health check, fall back to secure.** The
health check (`ssh ... nvidia-smi`, right after `pod create`, before touching pip/git) costs
one SSH round-trip — seconds, not minutes — so there's no real reason to skip straight to the
pricier secure tier. `remove` + recreate immediately if it fails (wrong driver, or
`GPU-Util: 100%` with `No running processes found` — a stuck GPU). Only switch to
`--cloud-type SECURE` if community keeps striking out (this session needed 2 community
retries before falling back, and secure worked cleanly on the first try after that).

## Getting the real upstream repo + deps

**Next time, start with `git clone https://github.com/microsoft/TRELLIS.2` (the real GitHub
repo), not the HF Space.** This session cloned the HF Space
(`huggingface.co/spaces/microsoft/TRELLIS.2`) instead, which turned out to be the wrong
source for our actual goal:

- The **HF Space**'s `requirements.txt` pins prebuilt wheel URLs for **one specific GPU**
  (see "Two blockers" below — `flex_gemm`'s cubin is `sm_120`-only, no PTX fallback, verified
  with `cuobjdump`) — it's a frozen snapshot of whatever hardware that particular Space
  instance happens to run on, not a portable install.
- The **GitHub repo**'s README documents a proper install script (`--new-env` flag) that
  **builds flash-attn/nvdiffrast/nvdiffrec/cumesh/flexgemm/o-voxel from source** for whatever
  GPU is actually present, and states: *"An NVIDIA GPU with at least 24GB of memory is
  necessary. The code has been verified on NVIDIA A100 and H100 GPUs."* — no Blackwell
  requirement at all. It also documents an `xformers` fallback path for GPUs that don't
  support flash-attn (e.g. V100).

**The goal of this exercise is to find our own port's bug, not to chase upstream's exact
deployment hardware.** We don't need to reproduce the HF Space bit-for-bit — we need *a*
correct CUDA reference to diff against, on hardware we can rent cheaply. The GitHub repo's
from-source path should let a $0.22-0.74/hr card (3090/4090) work, instead of the $1.69-2.09/hr
Blackwell rental this session ended up needing. Building from source costs real setup time
(compiling several CUDA extensions), so it's a real tradeoff against the HF Space's prebuilt
wheels being instant *if* your GPU happens to match — but for **repeatable, cheap** diagnostic
runs, from-source on commodity hardware is the right default. Only fall back to renting the
exact matching GPU (as this session did) if a from-source build itself fails or is out of
budget for the time available.

If you do end up needing the HF Space specifically (e.g. testing something Space-deployment
specific): `git clone https://huggingface.co/spaces/microsoft/TRELLIS.2` — code only, ~5MB;
weights are pulled at runtime via `from_pretrained`.
- `uv venv --python 3.12 .venv` (the Space's `requirements.txt` pins `cp312` wheels).
- `uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r
  requirements.txt`. **Without `--index-strategy unsafe-best-match` this fails** — the
  requirements file mixes a package-specific `--extra-index-url` (PyTorch's cu130 wheel index)
  with default PyPI, and uv's default index strategy won't consider a later index for a package
  once an earlier index has *any* version of it (hit an unresolvable `tqdm` conflict).
- **The prebuilt CUDA extension wheels ARE architecture-locked to the GPU named in their URL.**
  `flash-attn`/`flex_gemm`/`nvdiffrast`/`nvdiffrec_render`/`cumesh`/`o_voxel` are all published
  for `rtxpro6000` (RTX PRO 6000 Blackwell) specifically. They **import** fine on an RTX 3090
  (Ampere) — Python-level import doesn't touch the GPU — but the first real kernel dispatch
  (`flex_gemm`'s sparse conv, hit during `pipeline.run()`) fails with `torch.AcceleratorError:
  CUDA error: no kernel image is available for execution on the device`: the wheel's
  precompiled binary has no machine code for Ampere's compute capability. **Use an actual RTX
  PRO 6000 pod** (`runpodctl gpu list | grep -i "pro 6000"` — Server or Workstation Edition,
  96GB, ~$1.69-2.09/hr) — a cheaper Ampere/Ada card (3090/4090/A6000) will not work with these
  specific prebuilt wheels no matter how much VRAM it has. (An earlier version of this note
  wrongly claimed the GPU name in the wheel filename "is not a hard compatibility gate" based
  only on successful imports — that was disproven by the actual runtime crash; import success
  is not proof a CUDA extension works.)

## Two blockers specific to running the *real* upstream code

1. **`from_pretrained()` signature differs from our Mac port.** Our port's
   `trellis_space_generate.py` calls `Trellis2ImageTo3DPipeline.from_pretrained(path,
   load_rembg=...)` — that `load_rembg` kwarg is a **Mac-port-specific patch**, not present
   upstream. The real signature is just `from_pretrained(path)`; rembg loads lazily inside
   `preprocess_image()` only if the image lacks alpha. **Don't carry our port's API surface
   assumptions over** — `grep -n "def from_pretrained\|def run(\|def decode_latent" ...` on the
   actual cloned repo before an expensive run to catch signature drift in seconds instead of
   after a 14GB download + full sample.
2. **TRELLIS.2 depends on a gated HuggingFace model**:
   `facebook/dinov3-vitl16-pretrain-lvd1689m`. Meta requires explicit license acceptance +
   access approval on that model's HF page before it's downloadable. Needs an HF token with
   access already granted. Verify before transferring:
   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" \
     -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
     https://huggingface.co/api/models/facebook/dinov3-vitl16-pretrain-lvd1689m
   # 200 = has access
   ```
   Then `scp` the token file directly to `~/.cache/huggingface/token` **on the pod** (never
   `cat`/echo the raw token into a command that gets logged) before `from_pretrained`/`.run()`.
   This repo's local machine already had a granted token cached locally, reused as-is.

## Operational notes

- SSH commands that launch a background job on the pod (`nohup ... &`) can hit the local tool's
  ~120s SSH timeout even though the remote job launches fine and keeps running — the launch
  call just gets auto-backgrounded. Don't treat that timeout as a failure; poll the remote
  process/log separately (`ps aux | grep <script>`, `tail <log>`) to confirm it's actually
  alive.
- To mirror our Mac port's params exactly for a real comparison, copy `DEMO_PARAMS` verbatim
  from `scripts/trellis_space_generate.py` (seed, steps, guidance strength/rescale per stage,
  `pipeline_type`) rather than trusting defaults — the upstream `run()`'s own default `seed=42`
  differs from the demo's actual `seed=0`.

## Cleanup

```bash
runpodctl pod list             # confirm what's running
runpodctl pod remove <pod-id>  # do this explicitly, don't rely solely on --terminate-after
```
