#!/usr/bin/env python3
"""Build a hero character: a full-body model with a high-detail head grafted on, as one GLB.

    .venv/bin/python scripts/hero_route.py picture.png [--back back.png] [--out output/hero/NAME]
    ... --body model.glb --body-views model.svviews [--body-dump model.post.bin] --result out.glb

A face that is a few dozen pixels of a full-body picture comes out of Pixal3D soft: goggle
eyes, smeared paint. The hero route builds the head again from an upscaled crop, where it
fills Pixal3D's whole box, and joins it to the body:

  1. body       Pixal3D on the picture (front, or front + back), Finish to --body-faces
                with Pixel Match.
  2. measure    a throwaway rig of the body. Its topmost bone region (+ the bones whose
                middle sits inside it) is the head: no names, no hand-drawn box.
  3. crop       the head's pixels, squared and padded, upscaled 4x with Real-ESRGAN.
  4. head       Pixal3D on the crop, cleaned up to --head-faces, Pixel Match from the crop.
  5. graft      the head placed on the body's head (same centre, same height), lined up,
                swapped in along the body head's drawn outline.
  6. join       one object, two materials (body, head): a skin later swaps only the body's
                paint sheet, so the face survives. Textures compressed.
  7. check      close-ups of the head and the whole figure, to judge before using it.

Rigging and animation are downstream: rig `hero.glb` like any model. Every step skips when
its output exists, so a rerun picks up after a fix. Where the code is unsure (a fallback,
a coin-flip check) the log says UNSURE: that is where a human should look.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "viewer"))
sys.path.insert(0, str(REPO / "scripts"))

# The Python running this script: a hard-coded .venv/bin/python does not exist on Windows.
PYTHON = Path(sys.executable)
GENERATE = REPO / "scripts" / "pixal3d_generate.py"
FINISH = REPO / "scripts" / "retopo_repaint.py"
UPSCALE = REPO / "scripts" / "upscale_esrgan.py"
BLENDER = REPO / "scripts" / "hero_blender.py"
COMPRESS = REPO / "scripts" / "compress_glb_textures.py"
BODY_FACES = 32000
HEAD_FACES = 16000
#: How much of a failed step's own output is echoed, so a bug report carries the real error.
FAIL_TAIL_LINES = 15
SEED = 42
#: Finish settings shared with the studio's defaults, Pixel Match only (no repaint).
FINISH_FLAGS = ["--atlas", "2048", "--angle", "89.0", "--voxel", "0.004", "--metallic", "0.25",
                "--roughness", "0.65", "--ior", "1.45", "--paint-seed", "0", "--paint-res", "512",
                "--paint-steps", "15", "--paint-tex", "4096", "--texture-size", "2048", "--skip-paint"]
#: What else touched the output; Pixal3D's own licence is copied from its run records.
LICENCES = [
    {"component": "Real-ESRGAN x4plus", "license": "BSD-3-Clause", "purpose": "head crop upscale"},
    {"component": "SkinTokens", "license": "MIT", "purpose": "throwaway rig to find the head; not in the output"},
]


class Run:
    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        # UTF-8 throughout: Windows defaults to cp1252, which can't write every folder name.
        self.log = open(out / "run.log", "a", buffering=1, encoding="utf-8")  # noqa: SIM115 - open for the whole run
        # Kept on disk by step: a rerun skips finished steps and their notes must survive, while
        # a step that runs again replaces its own old notes.
        self.unsure_file = out / "unsure.json"
        self.notes: dict[str, list[str]] = (json.loads(self.unsure_file.read_text(encoding="utf-8"))
                                            if self.unsure_file.is_file() else {})
        if isinstance(self.notes, list):                  # older runs kept a flat list
            self.notes = {"earlier": self.notes}
        self.current = "start"
        self.started = time.time()

    def say(self, text: str) -> None:
        line = f"[hero {time.time() - self.started:6.0f}s] {text}"
        print(line, flush=True)
        self.log.write(line + "\n")

    @property
    def unsure(self) -> list[str]:
        return [note for notes in self.notes.values() for note in notes]

    def flag(self, text: str) -> None:
        notes = self.notes.setdefault(self.current, [])
        if text not in notes:
            notes.append(text)
            self.unsure_file.write_text(json.dumps(self.notes, indent=2), encoding="utf-8")
        self.say(f"UNSURE: {text}")

    def run(self, command: list, cwd: Path | None = None, tries: int = 1, required: bool = True) -> bool:
        for attempt in range(1, tries + 1):
            self.say("$ " + " ".join(str(c) for c in command))
            process = subprocess.Popen([str(c) for c in command], cwd=cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
            tail: deque[str] = deque(maxlen=FAIL_TAIL_LINES)
            for line in process.stdout:
                self.log.write(line)
                tail.append(line.rstrip("\n"))
            if not process.wait():
                return True
            if attempt < tries:
                self.flag(f"{Path(str(command[1])).name} failed once ({process.returncode}); tried again")
        if required:
            print("\n".join(tail), flush=True)   # run.log has it all; the studio only sees stdout
            last = next((line.strip() for line in reversed(tail) if line.strip()), "")
            raise SystemExit(f"[hero] step failed ({process.returncode})" + (f": {last}" if last else "")
                             + f"; see {self.out / 'run.log'}")
        return False

    def step(self, name: str, output: Path) -> bool:
        """True when the step must run; False (and a note) when its output is already there."""
        self.current = name
        if output.exists():
            self.say(f"{name}: kept {output.name}")
            return False
        self.notes.pop(name, None)
        self.notes.pop("earlier", None)
        self.unsure_file.write_text(json.dumps(self.notes, indent=2), encoding="utf-8")
        self.say(f"{name} ...")
        return True


STAGE_MARKER = "I2L_STAGE"   # the studio's Finish progress reads these (viewer/finish_api.py)
STAGES = ("body", "measure", "crop", "head", "graft", "join", "check")


def stage(run: Run, name: str, message: str) -> None:
    """Announce a stage on stdout for the studio, and in the log."""
    print(f"{STAGE_MARKER}::{name}::{message}", flush=True)
    run.log.write(f"{STAGE_MARKER}::{name}::{message}\n")


def missing_parts(parts: dict[str, Path]) -> list[str]:
    """The named parts whose file is not on disk: checked before 20 minutes of work, not after."""
    return [name for name, path in parts.items() if not Path(path).exists()]


def needed_parts() -> dict[str, Path]:
    """Everything the route runs, by what to install to get it."""
    import autorig_api
    import pixal3d_generate
    import retopo_repaint

    from image_to_3dlab import pixal3d_native, upscale

    return {
        "Pixal3D (3D models)": pixal3d_generate.CLI,
        "Auto-rig (finds the head)": autorig_api.venv_python(),
        "Hero face upscaler (Real-ESRGAN, 67 MB)": upscale.WEIGHTS_FILE,
        "Blender (Finish)": retopo_repaint.find_blender() or Path("/missing/blender"),
        # Upstream's old NVIDIA download lacks it; Setup's Rebuild for Pixal3D adds it.
        "Pixal3D's own clean-up (Setup: Rebuild Pixal3D)": pixal3d_native.POST_REPLAY,
    }


NO_DUMP = ("this 3D model kept no clean-up data, so its body would stay at full size. Make the 3D "
           "model again (same picture), then make the Hero version from the new one.")


#: Checked before the run: every Hero step needs the dump, and a model made again on this
#: build would still have none.
NO_DUMP_BUILD = ("Pixal3D on this machine can't save its clean-up data, which the Hero version needs. "
                 "Open Setup and pick Rebuild for Pixal3D, then try again (nothing was downloaded).")


def body_problem(body: Path | None, views: Path | None, dump: Path | None) -> str | None:
    """Why an existing model cannot be the Hero body, in plain words, or None."""
    if body is None:
        return None
    if not (views and (Path(views) / "transforms.json").is_file()):
        return ("the body model needs its cameras (--body-views with transforms.json): the hero "
                "route works on Pixal3D models made from a picture.")
    if not (dump and Path(dump).is_file()):
        return NO_DUMP
    return None


def finish(run: Run, model: Path, picture: Path, views: Path, out: Path, faces: int, dump: Path | None) -> None:
    command = [PYTHON, "-u", FINISH, model, picture, out, "--steps-dir", out.with_suffix(".steps"),
               "--views", views, "--faces", str(faces), *FINISH_FLAGS]
    command += ["--native-dump", dump] if dump else ["--keep-mesh"]
    run.run(command)
    record = json.loads(out.with_suffix(".retopo-repaint.json").read_text())
    route = (record.get("retopology") or {}).get("route")
    if dump and route == "classic":
        run.flag(f"Finish of {model.name} fell back to the classic clean-up (its frame check "
                 "objected; see run.log): hair and thin parts can shatter")


def head_cleanup(run: Run, raw: Path, out: Path, faces: int) -> None:
    """Pixal3D's own clean-up at FACES. Its mirror check is a coin-flip on a front-facing,
    near-symmetric head (0.2% vs 0.3% on the elf) and would drop to the classic route, which
    shatters hair; a head crop is always front-facing, so convert directly and say so."""
    from image_to_3dlab import pixal3d_native as native

    replay = out.with_name(out.stem + ".replay.glb")
    run.run(native.replay_command(raw.with_suffix(native.DUMP_SUFFIX), replay, faces, 2048))
    out.write_bytes(native.convert_replay_glb(replay.read_bytes()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("picture", type=Path)
    parser.add_argument("--back", type=Path, default=None, help="a back view of the same picture")
    parser.add_argument("--out", type=Path, default=None, help="run folder (default output/hero/<picture>)")
    parser.add_argument("--body-faces", type=int, default=BODY_FACES)
    parser.add_argument("--head-faces", type=int, default=HEAD_FACES)
    parser.add_argument("--match-paint", action="store_true",
                        help="also move the head's paint to the body's level (metal and roughness always are)")
    parser.add_argument("--cut-at", default=None,
                        help="swap only the head above this height (model units), or 'neck' to find it; "
                             "default swaps the whole head region")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--body", type=Path, default=None,
                        help="an existing Pixal3D model of the picture, used as the body instead of building one")
    parser.add_argument("--body-views", type=Path, default=None, help="that model's cameras (its .svviews folder)")
    parser.add_argument("--body-dump", type=Path, default=None, help="that model's clean-up dump (.post.bin)")
    parser.add_argument("--body-record", type=Path, default=None, help="that model's run record, for provenance")
    parser.add_argument("--result", type=Path, default=None, help="also copy the finished GLB here")
    args = parser.parse_args()

    missing = missing_parts(needed_parts())
    if missing:
        print("[hero] cannot start, not installed: " + "; ".join(missing)
              + ". Install them from the Setup page (nothing was downloaded).", file=sys.stderr)
        return 2
    import pixal3d_generate

    if not pixal3d_generate.dump_supported():
        print(f"[hero] {NO_DUMP_BUILD}", file=sys.stderr)
        return 2
    problem = body_problem(args.body, args.body_views, args.body_dump)
    if problem:
        print(f"[hero] {problem}", file=sys.stderr)
        return 2

    import numpy as np
    import trimesh
    from PIL import Image

    from image_to_3dlab import hero, matte
    from image_to_3dlab.pixal3d_native import smooth_normals
    from image_to_3dlab.provenance import sha256_file

    out = (args.out or REPO / "output" / "hero" / args.picture.stem[:60]).resolve()
    run = Run(out)
    run.say(f"hero route: {args.picture} -> {out}")
    picture = out / "picture.png"
    if not picture.exists():
        shutil.copy(args.picture, picture)
    back = None
    if args.back:
        back = out / "back.png"
        if not back.exists():
            shutil.copy(args.back, back)
    cutout = out / "picture_cutout.png"
    if run.step("cut-out", cutout):
        image, used = matte.cut_out(Image.open(picture).convert("RGB"))
        image.save(cutout)
        run.say(f"cut out with {used}")

    # 1. body
    body_raw = out / "body_raw.glb"
    stage(run, "body", "Building the body" if not args.body else "Finishing the body")
    if args.body and not body_raw.exists():
        # The studio's own 3D model of this picture: no second Pixal3D run for the body.
        shutil.copytree(args.body_views, body_raw.with_suffix(".svviews"), dirs_exist_ok=True)
        if args.body_dump and args.body_dump.is_file():
            shutil.copy(args.body_dump, body_raw.with_suffix(".post.bin"))
        if args.body_record and args.body_record.is_file():
            shutil.copy(args.body_record, body_raw.with_suffix(".json"))
        shutil.copy(args.body, body_raw)
        run.say(f"body: using {args.body}")
    if run.step("body: Pixal3D", body_raw):
        run.run([PYTHON, "-u", GENERATE, picture, body_raw, "--seed", str(args.seed),
                 *(["--back", back] if back else [])])
    body = out / "body.glb"
    if run.step(f"body: Finish {args.body_faces} + Pixel Match", body):
        dump = body_raw.with_suffix(".post.bin")
        if not dump.is_file():
            raise SystemExit(f"[hero] {NO_DUMP}")
        finish(run, body_raw, picture, body_raw.with_suffix(".svviews"), body, args.body_faces, dump)

    # 2. measure
    stage(run, "measure", "Finding the head")
    rig = out / "measure_rig.glb"
    if run.step("measure: throwaway rig", rig):
        import autorig_api

        # The rigger's own Blender server sometimes answers its first request badly
        # (BadStatusLine on the long-haired woman); a second go is cheap next to a lost run.
        for command, cwd in autorig_api.rig_commands(body, rig):
            run.run(command, cwd=cwd, tries=2)

    # 3. crop
    stage(run, "crop", "Cropping and upscaling the head")
    body_mesh = trimesh.load(body, force="mesh", process=False)
    positions, joints = hero.skinned_vertices(rig.read_bytes())
    bone = hero.regions(body_mesh.vertices, positions, joints)
    picked = hero.head_bones(bone, body_mesh.vertices[:, 1])
    cut = hero.load_picture(cutout)
    full_box = hero.picture_box(cut)
    px, py = hero.project(body_mesh.vertices[np.isin(bone, picked)], hero.frame(body_mesh.vertices, full_box))
    crop = hero.crop_box(px, py, cut.width, cut.height)
    run.say(f"head bones {picked}, crop box {crop} ({crop[2] - crop[0]} px)")
    if crop[2] - crop[0] < 48:
        run.flag(f"the head is only {crop[2] - crop[0]} px wide in the picture: the crop holds little detail")
    crop_raw, crop_up = out / "head_crop_raw.png", out / "head_crop.png"
    if run.step("crop + Real-ESRGAN", crop_up):
        Image.open(picture).convert("RGB").crop(crop).save(crop_raw)
        run.run([PYTHON, UPSCALE, crop_raw, crop_up, "--size", "1024"])
    crop_cutout = out / "head_crop_cutout.png"
    if run.step("crop cut-out", crop_cutout):
        matte.cut_out(Image.open(crop_up).convert("RGB"))[0].save(crop_cutout)
    head_picture, head_cutout, head_crop = crop_up, crop_cutout, crop

    # 4. head
    stage(run, "head", "Building the head on its own (the long step)")
    head_raw = out / "head_raw.glb"
    if run.step("head: Pixal3D on the crop", head_raw):
        run.run([PYTHON, "-u", GENERATE, head_picture, head_raw, "--seed", str(args.seed)])
    head_clean = out / "head_clean.glb"
    if run.step(f"head: clean-up {args.head_faces}", head_clean):
        head_cleanup(run, head_raw, head_clean, args.head_faces)
        run.flag("head clean-up skips Finish's mirror check (front-facing crop); glance at the check sheet")
    head = out / "head.glb"
    if run.step("head: Pixel Match", head):
        finish(run, head_clean, head_picture, head_raw.with_suffix(".svviews"), head, args.head_faces, None)

    head_baked = out / "head_baked.glb"
    if run.step("head: bake normal map from the raw head", head_baked):
        # The body's Finish bakes one; a head without would light differently next to it.
        # Finish skips the bake for a kept mesh, so run its bake step directly.
        import retopo_repaint

        weights = head.with_suffix(".steps") / "3_photo_weights.png"
        fade = (retopo_repaint.normal_fade_mask(weights, 2048, out / "head_photo_fade.png")
                if weights.is_file() else None)
        if not run.run(retopo_repaint.bake_command(head_raw, head, head_baked, 2048,
                                                   retopo_repaint.find_blender(), fade), required=False):
            # The bake refuses when the raw and cleaned heads do not line up within 5%; a hero
            # without a head bump map is still a hero, so carry on and say so.
            shutil.copy(head, head_baked)
            run.flag("the head's bump-map bake refused (see run.log); the head has no bump map")

    # 5. graft
    stage(run, "graft", "Putting the new head on the body")
    grafted = out / "grafted.glb"
    if run.step("graft", grafted):
        head_mesh = trimesh.load(head_baked, force="mesh", process=False)
        crop_pic = hero.load_picture(head_cutout)
        part_body, part_head, report = hero.graft(
            body_mesh, head_mesh, full_box, hero.picture_box(crop_pic), head_crop, crop_pic.width,
            bone, (cut.height, cut.width),
            cut_at=args.cut_at if args.cut_at in (None, "neck") else float(args.cut_at))
        run.say(f"graft {report}")
        # One piece: the head's paint and material always follow the body's own head.
        real = np.flatnonzero(np.isin(bone, picked))
        reach = 0.02 * float(np.ptp(body_mesh.vertices[real, 1]))
        head_idx, body_idx = hero.paired_points(part_head.vertices, body_mesh.vertices[real], reach)
        run.say(f"colour pairs: {len(head_idx)} head points matched to the body's own head")
        head_uv, body_uv = part_head.visual.uv[head_idx], body_mesh.visual.uv[real][body_idx]
        head_mat, body_mat = part_head.visual.material, body_mesh.visual.material
        slots = [("metallicRoughnessTexture", (1, 2))]
        if args.match_paint:
            slots.insert(0, ("baseColorTexture", (0, 1, 2)))
        for slot, channels in slots:
            head_tex, body_tex = getattr(head_mat, slot, None), getattr(body_mat, slot, None)
            if head_tex is None or body_tex is None:
                run.flag(f"no {slot} on the head or body: not matched")
                continue
            matched, how = hero.match_to_body(head_tex, hero.sample_texture(head_tex, head_uv),
                                              hero.sample_texture(body_tex, body_uv), channels)
            setattr(head_mat, slot, matched)
            run.say(f"head {slot} matched to body {how}")
            if how["big_change"]:
                run.flag(f"the head's {slot} was far from the body's ({how}); matched, worth a glance")
        if getattr(head_mat, "metallicRoughnessTexture", None) is not None:
            head_mat.metallicFactor = head_mat.roughnessFactor = 1.0
        if not 0.85 < report["scale_to_real_head"] < 1.15:
            run.flag(f"the new head needed scaling by {report['scale_to_real_head']} to fit: check the fit")
        scene = trimesh.Scene()
        for name, part in (("Body", part_body), ("Head", part_head)):
            # Without normals a glTF renders every triangle flat.
            part.vertex_normals = smooth_normals(part.vertices, part.faces)
            scene.add_geometry(part, node_name=name)
        scene.export(grafted, include_normals=True)
        (out / "graft.json").write_text(json.dumps({"crop": crop, **report}, indent=2))

    # 6. join
    stage(run, "join", "Joining head and body into one model")
    import autorig_api

    joined, final = out / "joined.glb", out / "hero.glb"
    if run.step("join", joined):
        run.run([autorig_api.venv_python(), BLENDER, "join", grafted, joined])
    if run.step("compress textures", final):
        run.run([PYTHON, COMPRESS, joined, final])

    # 7. check
    stage(run, "check", "Rendering the check sheet")
    check = out / "check"
    if run.step("check sheet", check / "head_front.png"):
        run.run([autorig_api.venv_python(), BLENDER, "check", final, check])
        shots = [Image.open(check / f"{name}.png").convert("RGB") for name in
                 ("head_front", "head_three_quarter", "head_side", "head_back", "body_front", "body_three_quarter")]
        sheet = Image.new("RGB", (sum(s.width for s in shots), shots[0].height))
        x = 0
        for shot in shots:
            sheet.paste(shot, (x, 0))
            x += shot.width
        sheet.save(out / "check_sheet.jpg", quality=90)

    sub_runs = []
    for model in (body_raw, head_raw):
        if model.with_suffix(".json").is_file():
            sub_runs.append(json.loads(model.with_suffix(".json").read_text()))
        else:
            run.flag(f"{model.name} has no run record: its licence is not in the provenance")
            sub_runs.append({})
    record = {
        "schema_version": 1, "route": "hero", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": {"picture": str(args.picture), "sha256": sha256_file(picture),
                  "back": str(args.back) if args.back else None},
        "output": {"path": str(final), "sha256": sha256_file(final)},
        "license": sub_runs[0].get("license"),
        "components": [c for s in sub_runs for c in s.get("components", [])] + LICENCES,
        "unsure": run.unsure,
        "parameters": {"body_faces": args.body_faces, "head_faces": args.head_faces, "seed": args.seed,
                       "crop": list(crop), "head_bones": picked},
        "sub_runs": [str(p.with_suffix(".json")) for p in (body_raw, head_raw)],
    }
    final.with_suffix(".provenance.json").write_text(json.dumps(record, indent=2))
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(final, args.result)
        args.result.with_suffix(".provenance.json").write_text(
            json.dumps({**record, "output": {"path": str(args.result), "sha256": sha256_file(args.result)}}, indent=2))
    run.say(f"done: {final} ({final.stat().st_size / 1e6:.1f} MB); check sheet {out / 'check_sheet.jpg'}")
    for note in run.unsure:
        run.say(f"  to check: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
