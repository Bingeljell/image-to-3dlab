"""Tests for the browser prop-sheet job.

Everything a browser sends becomes a Blender or a script argument, so the settings tests
cover what a page can send: bad names, LODs in the wrong order, stray keys. The rest
covers the two things a user sees go wrong: a progress bar that stalls or goes backwards,
and a result list that loses props after a turn re-bakes one of them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "viewer" / "props_api.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


props = _load("props_api", MODULE)


@pytest.fixture(autouse=True)
def gltfpack_found(monkeypatch):
    """The machine running the tests may or may not have gltfpack; these assume it does."""
    monkeypatch.setattr(props, "find_gltfpack", lambda: Path("/bin/gltfpack"))


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_defaults_are_used_when_nothing_is_sent():
    settings = props.normalise_settings({})
    assert settings == props.DEFAULT_SETTINGS
    assert settings["lods"] is not props.DEFAULT_SETTINGS["lods"]  # a copy


def test_the_yaw_threshold_default_is_the_splitters():
    """Repeated here because the splitter imports numpy and the server must not."""
    sys.path.insert(0, str(ROOT / "scripts"))
    splitter = _load("split_props_for_api", ROOT / "scripts" / "blender_split_props.py")
    assert props.YAW_THRESHOLD == splitter.YAW_THRESHOLD


def test_the_lod_defaults_are_the_finishers():
    finisher = props.finisher_module()
    assert props.DEFAULT_SETTINGS["lods"] == list(finisher.DEFAULT_LODS)
    assert props.DEFAULT_SETTINGS["atlas"] == finisher.DEFAULT_ATLAS
    assert props.ATLAS_SIZES == finisher.ATLAS_SIZES
    assert props.FACE_RANGE == finisher.FACE_RANGE


def test_names_arrive_one_per_line_from_the_textarea():
    settings = props.normalise_settings({"names": "barrel\n  crate \n\nchest\n"})
    assert settings["names"] == ["barrel", "crate", "chest"]


@pytest.mark.parametrize("names", [
    "barrel\nbarrel", "iron chest", "../etc", "a" * 41, ["x"] * 65, 12,
    "Barrel\nbarrel",                      # one file on a case-insensitive disk
    "-lid", "barrel\n--no-straighten", "-h",   # read as flags after --names
])
def test_names_that_cannot_be_file_names_or_repeat_are_refused(names):
    with pytest.raises(ValueError) as caught:
        props.normalise_settings({"names": names})
    assert "names" in str(caught.value)


def test_lods_arrive_as_text_or_a_list():
    assert props.normalise_settings({"lods": "8000,3000"})["lods"] == [8000, 3000]
    assert props.normalise_settings({"lods": [4000]})["lods"] == [4000]


@pytest.mark.parametrize("lods", [
    "", "5000,500", "5000,250000", "1000,2500", "3000,3000", "9000,8000,7000,6000,5000", "lots",
])
def test_lods_the_finisher_would_refuse_are_refused_first(lods):
    with pytest.raises(ValueError) as caught:
        props.normalise_settings({"lods": lods})
    assert "lods" in str(caught.value)


@pytest.mark.parametrize(("key", "value"), [
    ("atlas", 3000), ("atlas", "big"), ("metallic", 1.5), ("roughness", -0.1),
    ("ior", 0.5), ("yaw_threshold", 1.0), ("metallic", True),
    ("metallic", float("nan")), ("turns", {"chest": float("inf")}),
    ("lods", [float("inf")]), ("lods", "nan"),
])
def test_out_of_range_values_are_refused_with_their_key(key, value):
    with pytest.raises(ValueError) as caught:
        props.normalise_settings({key: value})
    assert key in str(caught.value)


def test_unknown_keys_are_dropped_rather_than_forwarded():
    settings = props.normalise_settings({"lodz": "1", "metallic": 0.4})
    assert "lodz" not in settings
    assert settings["metallic"] == pytest.approx(0.4)


@pytest.mark.parametrize(("degrees", "wrapped"), [
    (90, 90), (270, -90), (360, 0), (-180, 180), (180, 180), (450, 90),
])
def test_turns_wrap_so_four_quarter_turns_are_none(degrees, wrapped):
    assert props.wrap_degrees(degrees) == wrapped


def _job(tmp_path: Path, **settings):
    job = props.PropsJob("0" * 32, tmp_path)
    job.settings = props.normalise_settings(settings)
    return job


def test_the_split_command_carries_names_turns_and_threshold(tmp_path):
    job = _job(tmp_path, names="barrel\nchest", turns={"chest": 90})
    command = props.split_command(job, Path("/bin/blender"))
    assert command[:8] == ["/bin/blender", "-b", "--factory-startup",
                           "--python-exit-code", "1", "-P", str(props.SPLITTER), "--"]
    assert command[8:10] == [str(job.source_glb), str(job.split_dir)]
    assert command[command.index("--names") + 1:][:2] == ["barrel", "chest"]
    assert command[command.index("--turn") + 1] == "chest=90"
    assert command[command.index("--yaw-threshold") + 1] == str(props.YAW_THRESHOLD)


def test_the_split_command_numbers_props_when_no_names_are_given(tmp_path):
    assert "--names" not in props.split_command(_job(tmp_path), Path("/bin/blender"))


def test_the_split_command_is_one_the_splitter_accepts(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    splitter = _load("split_props_for_cmd", ROOT / "scripts" / "blender_split_props.py")
    job = _job(tmp_path, names="barrel\nchest", turns={"chest": -90})
    args = splitter.parse_args(props.split_command(job, Path("/bin/blender")))
    assert args.names == ["barrel", "chest"]
    assert args.turn == {"chest": -90.0}


def test_the_finish_command_is_one_the_finisher_accepts(tmp_path):
    job = _job(tmp_path, lods="6000,2000", atlas=2048)
    command = props.finish_command(job, Path("/bin/blender"), Path("/bin/gltfpack"))
    parsed = props.finisher_module().build_parser().parse_args(command[3:])
    assert parsed.source == job.split_dir
    assert parsed.out_dir == job.finished_dir
    assert parsed.lods == "6000,2000"
    assert parsed.atlas == 2048
    assert parsed.gltfpack == Path("/bin/gltfpack")
    assert parsed.blender == Path("/bin/blender")
    assert not parsed.no_compress


def test_a_turn_bakes_only_its_prop(tmp_path):
    job = _job(tmp_path)
    job.only = "chest"
    command = props.finish_command(job, Path("/bin/blender"), None)
    assert command[3] == str(tmp_path / "split" / "chest.glb")


def test_without_gltfpack_or_when_asked_the_lods_stay_uncompressed(tmp_path):
    missing = props.finish_command(_job(tmp_path), Path("/bin/blender"), None)
    declined = props.finish_command(_job(tmp_path, compress=False), Path("/bin/blender"),
                                    Path("/bin/gltfpack"))
    for command in (missing, declined):
        assert "--no-compress" in command
        assert "--gltfpack" not in command


def test_rows_follow_the_order_the_finisher_bakes_in(tmp_path):
    """Not reading order and not a plain sort: `a-b.glb` sorts before `a.glb`."""
    for name in ("tree_stump", "anvil", "a", "a-b"):
        (tmp_path / f"{name}.glb").write_bytes(b"x")
    (tmp_path / "props.json").write_text("{}")
    order = props.bake_order(tmp_path, ["tree_stump", "anvil", "a", "a-b"])
    assert order == [p.stem for p in props.finisher_module().collect_props(tmp_path)]
    assert order == ["a-b", "a", "anvil", "tree_stump"]


def test_a_turn_bakes_its_one_prop_whatever_else_is_there(tmp_path):
    for name in ("anvil", "chest"):
        (tmp_path / f"{name}.glb").write_bytes(b"x")
    assert props.bake_order(tmp_path, ["chest"]) == ["chest"]


def test_prop_rows_cannot_collide_with_the_panels_own_phases():
    meta = props.stage_meta(["done", "split"])
    assert meta["stages"] == ["split", "prop:done", "prop:split"]
    assert meta["stage_labels"]["prop:done"] == "done"


# What `finish_props.py` prints, in order, for two props and two LODs: its progress
# markers, made by its own helper, among the lines it prints for people.
_line = props.finisher_module().progress_line
FINISH_LINES = [
    _line("barrel", lod=0),
    "[barrel LOD0] /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup ...",
    "[barrel LOD0 gltfpack] /usr/bin/gltfpack -i a.glb -o b.glb ...",
    _line("barrel", lod=1),
    "[barrel LOD1 gltfpack] /usr/bin/gltfpack -i a.glb -o b.glb ...",
    _line("barrel", done="barrel: LOD0 425 KB, LOD1 387 KB"),
    _line("crate", lod=0),
    _line("crate", lod=1),
    _line("crate", done="crate: LOD0 239 KB, LOD1 227 KB"),
    "finished 2 props in 30s -> out",
]
MARKERS = [line for line in FINISH_LINES if line.startswith("FINISH::")]


def test_progress_runs_forward_through_a_real_run():
    clock = Clock()
    progress = props.PropsProgress(clock=clock)
    seen = [progress.begin_split()["overall_pct"]]
    clock.now = 8
    seen.append(progress.split_line("SPLIT:: wrote 2 props to out")["overall_pct"])
    start = progress.begin_props(["barrel", "crate"], lods=2)
    assert start["stages"] == ["split", "prop:barrel", "prop:crate"]
    seen.append(start["overall_pct"])
    for line in FINISH_LINES:
        clock.now += 5
        event = progress.finish_line(line)
        if event:
            seen.append(event["overall_pct"])
    assert seen == sorted(seen)
    assert seen[0] == 0
    assert seen[-1] == 100


def test_each_prop_row_counts_its_lods_and_ticks_when_done():
    progress = props.PropsProgress(clock=Clock())
    progress.begin_props(["barrel", "crate"], lods=2)
    events = [progress.finish_line(line) for line in MARKERS[:3]]
    assert events[0]["phase"] == "prop:barrel"
    assert events[0]["message"] == "barrel: baking LOD0"
    assert (events[0]["step"], events[0]["total"]) == (0, 2)
    assert (events[1]["step"], events[1]["total"]) == (1, 2)
    assert events[2]["stage_pct"] == 100
    assert events[2]["message"] == "barrel: LOD0 425 KB, LOD1 387 KB"
    split = 100 * props.split_share(2)
    assert events[2]["overall_pct"] == pytest.approx(split + (100 - split) / 2, abs=0.1)


def test_the_split_takes_more_of_the_bar_when_there_is_less_to_bake():
    assert props.split_share(1) > 0.2        # a turn: the split is a fifth of it
    assert props.split_share(9) < 0.05       # a full sheet: the bakes are the run


def test_a_one_prop_turn_does_not_claim_minutes_left_after_its_split():
    """The reported "~2 min left" for a 20-second turn."""
    clock = Clock()
    progress = props.PropsProgress(clock=clock)
    progress.begin_split()
    clock.now = props.SPLIT_SECONDS
    event = progress.begin_props(["barrel"], lods=3)
    assert event["total_eta_seconds"] < 30


def test_only_the_finishers_markers_for_this_runs_props_are_counted():
    """The lines around the markers are for people, and free to change."""
    progress = props.PropsProgress(clock=Clock())
    progress.begin_props(["barrel"], lods=3)
    for line in FINISH_LINES:
        if not line.startswith("FINISH::"):
            assert progress.finish_line(line) is None, line
    assert progress.finish_line(_line("stranger", lod=0)) is None
    assert progress.finish_line("FINISH::{not json") is None
    assert progress.finish_line('FINISH::["barrel"]') is None
    assert progress.finish_line('FINISH::{"prop": "barrel"}') is None
    assert progress.done == {"barrel": 0}


def test_split_output_other_than_its_own_markers_is_ignored():
    progress = props.PropsProgress(clock=Clock())
    assert progress.split_line("Blender 4.5.0 (hash abc)") is None
    assert progress.split_line("SPLIT:: 27 loose parts")["message"] == "27 loose parts"


def _write_run(root: Path, name: str = "sheet__props__20260925-120000",
               props_names=("barrel", "chest"), baked=2, web=True) -> Path:
    """A run with two LODs configured and `baked` of them on disk for every prop."""
    directory = root / name
    (directory / "split").mkdir(parents=True)
    (directory / "source.glb").write_bytes(b"glb")
    (directory / "settings.json").write_text(json.dumps(
        props.normalise_settings({"lods": [5000, 1000]})))
    (directory / "split" / "props.json").write_text(json.dumps({"props": [
        {"name": n, "faces": 1000, "size": [0.1, 0.1, 0.2], "yaw_degrees": 44.0 if n == "chest" else 0.0,
         "yaw_tie": n == "chest"} for n in props_names]}))
    for n in props_names:
        (directory / "split" / f"{n}.glb").write_bytes(b"prop")
        for index in range(baked):
            lod = directory / "finished" / n / f"{n}_LOD{index}.glb"
            lod.parent.mkdir(parents=True, exist_ok=True)
            lod.write_bytes(b"x" * 100)
            if web:
                lod.with_name(f"{n}_LOD{index}.web.glb").write_bytes(b"x" * 10)
    return directory


def test_a_run_is_described_from_disk_in_reading_order(tmp_path, monkeypatch):
    monkeypatch.setattr(props, "served_url", lambda p: "/" + p.relative_to(tmp_path).as_posix())
    run = props.describe_run(_write_run(tmp_path))
    assert [p["name"] for p in run["props"]] == ["barrel", "chest"]
    assert run["finished"]
    chest = run["props"][1]
    assert chest["yaw_tie"]
    assert [lod["bytes"] for lod in chest["lods"]] == [100, 100]
    assert chest["lods"][0]["web_bytes"] == 10
    # The plain LOD0: the viewer has no meshopt decoder for the .web one.
    assert chest["preview_url"].endswith("chest/chest_LOD0.glb")


def test_each_lod_reports_the_triangles_its_file_holds(tmp_path):
    import compress_glb_textures as glb_tools

    directory = _write_run(tmp_path, props_names=("chest",))
    lod = directory / "finished" / "chest" / "chest_LOD0.glb"
    lod.write_bytes(glb_tools.build_glb({
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [{"count": 10, "type": "VEC3", "componentType": 5126},
                      {"count": 4998 * 3, "type": "SCALAR", "componentType": 5125}],
    }, b"\0" * 4096))
    lods = props.describe_run(directory)["props"][0]["lods"]
    assert lods[0]["triangles"] == 4998
    assert lods[1]["triangles"] is None        # not a GLB: the page shows the target


def test_a_lod_baked_again_is_read_again(tmp_path):
    """The count is remembered per file, and a re-bake (a turn) is a new file."""
    import compress_glb_textures as glb_tools

    def lod(triangles: int, padding: int) -> bytes:
        return glb_tools.build_glb({
            "asset": {"version": "2.0"},
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "accessors": [{"count": 3, "type": "VEC3", "componentType": 5126},
                          {"count": triangles * 3, "type": "SCALAR", "componentType": 5125}],
        }, b"\0" * padding)

    path = tmp_path / "chest_LOD0.glb"
    path.write_bytes(lod(4998, 64))
    assert props.lod_triangles(path) == 4998
    assert props.lod_triangles(path) == 4998
    path.write_bytes(lod(2500, 128))
    assert props.lod_triangles(path) == 2500
    path.unlink()
    assert props.lod_triangles(path) is None


def test_a_turn_rewriting_the_finish_record_does_not_hide_other_props(tmp_path):
    directory = _write_run(tmp_path)
    (directory / "finished" / "finish_props.json").write_text(json.dumps(
        {"props": [{"name": "chest"}]}))
    assert [p["name"] for p in props.describe_run(directory)["props"]] == ["barrel", "chest"]


def test_a_run_without_lods_yet_previews_the_split_prop(tmp_path, monkeypatch):
    monkeypatch.setattr(props, "served_url", lambda p: "/" + p.relative_to(tmp_path).as_posix())
    directory = _write_run(tmp_path, baked=0)
    run = props.describe_run(directory)
    assert not run["finished"]
    assert run["props"][0]["preview_url"].endswith("split/barrel.glb")


def test_runs_are_listed_and_strangers_ignored(tmp_path):
    _write_run(tmp_path)
    (tmp_path / "not-a-run").mkdir()
    runs = props.list_runs(tmp_path)
    assert [r["directory"] for r in runs] == ["sheet__props__20260925-120000"]


@pytest.mark.parametrize("name", ["../finish", "sheet__props__x", "", "sheet__finish__20260925-120000"])
def test_a_run_name_that_create_would_not_make_is_refused(tmp_path, name):
    with pytest.raises(RuntimeError):
        props.run_directory(tmp_path, name)


def test_creating_a_run_writes_its_source_and_settings(tmp_path):
    manager = props.PropsJobManager(tmp_path)
    job = manager.create("Grid Sheet.glb", b"glb", {"names": "barrel"})
    assert job.directory.name.startswith("Grid-Sheet__props__")
    assert job.source_glb.read_bytes() == b"glb"
    assert json.loads(job.settings_path.read_text())["names"] == ["barrel"]
    with pytest.raises(RuntimeError):
        manager.create("other.glb", b"glb", {})


def test_bad_settings_leave_no_directory_behind(tmp_path):
    with pytest.raises(ValueError):
        props.PropsJobManager(tmp_path).create("sheet.glb", b"glb", {"lods": "1,2"})
    assert not any(tmp_path.iterdir())


def _finish_turn(job, monkeypatch, *, fail_at=None, finish_record=None):
    """Run a turn job with both scripts replaced by stand-ins that write what they would.

    The split re-exports every prop and rewrites props.json with the turn; the bake
    writes the turned prop's LODs, and `finish_record` as its finish_props.json. `fail_at` makes
    one step stop after writing its files, as a cancel or a crash part-way through does.
    """
    live = job.directory / "split"

    def step(job, label, command, feed):
        if label == "split":
            job.split_dir.mkdir(parents=True)
            record = json.loads((live / "props.json").read_text())
            for entry in record["props"]:
                if entry["name"] == job.only:
                    entry["extra_turn_degrees"] = job.settings["turns"].get(job.only, 0.0)
                (job.split_dir / f"{entry['name']}.glb").write_bytes(b"turned")
            (job.split_dir / "props.json").write_text(json.dumps(record))
        else:
            for index in range(len(job.settings["lods"])):
                lod = job.finished_dir / job.only / f"{job.only}_LOD{index}.glb"
                lod.parent.mkdir(parents=True, exist_ok=True)
                lod.write_bytes(b"turned")
            if finish_record is not None:
                (job.finished_dir / "finish_props.json").write_text(json.dumps(finish_record))
        if label == fail_at:
            job.status = "cancelled"
            return False
        return True

    monkeypatch.setattr(props, "blender_executable", lambda: Path("/bin/blender"))
    monkeypatch.setattr(props, "_run_step", step)
    props.run_job(job, props.PropsJobManager(job.directory.parent))
    return job


def _snapshot(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "run.log"}


def test_a_finished_turn_swaps_in_the_new_split_and_lods(tmp_path, monkeypatch):
    directory = _write_run(tmp_path)
    job = _finish_turn(props.PropsJobManager(tmp_path).turn(directory.name, "chest", 90),
                       monkeypatch)
    assert job.status == "done"
    assert (directory / "split" / "chest.glb").read_bytes() == b"turned"
    assert (directory / "finished" / "chest" / "chest_LOD0.glb").read_bytes() == b"turned"
    assert (directory / "finished" / "barrel" / "barrel_LOD0.glb").read_bytes() == b"x" * 100
    assert not (directory / "turn").exists()
    chest = next(p for p in props.describe_run(directory)["props"] if p["name"] == "chest")
    assert chest["extra_turn_degrees"] == 90.0


@pytest.mark.parametrize("fail_at", ["split", "LODs"])
def test_a_cancelled_or_failed_turn_leaves_the_run_exactly_as_it_was(tmp_path, monkeypatch, fail_at):
    """Cancelled during the bake, the chest had kept a turned split and lost its LODs."""
    directory = _write_run(tmp_path)
    before = _snapshot(directory)
    job = _finish_turn(props.PropsJobManager(tmp_path).turn(directory.name, "chest", 90),
                       monkeypatch, fail_at=fail_at)
    assert job.status == "cancelled"
    assert _snapshot(directory) == before
    assert not (directory / "turn").exists()


def test_a_prop_called_split_can_be_turned(tmp_path, monkeypatch):
    """The old split and the old prop both went to turn/old/, so this one collided."""
    directory = _write_run(tmp_path, props_names=("barrel", "split"))
    job = _finish_turn(props.PropsJobManager(tmp_path).turn(directory.name, "split", 90),
                       monkeypatch)
    assert job.status == "done", job.events[-1]
    assert (directory / "split" / "split.glb").read_bytes() == b"turned"
    assert (directory / "finished" / "split" / "split_LOD0.glb").read_bytes() == b"turned"
    assert json.loads((directory / "settings.json").read_text())["turns"] == {"split": 90.0}


def test_a_swap_that_fails_half_way_puts_the_run_back(tmp_path, monkeypatch):
    directory = _write_run(tmp_path)
    before = _snapshot(directory)
    real = Path.replace
    failed = []

    def failing(self, target):
        # The last of the moves fails once; putting things back must still work.
        if Path(target) == directory / "finished" / "chest" and not failed:
            failed.append(self)
            raise OSError("disk full")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", failing)
    job = _finish_turn(props.PropsJobManager(tmp_path).turn(directory.name, "chest", 90),
                       monkeypatch)
    assert job.status == "error"
    assert _snapshot(directory) == before
    assert json.loads((directory / "settings.json").read_text())["turns"] == {}


def test_a_turn_updates_its_prop_in_the_finish_record(tmp_path, monkeypatch):
    """The turn's own record went with its staging folder, leaving the old numbers."""
    directory = _write_run(tmp_path)
    (directory / "finished" / "finish_props.json").write_text(json.dumps({"lods": [5000, 1000], "props": [
        {"name": "barrel", "lods": [{"triangles": 5000}]},
        {"name": "chest", "lods": [{"triangles": 4999, "glb": "stale"}]},
    ]}))
    # Paths as the finisher writes them, relative to the folder it baked into.
    entry = {"name": "chest", "source": "../split/chest.glb",
             "lods": [{"triangles": 5000, "glb": "chest/chest_LOD0.glb"}]}
    job = _finish_turn(props.PropsJobManager(tmp_path).turn(directory.name, "chest", 90),
                       monkeypatch, finish_record={"props": [entry]})
    assert job.status == "done"
    finished = directory / "finished"
    saved = json.loads((finished / "finish_props.json").read_text())
    assert saved["lods"] == [5000, 1000]
    assert [e["name"] for e in saved["props"]] == ["barrel", "chest"]
    chest = saved["props"][1]
    assert chest == entry
    # And they still point at the files once the turn is swapped into the run.
    assert (finished / chest["source"]).resolve() == (directory / "split" / "chest.glb").resolve()
    assert (finished / chest["lods"][0]["glb"]).read_bytes() == b"turned"


def test_a_turn_that_would_lose_the_web_files_is_refused(tmp_path, monkeypatch):
    """Baked with gltfpack, turned without it: the prop's .web.glb files went with the swap."""
    directory = _write_run(tmp_path)
    before = _snapshot(directory)
    monkeypatch.setattr(props, "find_gltfpack", lambda: None)
    manager = props.PropsJobManager(tmp_path)
    with pytest.raises(RuntimeError, match="gltfpack"):
        manager.turn(directory.name, "chest", 90)
    assert manager.active is None
    assert _snapshot(directory) == before


def test_without_gltfpack_a_prop_baked_without_it_can_still_be_turned(tmp_path, monkeypatch):
    directory = _write_run(tmp_path, web=False)
    monkeypatch.setattr(props, "find_gltfpack", lambda: None)
    assert props.PropsJobManager(tmp_path).turn(directory.name, "chest", 90).only == "chest"


def test_a_step_that_fails_ends_the_job_with_its_code_and_log(tmp_path, monkeypatch):
    manager = props.PropsJobManager(tmp_path)
    job = manager.create("sheet.glb", b"glb", {})
    monkeypatch.setattr(props, "blender_executable", lambda: Path("/bin/blender"))
    monkeypatch.setattr(props, "split_command", lambda job, blender: [
        sys.executable, "-c", "print('not a GLB'); raise SystemExit(2)"])
    props.run_job(job, manager)
    assert job.status == "error"
    assert job.events[-1]["message"] == "split exited with code 2"
    assert "not a GLB" in job.events[-1]["log_tail"]
    assert manager.active is None


def test_without_blender_the_tab_and_the_job_say_how_to_get_it(tmp_path, monkeypatch):
    from image_to_3dlab.blender import missing_help

    monkeypatch.setattr(props, "blender_executable", lambda: None)
    assert props.tools_payload()["blender_problem"] == missing_help()
    manager = props.PropsJobManager(tmp_path)
    job = manager.create("sheet.glb", b"glb", {})
    props.run_job(job, manager)
    assert job.events[-1]["message"] == missing_help()
    monkeypatch.setattr(props, "blender_executable", lambda: Path("/bin/blender"))
    assert props.tools_payload()["blender_problem"] is None


def test_a_script_that_exits_ends_the_job_with_an_error(tmp_path, monkeypatch):
    """finish_props raises SystemExit, which `except Exception` let end the thread silently."""
    manager = props.PropsJobManager(tmp_path)
    job = manager.create("sheet.glb", b"glb", {"names": "barrel"})

    def split_without_glbs(job, label, command, feed):
        job.split_dir.mkdir(parents=True)
        (job.split_dir / "props.json").write_text(json.dumps({"props": [{"name": "barrel"}]}))
        return True

    monkeypatch.setattr(props, "blender_executable", lambda: Path("/bin/blender"))
    monkeypatch.setattr(props, "_run_step", split_without_glbs)
    props.run_job(job, manager)
    assert job.status == "error"
    assert job.events[-1]["phase"] == "error"
    assert "no GLBs" in job.events[-1]["message"]
    assert manager.active is None


def test_turning_twice_adds_up_once_each_turn_is_baked(tmp_path, monkeypatch):
    directory = _write_run(tmp_path)
    manager = props.PropsJobManager(tmp_path)
    first = manager.turn(directory.name, "chest", 90)
    assert first.only == "chest"
    assert first.settings["turns"] == {"chest": 90.0}
    _finish_turn(first, monkeypatch)
    assert first.status == "done"
    second = manager.turn(directory.name, "chest", 90)
    assert second.settings["turns"] == {"chest": 180.0}
    _finish_turn(second, monkeypatch)
    saved = json.loads((directory / "settings.json").read_text())
    assert saved["turns"] == {"chest": 180.0}


def test_a_cancelled_turn_leaves_no_turn_behind(tmp_path, monkeypatch):
    """The barrel that came out backwards: a cancelled turn, then one more click."""
    directory = _write_run(tmp_path)
    manager = props.PropsJobManager(tmp_path)
    _finish_turn(manager.turn(directory.name, "barrel", 90), monkeypatch, fail_at="LODs")
    assert json.loads((directory / "settings.json").read_text())["turns"] == {}
    again = manager.turn(directory.name, "barrel", 90)
    assert again.settings["turns"] == {"barrel": 90.0}


def test_four_quarter_turns_forget_the_turn(tmp_path, monkeypatch):
    directory = _write_run(tmp_path)
    manager = props.PropsJobManager(tmp_path)
    for _ in range(4):
        job = manager.turn(directory.name, "chest", 90)
        _finish_turn(job, monkeypatch)
    assert job.settings["turns"] == {}
    assert json.loads((directory / "settings.json").read_text())["turns"] == {}


@pytest.mark.parametrize(("prop", "degrees", "error"), [
    ("anvil", 90, RuntimeError), ("chest", 400, ValueError), ("chest", "lots", ValueError),
])
def test_a_turn_for_a_missing_prop_or_a_wild_angle_is_refused(tmp_path, prop, degrees, error):
    directory = _write_run(tmp_path)
    with pytest.raises(error):
        props.PropsJobManager(tmp_path).turn(directory.name, prop, degrees)


def test_generated_models_are_found_and_only_those_can_be_named(tmp_path):
    for relative in ("owl__pixal3d__1/owl__pixal3d__1.glb",
                     "research_only/grid__pixal3d__2/grid__pixal3d__2.glb",
                     "finish/x__finish__1/result.glb",
                     "props/sheet__props__1/finished/barrel/barrel_LOD0.glb"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"glb")
    found = sorted(m["path"] for m in props.generated_models(tmp_path))
    assert found == ["owl__pixal3d__1/owl__pixal3d__1.glb",
                     "research_only/grid__pixal3d__2/grid__pixal3d__2.glb"]
    assert props.generated_model(tmp_path, found[0]) == tmp_path / found[0]
    for stranger in ("../secret.glb", "finish/x__finish__1/result.glb", "/etc/passwd"):
        with pytest.raises(RuntimeError):
            props.generated_model(tmp_path, stranger)


def test_a_cancel_between_steps_stops_before_the_next_one_starts(tmp_path, monkeypatch):
    job = _job(tmp_path)
    job.cancel_requested = True
    monkeypatch.setattr(props.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("a cancelled job started a process"))
    assert props._run_step(job, "LODs", ["true"], lambda line: None) is False
    assert job.status == "cancelled"


def test_a_running_step_records_this_server_as_its_owner(tmp_path):
    """So a second viewer starting up leaves the running bake alone instead of killing it."""
    import os

    from image_to_3dlab import processes

    job = _job(tmp_path)
    seen = []

    def feed(line):
        seen.append(processes.parse_pid_record(job.directory.joinpath("pid").read_text()))

    assert props._run_step(job, "LODs", [sys.executable, "-c", "print('working')"], feed)
    assert seen[0] == (job.process.pid, os.getpid())


def test_a_job_cancelled_while_queued_never_runs(tmp_path, monkeypatch):
    job = props.PropsJobManager(tmp_path).create("sheet.glb", b"glb", {})
    job.cancel_requested = True
    job.status = "cancelling"
    monkeypatch.setattr(props, "blender_executable", lambda: Path("/bin/blender"))
    monkeypatch.setattr(props, "_run_step", lambda *a: pytest.fail("a step ran"))
    props.run_job(job, props.PropsJobManager(tmp_path))
    assert job.status == "cancelled"


def test_the_finisher_is_imported_once(monkeypatch):
    """Each import runs its sys.path.insert; per request, the server's path kept growing."""
    first = props.finisher_module()
    length = len(sys.path)
    for _ in range(5):
        assert props.finisher_module() is first
    assert len(sys.path) == length


def test_an_odd_props_record_does_not_break_the_run_list(tmp_path):
    directory = _write_run(tmp_path)
    (directory / "split" / "props.json").write_text(json.dumps(
        {"props": ["not a dict", {"name": "-flag"}, {"name": "barrel", "faces": 10}]}))
    assert [p["name"] for p in props.describe_run(directory)["props"]] == ["barrel"]


def test_a_generated_model_without_its_record_is_not_called_an_upload(tmp_path):
    """With Debug off the viewer keeps only the GLB, so the record is often gone."""
    job = props.PropsJobManager(tmp_path).create("sheet.glb", b"glb", {}, None, generated=True)
    licence = props.licence_of(json.loads(job.provenance_path.read_text()))
    assert licence["source"] == "generated"
    assert licence["name"] is None


def test_a_run_from_a_generated_model_carries_its_licence(tmp_path):
    record = {"license": {"name": "Qwen Research", "url": "https://example.test/licence"},
              "output": {"classification": "research-only"}}
    job = props.PropsJobManager(tmp_path).create("sheet.glb", b"glb", {}, record)
    saved = json.loads(job.provenance_path.read_text())
    assert saved["input"]["source"] == "generated"
    assert saved["input"]["sha256"] == __import__("hashlib").sha256(b"glb").hexdigest()
    assert saved["license"]["name"] == "Qwen Research"
    licence = props.describe_run(job.directory)["licence"]
    assert licence == {"source": "generated", "name": "Qwen Research",
                       "url": "https://example.test/licence", "classification": "research-only"}


def test_an_uploaded_model_says_its_licence_is_unknown(tmp_path):
    job = props.PropsJobManager(tmp_path).create("sheet.glb", b"glb", {})
    licence = props.describe_run(job.directory)["licence"]
    assert licence["source"] == "uploaded"
    assert licence["name"] is None


def test_the_generate_tabs_record_is_found_beside_its_glb(tmp_path):
    glb = tmp_path / "owl" / "owl.glb"
    glb.parent.mkdir()
    glb.write_bytes(b"glb")
    assert props.source_record_for(glb) is None
    glb.with_suffix(".json").write_text(json.dumps({"license": {"name": "X"}}))
    assert props.source_record_for(glb)["license"]["name"] == "X"


def test_the_provenance_sidecar_wins_over_the_run_manifest(tmp_path):
    """The sidecar is what survives Debug off; the manifest is deleted with the rest."""
    glb = tmp_path / "owl" / "owl.glb"
    glb.parent.mkdir()
    glb.write_bytes(b"glb")
    glb.with_suffix(".json").write_text(json.dumps({"license": {"name": "manifest"}}))
    glb.with_suffix(".provenance.json").write_text(json.dumps({"license": {"name": "sidecar"}}))
    assert props.source_record_for(glb)["license"]["name"] == "sidecar"
    glb.with_suffix(".json").unlink()
    assert props.source_record_for(glb)["license"]["name"] == "sidecar"


def test_job_ids_are_validated_before_lookup():
    assert props.PropsJobManager().get("../../etc") is None


def test_tools_say_whether_setup_can_install_gltfpack(monkeypatch):
    monkeypatch.setattr(props, "find_gltfpack", lambda: None)
    monkeypatch.setattr(props.gltfpack_bootstrap(), "can_install", lambda: True)
    assert props.tools_payload()["gltfpack_installable"] is True
    monkeypatch.setattr(props.gltfpack_bootstrap(), "can_install", lambda: False)
    assert props.tools_payload()["gltfpack_installable"] is False
    monkeypatch.setattr(props, "find_gltfpack", lambda: Path("/bin/gltfpack"))
    monkeypatch.setattr(props.gltfpack_bootstrap(), "can_install", lambda: True)
    assert props.tools_payload()["gltfpack_installable"] is False  # already there


def test_gltfpack_install_runs_the_bootstrap_with_yes():
    command = props.gltfpack_install_command()
    assert command[-2].endswith("bootstrap_gltfpack.py") and command[-1] == "--yes"


def test_the_server_can_load_the_gltfpack_bootstrap_in_a_fresh_process():
    # In this suite another test has already imported it, which hid a crash the live
    # server hit on the first visit to Setup (a dataclass needs its module registered).
    import subprocess

    code = ("import sys; sys.path.insert(0, 'viewer'); import props_api; "
            "print(props_api.tools_payload()['gltfpack_installable'] in (True, False))")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("True")
