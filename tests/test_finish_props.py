"""Tests for finishing prop-sheet props into compressed LODs.

The run is several Blender bakes per prop, so the parts that decide what gets baked are
checked here: the LOD face counts, which files are picked up, where each LOD is written,
and that gltfpack only compresses. A gltfpack call that also simplified would undo the
reason every LOD is re-baked from the original.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import compress_glb_textures as glb_tools
import pytest

from image_to_3dlab.blender import missing_help

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "finish_props.py"


def _load():
    spec = importlib.util.spec_from_file_location("finish_props", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finish = _load()


def _load_retopo():
    script = SCRIPT.parent / "blender_retopo_bake.py"
    spec = importlib.util.spec_from_file_location("retopo_for_finish", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retopo = _load_retopo()


def test_the_default_lods_are_most_detailed_first():
    assert finish.parse_lods("5000,2500,1000") == [5000, 2500, 1000]
    assert finish.parse_lods(",".join(map(str, finish.DEFAULT_LODS))) == list(finish.DEFAULT_LODS)


def test_one_lod_is_enough():
    assert finish.parse_lods("8000") == [8000]


@pytest.mark.parametrize("text", ["", "abc", "5000,500", "5000,250000", "1000,2500", "3000,3000"])
def test_bad_lods_are_refused(text):
    with pytest.raises(SystemExit):
        finish.parse_lods(text)


def test_a_directory_yields_its_glbs_in_order(tmp_path):
    for name in ("crate.glb", "barrel.glb", "props.json", "props.blend"):
        (tmp_path / name).write_bytes(b"x")
    assert finish.collect_props(tmp_path) == [tmp_path / "barrel.glb", tmp_path / "crate.glb"]


def test_a_single_glb_is_accepted(tmp_path):
    prop = tmp_path / "chest.glb"
    prop.write_bytes(b"x")
    assert finish.collect_props(prop) == [prop]


def test_nothing_to_finish_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        finish.collect_props(tmp_path)
    with pytest.raises(SystemExit):
        finish.collect_props(tmp_path / "missing.glb")


def test_each_lod_has_a_plain_and_a_web_file_per_prop():
    out = Path("out")
    assert finish.lod_path(out, "chest", 0) == out / "chest" / "chest_LOD0.glb"
    assert finish.lod_path(out, "chest", 2, web=True) == out / "chest" / "chest_LOD2.web.glb"


def test_gltfpack_only_compresses(tmp_path):
    command = finish.gltfpack_command(Path("/bin/gltfpack"), Path("a.glb"), Path("b.glb"))
    assert command[:5] == ["/bin/gltfpack", "-i", "a.glb", "-o", "b.glb"]
    assert "-cc" in command and "-tw" in command
    assert not any(flag.startswith("-s") for flag in command)


def test_gltfpack_keeps_the_named_node():
    command = finish.gltfpack_command(Path("/bin/gltfpack"), Path("a.glb"), Path("b.glb"))
    assert "-kn" in command


def test_an_explicit_gltfpack_wins_and_must_exist(tmp_path):
    binary = tmp_path / "gltfpack"
    binary.write_bytes(b"")
    assert finish.find_gltfpack(binary, which=lambda _: "/usr/bin/gltfpack") == binary
    with pytest.raises(SystemExit):
        finish.find_gltfpack(tmp_path / "nope")


def test_gltfpack_is_found_on_path_then_in_vendor(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, "REPO", tmp_path)
    assert finish.find_gltfpack(which=lambda _: "/usr/bin/gltfpack") == Path("/usr/bin/gltfpack")
    assert finish.find_gltfpack(which=lambda _: None) is None
    vendored = tmp_path / "vendor" / "gltfpack" / "gltfpack"
    vendored.parent.mkdir(parents=True)
    vendored.write_bytes(b"")
    assert finish.find_gltfpack(which=lambda _: None) == vendored


def test_the_detail_bake_report_is_read_from_its_log():
    log = ("Blender 4.5\nsome noise\n" + finish.DETAIL_MARKER + json.dumps({
        "source": "a.glb", "metallic_roughness": "transferred", "spread": 1.0052,
        "normal_flipped_fraction": 0.001, "zero_tangents_repaired": 2}) + "\nBlender quit\n")
    assert finish.detail_record(log) == {
        "metallic_roughness": "transferred", "spread": 1.0052,
        "normal_flipped_fraction": 0.001, "zero_tangents_repaired": 2}
    assert finish.detail_record("no report here") == {}
    assert finish.detail_record(finish.DETAIL_MARKER + "{not json") == {}


def test_the_limits_match_what_the_bake_accepts():
    """The two scripts check the same numbers; if the bake's change, these must too."""
    low, high = finish.FACE_RANGE
    for faces in (low, high):
        retopo.parse_args(["--", "a.glb", "b.glb", str(faces)])
    for faces in (low - 1, high + 1):
        with pytest.raises(SystemExit):
            retopo.parse_args(["--", "a.glb", "b.glb", str(faces)])
    # Every size a LOD can reach, from each --atlas choice down to the floor.
    for size in {finish.lod_atlas(a, i) for a in finish.ATLAS_SIZES for i in range(6)}:
        retopo.parse_args(["--", "a.glb", "b.glb", "5000", str(size)])
    with pytest.raises(SystemExit):
        retopo.parse_args(["--", "a.glb", "b.glb", "5000", str(finish.MIN_LOD_ATLAS // 2)])
    defaults = retopo.parse_args(["--", "a.glb", "b.glb"])
    assert defaults[4] == pytest.approx(math.radians(finish.ANGLE))
    assert defaults[5] == finish.VOXEL


def _mesh_glb(triangles: int, name: str = "RETOPO") -> bytes:
    """A GLB with one node and one indexed mesh; the counts are all anyone reads."""
    return glb_tools.build_glb({
        "asset": {"version": "2.0"},
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [{"name": "chest.001", "primitives": [{"attributes": {"POSITION": 0},
                                                         "indices": 1}]}],
        "accessors": [{"count": triangles + 2, "type": "VEC3", "componentType": 5126},
                      {"count": triangles * 3, "type": "SCALAR", "componentType": 5125}],
    }, b"")


def test_a_lod_is_named_after_itself():
    named = finish.name_lod(_mesh_glb(10), "chest_LOD1")
    document, _ = glb_tools.parse_glb(named)
    assert document["nodes"][0]["name"] == "chest_LOD1"
    assert document["meshes"][0]["name"] == "chest_LOD1"


def test_triangles_are_counted_from_the_file():
    assert finish.glb_triangles(_mesh_glb(1000)) == 1000
    unindexed = glb_tools.build_glb({
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}},
                                   {"attributes": {"POSITION": 0}, "mode": 1}]}],
        "accessors": [{"count": 30, "type": "VEC3", "componentType": 5126}],
    }, b"")
    assert finish.glb_triangles(unindexed) == 10     # the line primitive is not counted


class FakeRuns:
    """Stands in for Blender and gltfpack: writes each output, and notes what ran."""

    def __init__(self, fail_at: str | None = None):
        self.labels: list[str] = []
        self.commands: list[list[str]] = []
        self.fail_at = fail_at

    def __call__(self, command, log, label):
        self.labels.append(label)
        self.commands.append(command)
        if label == self.fail_at:
            raise SystemExit(f"[{label}] failed")
        after = command[command.index("--") + 1:] if "--" in command else []
        if "-o" in command:                                          # gltfpack
            output = Path(command[command.index("-o") + 1])
            output.write_bytes(Path(command[command.index("-i") + 1]).read_bytes())
        elif command[command.index("--python") + 1].endswith("blender_bake_detail.py"):
            source, target, output = map(Path, after[:3])
            assert source.is_file() and target.is_file()
            output.write_bytes(target.read_bytes())
            Path(log).write_text(finish.DETAIL_MARKER + json.dumps(
                {"metallic_roughness": "transferred", "spread": 1.001}) + "\n")
        else:                                                        # retopology
            Path(after[1]).write_bytes(_mesh_glb(int(after[2])))


@pytest.fixture
def sheet(tmp_path, monkeypatch):
    split = tmp_path / "split"
    split.mkdir()
    (split / "chest.glb").write_bytes(b"x")
    gltfpack = tmp_path / "gltfpack"
    gltfpack.write_bytes(b"")
    blender = tmp_path / "blender"
    blender.write_bytes(b"")
    runs = FakeRuns()
    monkeypatch.setattr(finish, "_run", runs)

    def run(*extra, fake=None):
        if fake is not None:
            monkeypatch.setattr(finish, "_run", fake)
        return finish.main([str(split), str(tmp_path / "out"), "--lods", "5000,1000",
                            "--gltfpack", str(gltfpack), "--blender", str(blender), *extra])

    return tmp_path / "out", runs, run


def test_a_run_names_its_lods_and_records_their_triangles(sheet):
    out, runs, run = sheet
    run()
    assert runs.labels == ["chest LOD0", "chest LOD0 detail", "chest LOD0 gltfpack",
                           "chest LOD1", "chest LOD1 detail", "chest LOD1 gltfpack"]
    record = json.loads((out / "finish_props.json").read_text())
    lods = record["props"][0]["lods"]
    assert [lod["triangles"] for lod in lods] == [5000, 1000]
    assert lods[0]["detail"] == {"metallic_roughness": "transferred", "spread": 1.001}
    for index in (0, 1):
        for web in (False, True):
            document, _ = glb_tools.parse_glb(finish.lod_path(out, "chest", index, web).read_bytes())
            assert document["nodes"][0]["name"] == f"chest_LOD{index}"


def test_each_lod_is_retopologised_then_detail_baked_from_the_original(sheet):
    out, runs, run = sheet
    run()
    retopo, detail = runs.commands[0], runs.commands[1]
    source = str(out.parent / "split" / "chest.glb")
    assert retopo[retopo.index("--") + 1] == source
    after = detail[detail.index("--") + 1:]
    # From the original onto the fresh retopology, at the LOD's own atlas size.
    assert after[:2] == [source, retopo[retopo.index("--") + 2]]
    assert after[3] == "1024"
    # Each LOD after the first gets half the atlas: LOD1 bakes at 512, in both steps.
    retopo1, detail1 = runs.commands[3], runs.commands[4]
    assert retopo1[retopo1.index("--") + 4] == "512"
    assert detail1[detail1.index("--") + 4] == "512"
    lods = json.loads((out / "finish_props.json").read_text())["props"][0]["lods"]
    assert [lod["atlas"] for lod in lods] == [1024, 512]
    # Baked in logs/, then named and moved to the LOD's own path.
    assert Path(after[2]).parent == out / "chest" / "logs"
    assert finish.lod_path(out, "chest", 0).is_file()
    # Both in-between meshes are gone once the LOD is in place.
    assert not Path(after[1]).exists()
    assert not Path(after[2]).exists()


def test_a_bake_stopped_before_its_lod_is_named_leaves_nothing_for_resume(sheet, monkeypatch):
    """--resume keeps any LOD file it finds, so an unnamed bake must never sit at that path."""
    out, _runs, run = sheet
    named = finish.name_lod

    def stopped(glb, name):
        raise KeyboardInterrupt

    monkeypatch.setattr(finish, "name_lod", stopped)
    with pytest.raises(KeyboardInterrupt):
        run()
    assert not finish.lod_path(out, "chest", 0).exists()
    monkeypatch.setattr(finish, "name_lod", named)
    again = FakeRuns()
    run("--resume", fake=again)
    assert again.labels[:2] == ["chest LOD0", "chest LOD0 detail"]
    document, _ = glb_tools.parse_glb(finish.lod_path(out, "chest", 0).read_bytes())
    assert document["nodes"][0]["name"] == "chest_LOD0"


def test_blender_is_found_automatically_or_the_run_says_where_to_get_it(tmp_path, monkeypatch):
    split = tmp_path / "chest.glb"
    split.write_bytes(b"x")
    monkeypatch.setattr(finish, "find_blender", lambda: None)
    with pytest.raises(SystemExit) as refused:
        finish.main([str(split), str(tmp_path / "out"), "--no-compress"])
    assert str(refused.value) == missing_help()
    assert not (tmp_path / "out").exists()


def test_the_record_points_at_its_files_relative_to_out_dir(sheet):
    """So a Props-tab turn, baked in a folder of its own, is right once it is moved in."""
    out, _runs, run = sheet
    run()
    entry = json.loads((out / "finish_props.json").read_text())["props"][0]
    assert entry["source"] == "../split/chest.glb"
    assert [lod["glb"] for lod in entry["lods"]] == ["chest/chest_LOD0.glb", "chest/chest_LOD1.glb"]
    assert entry["lods"][0]["web_glb"] == "chest/chest_LOD0.web.glb"
    for lod in entry["lods"]:
        assert (out / lod["glb"]).is_file() and (out / lod["web_glb"]).is_file()


def test_progress_goes_out_as_marker_lines(sheet, capsys):
    out, _runs, run = sheet
    run()
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith(finish.PROGRESS_MARKER)]
    fields = [json.loads(line[len(finish.PROGRESS_MARKER):]) for line in lines]
    assert fields[:2] == [{"prop": "chest", "lod": 0}, {"prop": "chest", "lod": 1}]
    assert fields[2]["prop"] == "chest" and fields[2]["done"].startswith("chest: LOD0 ")
    assert len(fields) == 3
    finish.lod_path(out, "chest", 1).unlink()
    run("--resume", fake=FakeRuns())
    again = [json.loads(line[len(finish.PROGRESS_MARKER):]) for line in capsys.readouterr().out.splitlines()
             if line.startswith(finish.PROGRESS_MARKER)]
    assert again[0] == {"prop": "chest", "lod": 1}     # only what is baked again


def test_a_gltfpack_cut_short_leaves_no_web_file_behind(sheet, monkeypatch):
    out, runs, run = sheet

    def half_written(command, log, label):
        if label.endswith("gltfpack"):
            Path(command[command.index("-o") + 1]).write_bytes(b"half")
            raise SystemExit(f"[{label}] failed")
        runs(command, log, label)

    monkeypatch.setattr(finish, "_run", half_written)
    with pytest.raises(SystemExit):
        run()
    assert finish.lod_path(out, "chest", 0).is_file()
    assert not finish.lod_path(out, "chest", 0, web=True).exists()


def test_resume_keeps_what_is_done_and_repacks_only_what_was_baked_again(sheet):
    out, _runs, run = sheet
    run()
    finish.lod_path(out, "chest", 1).unlink()
    again = FakeRuns()
    run("--resume", fake=again)
    assert again.labels == ["chest LOD1", "chest LOD1 detail", "chest LOD1 gltfpack"]


def test_resume_refuses_lods_baked_with_other_settings(sheet):
    _out, _runs, run = sheet
    run()
    again = FakeRuns()
    with pytest.raises(SystemExit, match="atlas"):
        run("--resume", "--atlas", "2048", fake=again)
    assert again.labels == []
    with pytest.raises(SystemExit, match="surface"):
        run("--resume", "--roughness", "0.9", fake=again)


def test_the_record_is_written_before_the_first_bake(sheet):
    """So a run that dies part way can still be resumed safely."""
    out, _runs, run = sheet
    with pytest.raises(SystemExit):
        run("--atlas", "2048", fake=FakeRuns(fail_at="chest LOD0"))
    assert json.loads((out / "finish_props.json").read_text())["atlas"] == 2048
    with pytest.raises(SystemExit, match="atlas"):
        run("--resume", fake=FakeRuns())


def test_resume_without_a_record_keeps_the_lods():
    assert finish.resume_mismatch(None, {"lods": [1], "atlas": 1024, "surface": {}}) == []


def test_lod_atlas_halves_per_lod_and_stops_at_256():
    assert [finish.lod_atlas(1024, i) for i in range(4)] == [1024, 512, 256, 256]
    assert [finish.lod_atlas(4096, i) for i in range(3)] == [4096, 2048, 1024]


def test_resume_refuses_lods_baked_before_the_atlas_halved(sheet):
    """Older records have no per-LOD atlas: their far LODs were baked at full size."""
    out, _runs, run = sheet
    run()
    record_path = out / "finish_props.json"
    record = json.loads(record_path.read_text())
    del record["lod_atlas"]
    record_path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="lod_atlas"):
        run("--resume")
