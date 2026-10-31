"""Tests for the text-to-image bootstrap.

The download is 14.2 GB, so what matters here is everything that happens *before* one
byte moves: that the user is told what is coming, that nothing starts without a yes, and
that a non-interactive run cannot silently spend someone's bandwidth.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "bootstrap_qwen_image", SCRIPTS / "bootstrap_qwen_image.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


boot = _load()
REAL_INSTALL_BINARY = boot.install_binary


@pytest.fixture(autouse=True)
def no_real_install(monkeypatch, tmp_path):
    """Never the real vendor/sdcpp, never the network. On 2026-10-19 a test that ran main()
    reached the real install_binary and put the Linux engine over a working Mac one."""
    monkeypatch.setattr(boot, "VENDOR", tmp_path / "sdcpp")
    monkeypatch.setattr(boot, "BINARY", tmp_path / "sdcpp" / "sd-cli")
    # a test that says the engine is installed means the current one, unless it says otherwise
    (tmp_path / "sdcpp").mkdir()
    boot.record_release(tmp_path / "sdcpp", boot.PINNED_RELEASE)
    monkeypatch.setattr(boot, "install_binary",
                        lambda *a, **k: pytest.fail("a test reached the real engine download"))

    def offline(*a, **k):
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(boot.urllib.request, "urlopen", offline)
    monkeypatch.setattr(boot.urllib.request, "urlretrieve", offline)
    # never the real Vulkan folders, never the real driver library
    monkeypatch.setattr(boot, "VULKAN_ICD_DIRS", (tmp_path / "icd-system",))
    monkeypatch.setattr(boot, "user_icd_dir", lambda: tmp_path / "icd-user")
    monkeypatch.setattr(boot, "nvidia_glx_present", lambda: False)


def test_announcement_names_backend_route_size_and_licence():
    """AGENTS.md: a download path must name the backend, the route and the size, and
    require an affirmative answer. The first three are checked here."""
    text = boot.announcement()
    assert "Qwen-Image 2.1" in text
    assert "stable-diffusion.cpp" in text
    assert "14.2 GB" in text
    assert "Qwen Research License" in text
    assert "Built with Qwen" in text


def test_announcement_lists_every_file_with_its_size():
    text = boot.announcement()
    for _, filename, size in boot.WEIGHTS:
        assert filename in text
        assert f"{size:.2f}" in text


def test_total_matches_the_catalogue_figure():
    """The Setup page and this script must not quote different numbers for one download."""
    assert boot.total_gb() == pytest.approx(14.2, abs=0.05)


def test_build_only_does_not_promise_weights(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: "macos-arm64")
    text = boot.announcement(weights=False)
    assert "weights:" not in text
    assert "build:" in text


def test_nothing_downloads_without_a_yes(monkeypatch, capsys):
    """Declining must stop, and must not call either installer."""
    monkeypatch.setattr(boot, "install_binary", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(boot, "install_weights", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(boot.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert boot.main([]) == 1
    assert "Nothing downloaded" in capsys.readouterr().out


def test_a_non_interactive_run_refuses_rather_than_hanging(monkeypatch, capsys):
    """No tty and no --yes means nobody is there to consent. Waiting on input() would
    hang a CI job or a subprocess forever; defaulting to yes would break the rule."""
    monkeypatch.setattr(boot, "install_binary", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(boot, "install_weights", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(boot.sys.stdin, "isatty", lambda: False)
    assert boot.main([]) == 1
    assert "--yes" in capsys.readouterr().out


def test_yes_proceeds_without_asking(monkeypatch):
    called = []
    monkeypatch.setattr(boot, "install_binary", lambda *a, **k: called.append("build"))
    monkeypatch.setattr(boot, "install_weights", lambda *a, **k: called.append("weights"))
    monkeypatch.setattr(boot, "binary_present", lambda: False)
    # The probe runs on whatever machine this test happens to be on; answer as a GPU box.
    monkeypatch.setattr(boot, "probe_gpu", lambda *a: "ggml_vulkan: Found 1 Vulkan devices:")
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("should not ask"))
    assert boot.main(["--yes"]) == 0
    assert called == ["build", "weights"]


def test_weights_only_skips_the_binary(monkeypatch):
    called = []
    monkeypatch.setattr(boot, "install_binary", lambda *a, **k: called.append("build"))
    monkeypatch.setattr(boot, "install_weights", lambda *a, **k: called.append("weights"))
    assert boot.main(["--yes", "--weights-only"]) == 0
    assert called == ["weights"]


def test_an_existing_binary_is_not_redownloaded(monkeypatch, capsys):
    """Re-running a bootstrap on an installed tree must no-op, not re-fetch. AGENTS.md
    asks patch scripts to be idempotent and the same courtesy applies here."""
    monkeypatch.setattr(boot, "install_binary", lambda *a, **k: pytest.fail("redownloaded"))
    monkeypatch.setattr(boot, "install_weights", lambda *a, **k: None)
    monkeypatch.setattr(boot, "binary_present", lambda: True)
    monkeypatch.setattr(boot, "binary_stale", lambda: False)
    # The probe still runs on a re-run (the binary may have been swapped for a dud).
    monkeypatch.setattr(boot, "probe_gpu", lambda *a: "ggml_vulkan: Found 1 Vulkan devices:")
    assert boot.main(["--yes"]) == 0
    assert "already installed" in capsys.readouterr().out


# --- The pinned engine (2026-10-19): the Sept build took 134 s to decode one picture on Metal,
# master-975 takes 10 s. Every machine is pinned and updated in place.
def test_every_machine_with_a_build_is_pinned_to_one_release():
    assert set(boot.PINNED) == set(boot.BUILDS)
    for key in boot.BUILDS:
        assert boot.release_url(key).endswith("/releases/tags/" + boot.PINNED_RELEASE)
    assert boot.release_url(None).endswith("/releases/latest")


def test_the_pinned_release_has_every_build_we_install():
    """Its asset list, 2026-10-18: one archive (or two, for Windows) per machine type."""
    assets = [{"name": n} for n in (
        "cudart-sd-bin-win-cu12-x64.zip", "sd-master-95b3090-bin-Darwin-macOS-26.6.2-arm64.zip",
        "sd-master-95b3090-bin-Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip",
        "sd-master-95b3090-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
        "sd-master-95b3090-bin-Linux-Ubuntu-24.04-x86_64.zip", "sd-master-95b3090-bin-win-cpu-x64.zip",
        "sd-master-95b3090-bin-win-cuda12-x64.zip", "sd-master-95b3090-bin-win-rocm-10.1.0-x64.zip",
        "sd-master-95b3090-bin-win-vulkan-x64.zip")]
    for build in boot.BUILDS.values():
        assert len(boot.pick_assets(assets, build)) == len(build.assets)


def test_the_installed_release_is_recorded_and_read_back(tmp_path):
    assert boot.installed_release(tmp_path) is None
    boot.record_release(tmp_path, "master-975-95b3090")
    assert boot.installed_release(tmp_path) == "master-975-95b3090"


@pytest.mark.parametrize("key, installed, stale", [
    ("macos-arm64", None, True),                     # installed before we kept a record
    ("macos-arm64", "master-899-28b454b", True),
    ("macos-arm64", "master-975-95b3090", False),
    ("linux-amd", None, True),
    ("windows-nvidia", "master-899-28b454b", True),
    ("linux-nvidia", "master-975-95b3090", False),
    (None, None, False),                             # no build for this machine at all
])
def test_only_a_pinned_machine_on_another_release_is_stale(monkeypatch, tmp_path, key, installed, stale):
    monkeypatch.setattr(boot, "target", lambda: key)
    if installed:
        boot.record_release(tmp_path, installed)
    assert boot.binary_stale(tmp_path) is stale


def test_a_stale_mac_engine_is_replaced_and_the_weights_left_alone(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(boot, "target", lambda: "macos-arm64")
    monkeypatch.setattr(boot, "binary_present", lambda: True)
    monkeypatch.setattr(boot, "binary_stale", lambda: True)
    monkeypatch.setattr(boot, "install_binary", lambda *a, **k: called.append("build"))
    monkeypatch.setattr(boot, "install_weights", lambda *a, **k: called.append("weights"))
    assert boot.main(["--yes", "--build-only"]) == 0
    assert called == ["build"]
    assert "Updating" in capsys.readouterr().out


# The asset list of stable-diffusion.cpp release master-899-28b454b, 2026-09-23.
RELEASE = [{"name": n} for n in (
    "cudart-sd-bin-win-cu12-x64.zip",
    "sd-master-28b454b-bin-Darwin-macOS-26.6.2-arm64.zip",
    "sd-master-28b454b-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.14.0.zip",
    "sd-master-28b454b-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
    "sd-master-28b454b-bin-Linux-Ubuntu-24.04-x86_64.zip",
    "sd-master-28b454b-bin-win-cpu-x64.zip",
    "sd-master-28b454b-bin-win-cuda12-x64.zip",
    "sd-master-28b454b-bin-win-rocm-7.14.0-x64.zip",
    "sd-master-28b454b-bin-win-vulkan-x64.zip",
)]


@pytest.mark.parametrize("key,expected", [
    ("macos-arm64", ["sd-master-28b454b-bin-Darwin-macOS-26.6.2-arm64.zip"]),
    ("windows-nvidia", ["sd-master-28b454b-bin-win-cuda12-x64.zip",
                        "cudart-sd-bin-win-cu12-x64.zip"]),
    ("linux-nvidia", ["sd-master-28b454b-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"]),
    # The ROCm build, not the Vulkan one: this target means an AMD card with ROCm on it,
    # and native HIP is faster than going through Vulkan for that hardware.
    ("linux-amd", ["sd-master-28b454b-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.14.0.zip"]),
])
def test_each_machine_gets_its_own_build_from_a_real_release(key, expected):
    picked = boot.pick_assets(RELEASE, boot.BUILDS[key])
    assert [a["name"] for a in picked] == expected


def test_asset_matching_survives_upstream_renaming_its_build_machine():
    """The asset name carries upstream's build-machine OS version, which changes without
    warning. Matching it exactly would break the installer on their next CI upgrade."""
    renamed = [{"name": "sd-master-1234567-bin-Darwin-macOS-15.0-arm64.zip"}]
    assert boot.pick_assets(renamed, boot.BUILDS["macos-arm64"]) == renamed


def test_asset_matching_prefers_nothing_over_the_wrong_architecture():
    intel = [{"name": "sd-bin-Darwin-macOS-14-x64.zip"}]
    assert boot.pick_assets(intel, boot.BUILDS["macos-arm64"]) == []


def test_windows_without_the_cuda_runtime_is_no_install_at_all():
    """Half an install -- sd-cli.exe without its CUDA DLLs -- fails to start with no
    useful message, so a release missing either archive counts as missing both."""
    no_runtime = [a for a in RELEASE if not a["name"].startswith("cudart")]
    assert boot.pick_assets(no_runtime, boot.BUILDS["windows-nvidia"]) == []


def test_announcement_states_this_machines_download_size(monkeypatch):
    """AGENTS.md: a download names its size. Windows pulls ~850 MB for the binary, not
    the 35 MB a Mac does, so the announcement has to know which machine it is on."""
    monkeypatch.setattr(boot, "target", lambda: "windows-nvidia")
    text = boot.announcement(weights=False)
    assert "CUDA 12 on Windows" in text and "~850 MB" in text


def test_announcement_on_an_amd_machine_names_the_rocm_build(monkeypatch):
    """An AMD Linux box gets the native ROCm build named with its size, not a refusal."""
    monkeypatch.setattr(boot, "target", lambda: "linux-amd")
    text = boot.announcement(weights=False)
    assert "ROCm on Linux (AMD)" in text and "~278 MB" in text


def test_announcement_on_an_unsupported_machine_offers_no_build(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: None)
    text = boot.announcement(weights=False)
    assert "no prebuilt build" in text and "build:" not in text


def test_install_refuses_an_unsupported_machine_before_touching_the_network(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: None)
    monkeypatch.setattr(boot.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("went to the network"))
    with pytest.raises(SystemExit, match="no prebuilt"):
        REAL_INSTALL_BINARY()


def test_finish_install_flattens_and_marks_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(boot.host, "os_family", lambda *a: "linux")
    nested = tmp_path / "build" / "bin"
    nested.mkdir(parents=True)
    (nested / "sd-cli").write_text("")
    (nested / "libstable-diffusion.so").write_text("")
    os.symlink("libstable-diffusion.so", nested / "libsd.so")
    binary = boot.finish_install(tmp_path)
    assert binary == tmp_path / "sd-cli"
    assert (tmp_path / "libstable-diffusion.so").exists()
    # The flatten must carry real links as links, not turn them into text files.
    assert (tmp_path / "libsd.so").is_symlink()
    assert binary.stat().st_mode & 0o111
    assert not (tmp_path / "build").exists()


def test_symlinks_in_the_release_are_restored(tmp_path):
    """zipfile extracts a stored symlink as a text file holding its target, and the
    dynamic loader then dies with 'file too short' on the first versioned .so. The
    installer must put real links back before anything runs."""
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("build/bin/libfoo.so.1", b"real library bytes")
        link = zipfile.ZipInfo("build/bin/libfoo.so")
        link.external_attr = 0xA0000000 | (0o755 << 16)
        bundle.writestr(link, "libfoo.so.1")
    destination = tmp_path / "dest"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
        boot.restore_symlinks(bundle, destination)
    link = destination / "build" / "bin" / "libfoo.so"
    assert link.is_symlink()
    assert link.read_bytes() == b"real library bytes"  # follows the link


def test_restore_symlinks_leaves_real_files_alone(tmp_path):
    """A regular file that happens to contain a path must not become a link."""
    destination = tmp_path / "dest"
    (destination / "build" / "bin").mkdir(parents=True)
    (destination / "build" / "bin" / "ggml.txt").write_text("libfoo.so.1\n")
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("build/bin/ggml.txt", "libfoo.so.1\n")
    with zipfile.ZipFile(archive) as bundle:
        boot.restore_symlinks(bundle, destination)
    assert not (destination / "build" / "bin" / "ggml.txt").is_symlink()


@pytest.mark.parametrize("target", ["/etc/passwd", "../../../outside", "../../../../etc/passwd"])
def test_a_link_pointing_out_of_the_install_is_not_restored(tmp_path, target):
    """The archive picks where its links point; one aimed outside the folder stays the
    harmless text file zipfile wrote, so nothing later writes or chmods through it."""
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        link = zipfile.ZipInfo("build/bin/sd-cli")
        link.external_attr = 0xA0000000 | (0o755 << 16)
        bundle.writestr(link, target)
    destination = tmp_path / "dest"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
        boot.restore_symlinks(bundle, destination)
    assert not (destination / "build" / "bin" / "sd-cli").is_symlink()


def test_a_link_to_a_sibling_stays_inside(tmp_path):
    destination = tmp_path / "dest"
    link = destination / "build" / "bin" / "libfoo.so"
    assert boot.link_stays_inside(link, "libfoo.so.1", destination)
    assert boot.link_stays_inside(link, "../lib/libfoo.so.1", destination)


def _nvidia_linux(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: "linux-nvidia")
    monkeypatch.setattr(boot, "binary_present", lambda: True)


def test_nvidia_install_stops_before_the_weights_if_the_gpu_is_unreachable(
        monkeypatch, capsys):
    """Verify the environment before the expensive step: 13 GB of weights are no use to a
    binary that will run on the CPU."""
    _nvidia_linux(monkeypatch)
    monkeypatch.setattr(boot, "probe_gpu", lambda *a: "load_backend: loaded CPU backend")
    monkeypatch.setattr(boot, "install_weights", lambda: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1
    assert "libegl1 libgl1" in capsys.readouterr().out


def test_nvidia_install_continues_when_the_gpu_answers(monkeypatch):
    _nvidia_linux(monkeypatch)
    monkeypatch.setattr(boot, "probe_gpu", lambda *a: "ggml_vulkan: Found 1 Vulkan devices:")
    fetched = []
    monkeypatch.setattr(boot, "install_weights", lambda: fetched.append(1))
    assert boot.main(["--yes"]) == 0
    assert fetched == [1]


def test_a_missing_vulkan_icd_is_written_when_the_nvidia_library_is_there(tmp_path):
    """NVIDIA containers often mount the driver library but not the small file that tells
    Vulkan where it is, so sd-cli sees no GPU (RunPod A40, 2026-10-22)."""
    written = boot.add_nvidia_icd(search_dirs=(tmp_path / "system",),
                                  write_dir=tmp_path / "user", glx_present=True)
    assert written == tmp_path / "user" / "nvidia_icd.json"
    icd = json.loads(written.read_text())
    assert icd["ICD"]["library_path"] == "libGLX_nvidia.so.0"


def test_an_existing_nvidia_icd_is_left_alone(tmp_path):
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "nvidia_icd.json").write_text("{}")
    assert boot.add_nvidia_icd(search_dirs=(tmp_path / "system",),
                               write_dir=tmp_path / "user", glx_present=True) is None
    assert not (tmp_path / "user").exists()


def test_no_icd_is_written_without_the_nvidia_library(tmp_path):
    assert boot.add_nvidia_icd(search_dirs=(tmp_path / "system",),
                               write_dir=tmp_path / "user", glx_present=False) is None
    assert not (tmp_path / "user").exists()


def test_nvidia_install_writes_the_icd_and_checks_the_gpu_again(monkeypatch, tmp_path):
    _nvidia_linux(monkeypatch)
    monkeypatch.setattr(boot, "nvidia_glx_present", lambda: True)
    answers = iter(["load_backend: loaded CPU backend", "ggml_vulkan: Found 1 Vulkan devices:"])
    monkeypatch.setattr(boot, "probe_gpu", lambda *a: next(answers))
    fetched = []
    monkeypatch.setattr(boot, "install_weights", lambda: fetched.append(1))
    assert boot.main(["--yes"]) == 0
    assert fetched == [1]
    assert (tmp_path / "icd-user" / "nvidia_icd.json").exists()


def test_the_icd_is_not_tried_on_amd(monkeypatch, tmp_path):
    _amd_linux(monkeypatch)
    monkeypatch.setattr(boot, "nvidia_glx_present", lambda: True)
    monkeypatch.setattr(boot, "probe_gpu", lambda *a: "load_backend: loaded CPU backend")
    monkeypatch.setattr(boot, "install_weights", lambda: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1
    assert not (tmp_path / "icd-user").exists()


def _amd_linux(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: "linux-amd")
    monkeypatch.setattr(boot, "binary_present", lambda: True)


def test_amd_install_stops_before_the_weights_if_the_gpu_is_unreachable(
        monkeypatch, capsys):
    """Same rule as NVIDIA: a binary that will only run on the CPU is no use before 13 GB
    of weights have been fetched for it."""
    _amd_linux(monkeypatch)
    monkeypatch.setattr(boot, "probe_gpu", lambda *a: "load_backend: loaded CPU backend")
    monkeypatch.setattr(boot, "install_weights", lambda: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1
    assert "rocminfo" in capsys.readouterr().out


def test_amd_install_continues_when_the_gpu_answers(monkeypatch):
    _amd_linux(monkeypatch)
    monkeypatch.setattr(boot, "probe_gpu",
                        lambda *a: "ggml_cuda_init: found 2 ROCm devices (Total VRAM: 55486 MiB):")
    fetched = []
    monkeypatch.setattr(boot, "install_weights", lambda: fetched.append(1))
    assert boot.main(["--yes"]) == 0
    assert fetched == [1]


def test_a_mac_is_not_probed(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: "macos-arm64")
    monkeypatch.setattr(boot, "binary_present", lambda: True)
    monkeypatch.setattr(boot, "probe_gpu", lambda *a: pytest.fail("probed a Mac"))
    monkeypatch.setattr(boot, "install_weights", lambda: None)
    assert boot.main(["--yes"]) == 0


def test_probe_gpu_runs_the_binary_and_returns_its_chatter(tmp_path):
    fake = tmp_path / "sd-cli"
    fake.write_text("#!/bin/sh\necho 'ggml_vulkan: Found 1 Vulkan devices:'\nexit 1\n")
    fake.chmod(0o755)
    assert "Found 1" in boot.probe_gpu(fake)


def test_probe_gpu_survives_a_binary_that_will_not_start(tmp_path):
    assert boot.probe_gpu(tmp_path / "missing") == ""


# --- --vision: the projector edit mode (back pictures) needs ---------------------------------


def test_vision_announcement_names_the_file_size_and_why():
    text = boot.vision_announcement()
    assert "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf" in text and "0.75 GB" in text
    assert "back" in text and "Qwen" in text


def test_vision_is_the_file_the_image_tool_looks_for():
    sys.path.insert(0, str(SCRIPTS.parent / "viewer"))
    import image_api
    repo, filename, _ = boot.VISION
    assert filename == image_api.VISION_FILE[1]
    assert "models--" + repo.replace("/", "--") == image_api.VISION_FILE[0]


def test_vision_without_a_yes_downloads_nothing(monkeypatch):
    monkeypatch.setattr(boot, "install_vision", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(boot.sys.stdin, "isatty", lambda: False)
    assert boot.main(["--vision"]) == 1


def test_vision_with_a_yes_fetches_only_the_projector(monkeypatch):
    monkeypatch.setattr(boot, "install_binary", lambda *a, **k: pytest.fail("binary"))
    monkeypatch.setattr(boot, "install_weights", lambda *a, **k: pytest.fail("weights"))
    fetched = []
    monkeypatch.setattr(boot, "install_vision", lambda *a, **k: fetched.append(1))
    assert boot.main(["--vision", "--yes"]) == 0
    assert fetched == [1]


def test_a_file_that_is_not_the_tested_one_is_deleted_and_stops_the_install(tmp_path):
    """Security review 2026-10-31: a release file can be swapped under a pinned tag."""
    good = tmp_path / "good.zip"
    good.write_bytes(b"tested")
    boot.check_sha256(good, boot.sha256_of(good), "good.zip")          # passes, file kept
    assert good.exists()
    with pytest.raises(SystemExit, match="does not match"):
        boot.check_sha256(good, "0" * 64, "good.zip")
    assert not good.exists()
    unknown = tmp_path / "new.zip"
    unknown.write_bytes(b"x")
    with pytest.raises(SystemExit, match="no fingerprint"):
        boot.check_sha256(unknown, None, "new.zip")
    assert not unknown.exists()


@pytest.mark.parametrize("key", sorted(boot.BUILDS))
def test_every_machine_s_pinned_archives_have_a_fingerprint(key):
    """Every archive pick_assets would choose for a build has its SHA-256 on record."""
    fake_release = [{"name": name} for name in boot.ARCHIVE_SHA256]
    picked = boot.pick_assets(fake_release, boot.BUILDS[key])
    assert len(picked) == len(boot.BUILDS[key].assets)
    assert boot.PINNED[key] == boot.PINNED_RELEASE


def test_install_checks_the_fingerprint_before_unpacking(monkeypatch, tmp_path):
    monkeypatch.setattr(boot, "target", lambda: "macos-arm64")
    name = "sd-master-95b3090-bin-Darwin-macOS-26.6.2-arm64.zip"
    release = {"tag_name": boot.PINNED_RELEASE, "assets": [{"name": name, "size": 1, "browser_download_url": "u"}]}

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(boot.urllib.request, "urlopen", lambda *a, **k: Reply(json.dumps(release).encode()))

    def swapped(url, target):
        with zipfile.ZipFile(target, "w") as bundle:
            bundle.writestr("sd-cli", "not what we tested")
    monkeypatch.setattr(boot.urllib.request, "urlretrieve", swapped)
    dest = tmp_path / "engine"
    with pytest.raises(SystemExit, match="does not match"):
        REAL_INSTALL_BINARY(dest)
    assert not (dest / "sd-cli").exists() and not (dest / "release.zip").exists()


def test_every_qwen_file_is_pinned_and_a_swapped_one_stops_setup(monkeypatch, tmp_path):
    """Every file setup fetches has a tested revision and SHA-256; a mismatch stops it."""
    assert set(boot.PINS) == {(repo, name) for repo, name, _ in (*boot.WEIGHTS, boot.VISION)}
    assert all(len(rev) == 40 and len(sha) == 64 for rev, sha in boot.PINS.values())
    asked = []

    def fake(repo_id, filename, revision=None):
        asked.append(revision)
        path = tmp_path / Path(filename).name
        path.write_bytes(b"tested")
        return str(path)

    good = {("r", "f.gguf"): ("a" * 40, boot.hashlib.sha256(b"tested").hexdigest())}
    assert boot.fetch_pinned(fake, "r", "f.gguf", 0.1, good).read_bytes() == b"tested"
    bad = {("r", "f.gguf"): ("a" * 40, "0" * 64)}
    with pytest.raises(SystemExit, match="not the file we tested"):
        boot.fetch_pinned(fake, "r", "f.gguf", 0.1, bad)
    assert asked == ["a" * 40] * 2
    assert (tmp_path / "f.gguf").exists()          # a shared cache file is never deleted
