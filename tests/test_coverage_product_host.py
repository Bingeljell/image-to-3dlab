"""Exercise host detection and local system probes without requiring each device."""

import os
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from image_to_3dlab import host


@pytest.mark.parametrize(("family", "result", "expected"), [
    ("macos", "Apple M5 Pro\n", "Apple M5 Pro"),
    ("macos", "", None),
    ("macos", OSError("sysctl missing"), None),
    ("linux", "NVIDIA GeForce RTX 3070\n", "NVIDIA GeForce RTX 3070"),
])
def test_chip_name_names_graphics_hardware_from_system_probe(family, result, expected):
    def run(args, **kwargs):
        if isinstance(result, Exception):
            raise result
        expected_tool = "sysctl" if family == "macos" else "/tools/nvidia-smi"
        assert args[0] == expected_tool
        return subprocess.CompletedProcess(args, 0, stdout=result)

    assert host.chip_name(which=lambda name: "/tools/" + name, run=run, family=family) == expected


def test_unknown_linux_graphics_keeps_existing_platform():
    assert host.host_platform(sys_platform="linux", machine="x86_64", nvidia=lambda: False,
                              amd=lambda: False) == host.OTHER


@pytest.mark.parametrize(("found", "default", "result"), [
    ("/tools/nvcc", False, "/tools/nvcc"),
    (None, True, "/usr/local/cuda/bin/nvcc"),
    (None, False, None),
])
def test_find_compiler_falls_back_to_toolkit_path(monkeypatch, found, default, result):
    monkeypatch.setattr(host.shutil, "which", lambda name: found if name == "nvcc" else None)
    class ToolkitPath(type(host.CGROUP)):
        def exists(self):
            return default

    monkeypatch.setattr(host, "Path", ToolkitPath)
    assert host.find_nvcc() == result


@pytest.mark.parametrize("version", [None, SimpleNamespace(cuda="13.0", hip="7.1")])
def test_host_detects_installed_torch_build_without_importing_runtime(monkeypatch, version):
    monkeypatch.setitem(sys.modules, "torch", None if version is None else SimpleNamespace(version=version))
    assert host._installed_torch_cuda() == (None if version is None else "13.0")
    assert host._installed_torch_hip() == (None if version is None else "7.1")


def test_host_entry_point_reports_missing_lockfile(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["host", "without-cuda", str(tmp_path / "missing.txt")])
    with pytest.raises(SystemExit) as exit:
        runpy.run_path(host.__file__, run_name="__main__")
    assert exit.value.code == 2
    assert "cannot read" in capsys.readouterr().err


def test_total_memory_handles_unsupported_sysconf(monkeypatch):
    def unsupported(_name):
        raise ValueError("not supported")

    monkeypatch.setattr(os, "sysconf", unsupported, raising=False)
    assert host.total_memory() is None
