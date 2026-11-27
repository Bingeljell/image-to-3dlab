"""Prevent accidental external work through Python's socket and process APIs."""

from __future__ import annotations

import inspect
import ipaddress
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path, PureWindowsPath

SUPPORT = Path(__file__).resolve().parent
NODE_PRELOAD = f"--import={SUPPORT.joinpath('offline.mjs').as_uri()}"
NODE_COVERAGE_PRELOAD = f"--import={SUPPORT.joinpath('coverage-hook.mjs').as_uri()}"
COVERAGE_ENV = {
    "AF_COVERAGE_INVENTORY", "AF_COVERAGE_JS_RAW", "AF_COVERAGE_ROOT",
    "COVERAGE_FILE", "COVERAGE_PROCESS_CONFIG", "COVERAGE_PROCESS_START", "COVERAGE_RCFILE",
}
REPO = SUPPORT.parents[1]
# These tools support local repositories, JavaScript fixtures and process/host checks.
LOCAL_TOOLS = {"git", "lscpu", "node", "ps", "sleep", "sysctl", "taskkill", "true", "uname"}
WINDOWS_EXECUTABLE = re.compile(r'"([^"]+)"(?=[ \t]|$)|([^"\s]+)(?=[ \t]|$)')
OFFLINE_ENV = {
    "AF_OFFLINE_TESTS": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_XET": "1",
    "GIT_ALLOW_PROTOCOL": "file",
}
BLENDER_PORT = 9876
_integration = None
_installed = False


def _blocked(operation, value=None):
    detail = "" if value is None else f" {value!r}"
    raise RuntimeError(f"Offline tests block {operation}{detail}")


def _local_host(host):
    if isinstance(host, bytes):
        host = os.fsdecode(host)
    if host in ("localhost", "localhost."):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or bool(getattr(address, "ipv4_mapped", None) and address.ipv4_mapped.is_loopback)


def _network_socket(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        host, port, *_ = address
        if not _local_host(host) or port == BLENDER_PORT:
            _blocked("connection to", address)
    elif sock.family != getattr(socket, "AF_UNIX", None):
        _blocked("socket address", address)


def _program_name(value):
    return PureWindowsPath(os.fsdecode(value)).name.lower().removesuffix(".exe")


def _in_fixture(path, env):
    root = env.get("AF_TEST_FIXTURE_ROOT")
    return bool(root and path and Path(path).resolve().is_relative_to(Path(root).resolve()))


def _fixture_program(executable, env):
    path = Path(executable)
    if not path.is_absolute():
        path = shutil.which(executable, path=env.get("PATH"))
        if path is None:
            return False
        path = Path(path)
    if not _in_fixture(path, env):
        return False
    if not path.exists():
        return True  # Let subprocess produce its ordinary FileNotFoundError.
    if not path.is_file():
        return False
    with path.open("rb") as program:
        return program.read(2) == b"#!"


def _fixture_shell(args, env):
    if not isinstance(args, (list, tuple)) or len(args) < 2:
        return False
    command = os.fsdecode(args[1])
    if command == "-n":
        return len(args) == 3  # Syntax check only.
    if command == "-c":
        # The RunPod launch test supplies a temporary home holding its fake lab.
        return (
            len(args) == 3
            and _in_fixture(env.get("HOME"), env)
            and re.fullmatch(
                r"cd ~/image-to-3dlab; RUNPOD_POD_ID=[\w-]+ nohup \./lab > lab\.log 2>&1 < /dev/null &",
                os.fsdecode(args[2]),
            ) is not None
        )
    paths = env.get("PATH", "").split(os.pathsep)
    if Path(command).resolve() == REPO / "install.sh":
        return len(paths) == 1 and _in_fixture(paths[0], env) and _in_fixture(env.get("HOME"), env)
    return (
        Path(command).resolve() in {REPO / "scripts" / name for name in (
            "bootstrap_macos.sh", "bootstrap_trellis_macos.sh",
        )}
        and _in_fixture(paths[0], env)
        and all(path in {"/usr/bin", "/bin"} for path in paths[1:])
    )


def _check_program(executable, args, env):
    env = os.environ if env is None else env
    if executable is None:
        if sys.platform != "win32" or not isinstance(args, str):
            _blocked("process without executable")
        if args == "lscpu":
            # NumPy treats a missing Linux CPU probe as lack of SVE support.
            # Report it as unavailable without searching Windows PATH.
            raise FileNotFoundError("lscpu is unavailable on Windows")
        first = WINDOWS_EXECUTABLE.match(args)
        if first is None:
            _blocked("Windows command line without a clear executable")
        executable = first.group(1) or first.group(2)
    executable = os.fsdecode(executable)
    name = _program_name(executable)
    if _integration == "blender" and (
        name in {"blender", "org.blender.blender"}
        or executable == env.get("I2L_BLENDER")
    ):
        return
    if _integration == "runtime" and name == "c++":
        return
    if isinstance(args, str):  # Windows passes a command line to the audit hook.
        args = [quoted or bare for quoted, bare in re.findall(r'"([^"]+)"|(\S+)', args)]
    if sys.platform == "win32" and (
        name == "lscpu" or (args and _program_name(args[0]) == "lscpu")
    ):
        _blocked("process", executable)
    if name == "ldconfig" or (args and _program_name(args[0]) == "ldconfig"):
        # CPython's ctypes reads the glibc cache via this absolute path.
        # The OS follows the system's /sbin symlink on merged-/usr installs.
        if (
            sys.platform == "linux"
            and executable == "/sbin/ldconfig"
            and tuple(map(os.fsdecode, args)) == ("/sbin/ldconfig", "-p")
        ):
            return
        _blocked("process", executable)
    if name in {"bash", "sh"} and _fixture_shell(args, env):
        return
    if re.fullmatch(r"python(?:3(?:\.\d+)?)?", name):
        for argument in args[1:]:
            argument = os.fsdecode(argument)
            if argument == "-c":
                break
            script = _program_name(argument)
            if (
                script in {
                    "pipeline.py", "download_weights.py", "classify_trellis_input.py",
                    "run_shape.py", "run_paint.py", "run_paint_pbr.py",
                }
                or script.startswith("bootstrap_")
                or script.endswith("_generate.py")
            ):
                _blocked("model or installer command", script)
        return
    if name not in LOCAL_TOOLS and not _fixture_program(executable, env):
        _blocked("process", executable)


def _audit(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
        host = args[0]
        if host is not None and not _local_host(host):
            _blocked("name lookup for", host)
    elif event == "socket.getnameinfo":
        if not _local_host(args[0][0]):
            _blocked("reverse name lookup for", args[0])
    elif event == "socket.bind":
        sock, address = args
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            host = address[0]
            if host not in ("", "0.0.0.0", "::") and not _local_host(host):
                _blocked("bind to", address)
    elif event in {"socket.connect", "socket.sendto", "socket.sendmsg"}:
        sock, address = args
        if address is not None:
            _network_socket(sock, address)
    elif event == "subprocess.Popen":
        _check_program(args[0], args[1], args[3])
    elif event in {"os.exec", "os.posix_spawn", "os.system", "os.startfile", "os.startfile/2"}:
        _blocked("shell or application launch")


def _child_environment(env):
    result = {os.fsdecode(key): os.fsdecode(value) for key, value in env.items()}
    result.update(OFFLINE_ENV)
    for key in COVERAGE_ENV & os.environ.keys():
        result.setdefault(key, os.environ[key])
    if fixture_root := os.environ.get("AF_TEST_FIXTURE_ROOT"):
        result["AF_TEST_FIXTURE_ROOT"] = fixture_root
    paths = result.get("PYTHONPATH", "").split(os.pathsep)
    result["PYTHONPATH"] = os.pathsep.join([str(SUPPORT), *(p for p in paths if p and p != str(SUPPORT))])
    options = result.get("NODE_OPTIONS", "")
    preloads = [NODE_PRELOAD]
    if result.get("AF_COVERAGE_INVENTORY"):
        preloads.append(NODE_COVERAGE_PRELOAD)
    missing_preloads = [preload for preload in preloads if preload not in options.split()]
    result["NODE_OPTIONS"] = " ".join([options, *missing_preloads]).strip()
    return result


def _use_native_windows_version():
    if sys.platform != "win32":
        return
    import platform

    if hasattr(platform, "_syscmd_ver"):
        # Python 3.11 tries the shell's ver command first. Returning these
        # defaults lets win32_ver use its own sys.getwindowsversion fallback.
        def native_defaults(system="", release="", version="", supported_platforms=()):
            return system, release, version

        platform._syscmd_ver = native_defaults


def install():
    global _installed
    if _installed:
        return
    _installed = True
    os.environ.update(OFFLINE_ENV)
    sys.addaudithook(_audit)
    _use_native_windows_version()

    real_init = subprocess.Popen.__init__
    signature = inspect.signature(real_init)

    def guarded_init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        if bound.arguments.get("shell"):
            _blocked("implicit shell command")
        env = bound.arguments.get("env")
        child_env = _child_environment(os.environ if env is None else env)
        if sys.platform == "win32":
            command = bound.arguments["args"]
            executable = bound.arguments.get("executable")
            if isinstance(command, str) and not WINDOWS_EXECUTABLE.fullmatch(command):
                _blocked("raw Windows command line; pass arguments as a sequence")
            if isinstance(command, (tuple, list)) and command:
                # Validate Python script paths before CPython flattens argv.
                _check_program(command[0] if executable is None else executable, command, child_env)
        bound.arguments["env"] = child_env
        real_init(*bound.args, **bound.kwargs)

    subprocess.Popen.__init__ = guarded_init


def set_integration(kind):
    """Permit the relevant external executable around an opt-in integration case."""
    global _integration
    _integration = kind
