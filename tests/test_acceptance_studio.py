"""Follow a model upload from the stdio MCP server into the Studio library."""

from __future__ import annotations

import asyncio
import functools
import json
import sys
import threading
from contextlib import asynccontextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import anyio
import backend_catalog
import finish_api
import generate_api
import pytest
import requests
import trimesh
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from image_to_3dlab import processes

REPO = Path(__file__).resolve().parents[1]
CHILD = Path(__file__).with_name("acceptance_studio") / "finish_child.py"
TIMEOUT = 15
PROGRESS = "Retopologising to 40,000 faces"


@pytest.fixture
def studio(tmp_path, monkeypatch):
    root = tmp_path / "Studio café"
    root.mkdir()
    output = root / "output"
    manager = finish_api.FinishJobManager(output / "finish", output / "generated")
    monkeypatch.setattr(generate_api, "OUTPUT_ROOT", output)
    monkeypatch.setattr(finish_api, "OUTPUT_ROOT", manager.output_root)
    monkeypatch.setattr(generate_api, "FINISH_JOBS", manager)
    monkeypatch.setattr(generate_api, "run_finish_job", functools.partial(finish_api.run_job, manager=manager))
    monkeypatch.setattr(generate_api, "find_blender", lambda: root / "fixture Blender")
    monkeypatch.setattr(generate_api, "pixal3d_front_back_ready", lambda: (False, None))
    monkeypatch.setattr(generate_api.image_api, "BINARY", root / "missing image engine")

    # Keep the shipped catalog and its sizes. Present a supported host with an empty cache.
    monkeypatch.setattr(backend_catalog, "host_platform", lambda: backend_catalog.APPLE)
    monkeypatch.setattr(backend_catalog.Backend, "probes_for", lambda self, host=None: (root / self.id,))
    monkeypatch.setattr(backend_catalog.WeightSet, "state", lambda self: (False, 0, None))
    monkeypatch.setenv("I3D_NO_USAGE", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")

    # Track attempted setup requests before dispatch so the consent test can catch them.
    # Refuse them in this isolated Studio to keep the real installer's writes out of the test.
    setup_attempts = []

    class IsolatedHandler(generate_api.Handler):
        _chain_runner = None
        studio_token = "fixture-key"

        def _route_post(self):
            if self._path_parts()[:2] == ["api", "setup"]:
                setup_attempts.append(self.path)
                self._send_json(500, {"error": "Unexpected setup request in the offline Studio"})
                return
            super()._route_post()

    # The test's only worker command copies the uploaded GLB. Its gate keeps the job alive
    # until the test checks the result endpoint or cancels it.
    gates = {}

    def finish_command(job, settings, *, resume):
        assert not resume
        ready = job.directory / "child ready"
        release = job.directory / "release child"
        gates[job.id] = ready, release
        mode = "cancel" if job.directory.name.startswith("cancel-") else "complete"
        return [sys.executable, str(CHILD), str(job.asset_path), str(job.result_glb),
                str(ready), str(release), mode]

    monkeypatch.setattr(finish_api, "build_command", finish_command)
    with ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(IsolatedHandler, directory=str(root))) as server:
        thread = threading.Thread(target=server.serve_forever, name="acceptance-studio", daemon=True)
        thread.start()
        try:
            yield root, f"http://127.0.0.1:{server.server_port}", manager, gates, setup_attempts
        finally:
            try:
                cleanup_studio(IsolatedHandler._chain_runner, manager, gates)
            finally:
                server.shutdown()
                thread.join(TIMEOUT)
                assert not thread.is_alive()


def cleanup_studio(runner, manager, gates):
    if runner is not None and (chain := runner.running()) is not None:
        runner.cancel(chain.id)
    for _, release in gates.copy().values():
        release.touch()

    names = {f"finish-{job_id[:8]}" for job_id in manager.jobs.copy()}
    if runner is not None:
        names.update(f"chain-{chain_id[:8]}" for chain_id in runner.chains.copy())
    workers = [worker for worker in threading.enumerate() if worker.name in names]
    for worker in workers:
        worker.join(TIMEOUT)

    leaks = []
    for job in manager.jobs.copy().values():
        if job.process is not None and job.process.returncode is None:
            warning = processes.stop_and_reap(job.process, grace=2, force_grace=2)
            leaks.append((job.process.pid, warning))
    assert not leaks, f"Studio left these Finish children for test cleanup: {leaks}"
    assert not [worker.name for worker in workers if worker.is_alive()]


@asynccontextmanager
async def mcp_client(root, url):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "image_to_3dlab.mcp_server"],
        cwd=root,
        env={
            "ASSETFURNACE_URL": url,
            "I3D_STUDIO_TOKEN": "fixture-key",
            "PYTHONPATH": str(REPO),
            "PYTHONUTF8": "1",
            "NO_PROXY": "127.0.0.1",
            "HOME": str(root),
            "USERPROFILE": str(root),
            "HF_HOME": str(root / "empty cache"),
        },
    )
    async with stdio_client(params) as (receive, send), ClientSession(receive, send) as session:
        await session.initialize()
        yield session


async def tool(session, name, **arguments):
    response = await session.call_tool(name, arguments)
    assert not response.is_error, response
    assert len(response.content) == 1, response
    payload = json.loads(response.content[0].text)
    assert "error" not in payload or "status" in payload, payload
    return payload


async def until(check):
    with anyio.fail_after(TIMEOUT):
        while not (result := await check()):
            await anyio.sleep(0.1)
        return result


async def chain_status(session, run_id, *, message=None):
    async def check():
        status = await tool(session, "run_status", run_id=run_id, wait_seconds=0)
        if message is None:
            return status if status["status"] != "running" else None
        assert status["status"] == "running", status
        return status if status.get("message") == message else None

    return await until(check)


async def upload(session, root, name):
    model = root / name
    trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]).export(model)
    accepted = await tool(session, "create", want="prop", model_path=str(model),
                          finish_upload=True, front_and_back=False)
    assert accepted["steps"] == ["finished"], accepted
    return accepted["run_id"], model


async def finish_started(session, run_id, manager, gates):
    await chain_status(session, run_id, message=PROGRESS)
    job = manager.jobs[manager.active]
    ready, release = gates[job.id]

    async def child_ready():
        return ready.is_file()

    await until(child_ready)
    assert job.process is not None and job.process.returncode is None
    assert job.directory.joinpath("pid").is_file()
    return job, release


def test_stdio_finish_upload_progress_library_and_cancel(studio):
    root, url, manager, gates, _ = studio

    async def exercise():
        with requests.Session() as http:
            http.trust_env = False
            http.headers["Authorization"] = "Bearer fixture-key"
            async with mcp_client(root, url) as session:
                run_id, model = await upload(session, root, "mesh café.glb")
                job, release = await finish_started(session, run_id, manager, gates)
                result_url = f"/api/finish/{job.id}/result.glb"
                assert job.result_glb.read_bytes() == model.read_bytes()
                assert http.get(url + result_url, timeout=5).status_code == 404

                release.touch()
                completed = await chain_status(session, run_id)
                assert completed["status"] == "done", completed
                assert completed["done"] == ["finished"]
                asset_id = completed["asset_id"]
                assets = await tool(session, "list_assets")
                asset = next(asset for asset in assets["assets"] if asset["id"] == asset_id)
                assert asset["files"]["finished"] == completed["made"]["finished"]
                info = await tool(session, "model_info", asset_id=asset_id, stage="finished")
                assert Path(info["path"]) == job.result_glb.resolve()
                assert info["stats"]["triangles"] == 1
                assert info["stats"]["vertices"] == 3
                response = http.get(url + result_url, timeout=5)
                assert response.status_code == 200
                assert response.content == model.read_bytes()
                assert job.process.returncode == 0

                run_id, _ = await upload(session, root, "cancel café.glb")
                cancelled_job, _ = await finish_started(session, run_id, manager, gates)
                assert (await tool(session, "cancel_run", run_id=run_id))["cancelled"]
                cancelled = await chain_status(session, run_id)
                assert cancelled["status"] == "cancelled", cancelled
                assert "made" not in cancelled and "asset_id" not in cancelled
                assert cancelled_job.process.returncode not in {0, None}
                assert not processes.process_alive(cancelled_job.process.pid)
                assert not cancelled_job.directory.joinpath("pid").exists()
                assert not cancelled_job.result_glb.exists()
                assert http.get(f"{url}/api/finish/{cancelled_job.id}/result.glb", timeout=5).status_code == 404
                assert (await tool(session, "list_assets"))["assets"] == assets["assets"]

                next_id, _ = await upload(session, root, "retry café.glb")
                next_job, release = await finish_started(session, next_id, manager, gates)
                release.touch()
                following = await chain_status(session, next_id)
                assert following["status"] == "done", following
                assert following["asset_id"] != asset_id
                assert next_job.process.returncode == 0
                assert manager.active is None

    async def bounded():
        with anyio.fail_after(3 * TIMEOUT):
            await exercise()

    asyncio.run(bounded())


def test_stdio_install_plan_names_download_without_consent(studio):
    root, url, manager, _, setup_attempts = studio

    async def exercise():
        with anyio.fail_after(TIMEOUT):
            async with mcp_client(root, url) as session:
                plan = await tool(session, "install_backend", backend_id="sf3d")
        assert plan["backend"] == "sf3d"
        assert plan["label"] == "Stable Fast 3D"
        assert plan["route"] == backend_catalog.BY_ID["sf3d"].best_for
        assert plan["total_size"] == backend_catalog.human_bytes(backend_catalog.BY_ID["sf3d"].bytes_expected)
        assert plan["downloads"] and all(item["size"] for item in plan["downloads"])
        assert "confirm=true" in plan["to_start"]
        assert "started" not in plan
        assert not setup_attempts
        assert not manager.jobs

    asyncio.run(exercise())
