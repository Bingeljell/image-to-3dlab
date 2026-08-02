from __future__ import annotations

import json
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urljoin

ASSET_EXTENSIONS = {".glb", ".gltf", ".obj", ".fbx", ".ply", ".stl", ".zip"}


def patch_input_image(
    workflow: dict[str, Any], filename: str, node_id: str | None
) -> str:
    if node_id is None:
        candidates = [
            key
            for key, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"workflow has {len(candidates)} LoadImage nodes; pass --image-node with one of: "
                f"{', '.join(candidates) or '(none)'}"
            )
        node_id = candidates[0]
    try:
        inputs = workflow[node_id]["inputs"]
    except KeyError as exc:
        raise ValueError(f"image node {node_id!r} is missing or has no inputs") from exc
    if "image" not in inputs:
        raise ValueError(f"node {node_id!r} has no 'image' input")
    inputs["image"] = filename
    return node_id


def _walk_file_records(value: Any) -> Iterator[dict[str, str]]:
    if isinstance(value, dict):
        if isinstance(value.get("filename"), str):
            yield value
        for child in value.values():
            yield from _walk_file_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_file_records(child)


def find_asset(
    history_entry: dict[str, Any], output_node: str | None = None
) -> dict[str, str]:
    outputs = history_entry.get("outputs", {})
    if output_node is not None:
        if output_node not in outputs:
            raise RuntimeError(f"output node {output_node!r} did not produce an output")
        outputs = {output_node: outputs[output_node]}
    records = list(_walk_file_records(outputs))
    assets = [
        r for r in records if Path(r["filename"]).suffix.lower() in ASSET_EXTENSIONS
    ]
    if not assets:
        names = ", ".join(r["filename"] for r in records) or "none"
        raise RuntimeError(
            f"workflow completed but returned no 3D asset (files found: {names})"
        )
    return assets[0]


class ComfyUIClient:
    def __init__(self, base_url: str, session: Any | None = None):
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "quality mode requires requests; install requirements.txt"
            ) from exc
        self._requests = requests
        self.base_url = base_url.rstrip("/") + "/"
        self.session = session or requests.Session()

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    def _check_server(self) -> None:
        try:
            response = self.session.get(self._url("system_stats"), timeout=10)
            response.raise_for_status()
        except self._requests.RequestException as exc:
            raise RuntimeError(
                f"ComfyUI is not reachable at {self.base_url}: {exc}"
            ) from exc

    def _upload(self, image: Path) -> str:
        mime = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
        remote_name = f"{uuid.uuid4().hex}_{image.name}"
        with image.open("rb") as handle:
            try:
                response = self.session.post(
                    self._url("upload/image"),
                    files={"image": (remote_name, handle, mime)},
                    data={
                        "type": "input",
                        "subfolder": "image-to-3dlab",
                        "overwrite": "false",
                    },
                    timeout=120,
                )
                response.raise_for_status()
            except self._requests.RequestException as exc:
                raise RuntimeError(
                    f"could not upload input image to ComfyUI: {exc}"
                ) from exc
        payload = response.json()
        name = payload.get("name", remote_name)
        subfolder = payload.get("subfolder", "")
        return f"{subfolder}/{name}".lstrip("/") if subfolder else name

    def _wait(
        self, prompt_id: str, timeout: float, poll_interval: float
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                response = self.session.get(
                    self._url(f"history/{prompt_id}"), timeout=30
                )
                response.raise_for_status()
            except self._requests.RequestException as exc:
                raise RuntimeError(
                    f"could not read ComfyUI job history: {exc}"
                ) from exc
            payload = response.json()
            if prompt_id in payload:
                entry = payload[prompt_id]
                status = entry.get("status", {})
                if (
                    status.get("status_str") == "error"
                    or status.get("completed") is False
                ):
                    messages = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI workflow failed: {messages}")
                return entry
            time.sleep(poll_interval)
        raise RuntimeError(f"ComfyUI job {prompt_id} timed out after {timeout:g}s")

    def _download(self, record: dict[str, str], output_dir: Path) -> Path:
        params = {
            "filename": record["filename"],
            "subfolder": record.get("subfolder", ""),
            "type": record.get("type", "output"),
        }
        try:
            response = self.session.get(
                self._url("view"), params=params, timeout=300, stream=True
            )
            response.raise_for_status()
        except self._requests.RequestException as exc:
            raise RuntimeError(f"could not download ComfyUI output: {exc}") from exc
        destination = output_dir.resolve() / Path(record["filename"]).name
        with destination.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
        return destination

    def generate(
        self,
        image: Path,
        workflow_path: Path,
        output_dir: Path,
        image_node: str | None,
        output_node: str | None,
        timeout: float,
        poll_interval: float,
    ) -> Path:
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("--timeout and --poll-interval must be positive")
        try:
            workflow = json.loads(workflow_path.expanduser().read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read workflow {workflow_path}: {exc}") from exc
        if not isinstance(workflow, dict):
            raise ValueError(
                "workflow JSON must be an API-format object, not a UI workflow array"
            )

        self._check_server()
        uploaded_name = self._upload(image)
        patch_input_image(workflow, uploaded_name, image_node)
        try:
            response = self.session.post(
                self._url("prompt"),
                json={"prompt": workflow, "client_id": uuid.uuid4().hex},
                timeout=30,
            )
            response.raise_for_status()
        except self._requests.RequestException as exc:
            raise RuntimeError(f"could not queue ComfyUI workflow: {exc}") from exc
        payload = response.json()
        if payload.get("node_errors"):
            raise RuntimeError(
                f"ComfyUI rejected the workflow: {payload['node_errors']}"
            )
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI returned no prompt_id: {payload}")
        entry = self._wait(prompt_id, timeout, poll_interval)
        return self._download(find_asset(entry, output_node), output_dir)
