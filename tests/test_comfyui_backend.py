from pathlib import Path

from image_to_3dlab.comfyui_backend import ComfyUIClient, find_asset, patch_input_image


class _Body:
    def raise_for_status(self):
        return None

    def iter_content(self, _size):
        yield b"new-mesh"


class _Session:
    def get(self, *args, **kwargs):
        return _Body()


def _client() -> ComfyUIClient:
    return ComfyUIClient("http://127.0.0.1:8188", session=_Session())


def test_patch_auto_detected_load_image():
    workflow = {"7": {"class_type": "LoadImage", "inputs": {"image": "old.png"}}}
    assert patch_input_image(workflow, "incoming/new.png", None) == "7"
    assert workflow["7"]["inputs"]["image"] == "incoming/new.png"


def test_find_nested_glb_output():
    entry = {
        "outputs": {
            "42": {
                "meshes": [
                    {"filename": "model.glb", "subfolder": "hunyuan", "type": "output"}
                ]
            }
        }
    }
    assert find_asset(entry)["filename"] == "model.glb"


def test_download_leaves_an_existing_mesh_alone(tmp_path, monkeypatch):
    existing = tmp_path / "model.glb"
    existing.write_bytes(b"keep me")
    opened = []
    real_open = Path.open

    def spy(self, mode="r", *args, **kwargs):
        opened.append((Path(self), mode))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    destination = _client()._download({"filename": "model.glb"}, tmp_path)

    assert existing.read_bytes() == b"keep me"
    assert destination.resolve() != existing.resolve()
    assert destination.parent == tmp_path.resolve()
    assert destination.read_bytes() == b"new-mesh"
    assert not any(
        path.resolve() == existing.resolve() and mode == "wb" for path, mode in opened
    )


def test_two_downloads_do_not_share_a_file(tmp_path):
    client = _client()
    first = client._download({"filename": "model.glb"}, tmp_path)
    second = client._download({"filename": "model.glb"}, tmp_path)

    assert first != second
    assert first.read_bytes() == b"new-mesh"
    assert second.read_bytes() == b"new-mesh"
