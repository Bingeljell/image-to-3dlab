from image_to_3dlab.comfyui_backend import find_asset, patch_input_image


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
