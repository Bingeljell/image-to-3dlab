# Hunyuan3D workflow

Put a ComfyUI workflow exported with **Workflow > Export (API)** in this directory.
The exact graph is deliberately not pinned here: node names and model files differ between
ComfyUI-3D-Pack and Hunyuan3D wrappers, while the API graph contract is stable.

The workflow must:

1. contain a `LoadImage` node;
2. run Hunyuan shape generation and, if desired, its paint stage; and
3. save/export a `.glb`, `.gltf`, `.obj`, `.fbx`, `.ply`, `.stl`, or `.zip` in an output node.

If there are multiple `LoadImage` nodes, pass `--image-node ID`. If output detection is
ambiguous, pass `--output-node ID`.

