# Real Blender checks

These checks run in a separate background process using tiny generated models.
They do not load model weights or interact with an existing Blender window.

Use Blender 5.2.2 and run from the repository root:

```sh
blender --background --factory-startup --threads 2 --python-exit-code 1 \
  --python tests/real_tools/blender/check.py \
  -- --output real-tool-results/blender
```

On macOS, Blender's executable is usually
`/Applications/Blender.app/Contents/MacOS/Blender`. On Linux without a display,
prefix the command with `LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a`.
Blender supplies all Python dependencies for this test.

The test imports the production scene worker and checks:

- A rendered 64-by-64 image contains a visible model and background.
- Normal repair exports a model that reopens with exactly the intended geometry
  and outward-facing triangles. Objects in other scenes must not leak into it.
- The original scene, camera, objects and active selection survive successful
  work, a failed import, a render write failure and an export write failure.
- Reusing the private render scene removes its old objects and collections.

Images, input and output models, and a JSON result are saved in
`real-tool-results/blender/`. A Python assertion or exception fails the Blender
process through `--python-exit-code 1`.
