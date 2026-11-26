# Real Blender and browser tests

These checks run the actual applications with tiny generated models. They are
separate from the offline unit suite and its coverage totals. They never load
model weights or use an existing Blender window or Studio server.

Both jobs run on pull requests, merge-queue checks and pushes to `main`. The
Merge gate requires them to succeed, along with the unit and coverage jobs.
GitHub still needs an administrator to require that gate in the main ruleset.

## Blender

CI installs Blender 5.2.2 with the community
[setup-blender action](https://github.com/BradyAJohnston/setup-blender), pinned
to a specific action commit. It uses Ubuntu, Xvfb and software graphics.
Locally, use the same Blender version and run from the repository root:

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
- Reusing a private render scene clears its own links and preserves objects,
  collections and references shared with other scenes.
- Failed model, marker and rig replacements preserve previous work.
- Staging normalizes models with transformed roots; baking restores camera,
  frame and output settings after success and failure.
- Animation replacement preserves other rigs and saved clips. Short clips retain
  distinct poses and positive duration after export and re-import.
- Two region bakes retain separate textures, and the production preview launcher
  renders animation endpoints in a separate Blender child process.

Images, input and output models, and a JSON result are saved in
`real-tool-results/blender/`. A Python assertion or exception fails the Blender
process through `--python-exit-code 1`.

## Studio browser

Install the project dependencies and the pinned Playwright browser:

```sh
uv sync --frozen
npm ci
npx playwright install --with-deps chromium
npm run test:browser
```

Use the Node version in `.node-version`. The test uses `.venv/bin/python` on
Unix or `.venv/Scripts/python.exe` on Windows; set `PYTHON` to override it.

The test starts the shipped Studio HTTP handler on a private loopback port,
with a temporary library containing a generated magenta triangle. It runs
the shipped JavaScript and bundled Three.js in real Chromium. The fixture
refuses write requests; this suite checks library loading and viewing, not
generation or installation.

The checks require a visible loading state, a successful model request,
working WebGL, and actual magenta pixels in the rendered canvas. Switching
to wireframe must visibly reduce the filled area. Opening a malformed local
model must show the Studio's error message. Unexpected browser errors or
network requests fail the test. The suite also opens the shipped winding-demo
page using its import map, supplies pinned Three.js modules locally, and checks
that winding can be reversed. A model with multiple saved animations must wait
for an explicit move choice and play the selected clip.

Chromium explicitly uses SwiftShader, a CPU graphics implementation, for this
trusted local test content. Missing WebGL fails the test rather than skipping
it. Screenshots, a trace and an HTML report are saved under
`real-tool-results/browser/`.

## What these checks establish

They verify real rendering and application behavior on small fixtures. They
do not establish graphics-driver compatibility on every device, model-generation
quality, GPU backend correctness, or acceptable performance on large assets.
Those remain separate release checks.

GitHub uploads the real-tool results even when a test fails. Inspect the job
log when installation or startup fails before artifacts can be written.
