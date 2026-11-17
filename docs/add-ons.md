# Add-ons

An add-on is a small repo that adds a new tool to AssetFurnace. It lives in its own
Git repository, is pinned to an exact commit, and runs on your own machine.

This page is for people who want to **write** an add-on. If you just want to use one,
see [`docs/agents.md`](agents.md) or the Add-ons page in the studio.

## What a v1 add-on can do

Schema v1 add-ons can make a 3D model from inputs you provide. You define one or more
tools in `addon.json`; each tool runs a script you ship, and AssetFurnace writes the
result and a `.provenance.json` sidecar into your `output/` folder.

Add-ons do **not** hook into Rig, Animate or Finish. If your tool produces a raw GLB,
the user can carry it on with the studio's usual steps themselves.

## Repo layout

```
pedestal-addon/
  addon.json       # manifest: who made it, what it does, what it needs
  pedestal.py      # the script that builds the model
  README.md        # what it does and how to use it
  tests/           # your own tests, run however you like
```

Only `addon.json` is required by AssetFurnace. The script name and every other file are
up to you.

## `addon.json` field by field

```json
{
  "schema": 1,
  "id": "pedestal",
  "name": "Pedestal",
  "version": "0.1.0",
  "author": {
    "name": "Your name",
    "url": "https://github.com/you"
  },
  "licence": "Apache-2.0",
  "summary": "A round display stand with height, width and step count.",
  "needs": {
    "assetfurnace": ">=0.6.0",
    "blender": ">=4.2"
  },
  "setup": ["python3", "-c", "print('setup ok')"],
  "models": [],
  "tools": [
    {
      "id": "build",
      "label": "Build pedestal",
      "hint": "A round stepped stand for showing off a model.",
      "inputs": [
        {
          "id": "height",
          "type": "number",
          "label": "Height (m)",
          "min": 0.1,
          "max": 2.0
        },
        {
          "id": "width",
          "type": "number",
          "label": "Width (m)",
          "min": 0.1,
          "max": 2.0
        },
        {
          "id": "steps",
          "type": "number",
          "label": "Steps",
          "min": 1,
          "max": 10
        }
      ],
      "run": ["{blender}", "-b", "-P", "pedestal.py", "--", "{out}", "--height", "{height}", "--width", "{width}", "--steps", "{steps}"],
      "makes": "glb"
    }
  ]
}
```

### Required top-level fields

- `schema`: must be `1`.
- `id`: lowercase letters, numbers, hyphens and underscores only. Must start with a
  letter or number. This becomes the folder name in `addons/` and the run folder name
  in `output/`.
- `name`: the human name shown in the studio and CLI.
- `version`: any string you like; shown to users but not parsed.
- `author`: an object with at least `name`. `url` is optional and shown as a link.
- `licence`: the SPDX name or short name of the licence for the add-on itself.
- `summary`: one sentence describing what the add-on does.

### Optional top-level fields

- `type`: `"tool"` (default) or `"app"`. Tools make models; apps are reserved for future
  use and behave the same as tools today.
- `runs`: `"local"` (default) or `"cloud"`. Say `"cloud"` if the add-on sends the user's models or
  pictures to a service off their machine. Users see a Cloud badge and a plain warning before they install.
  Leave it out and it counts as local, so declaring it honestly is on you.
- `needs`: a map of requirements shown to the user before install. Keys are free-form
  strings such as `"assetfurnace"` or `"blender"`; values are free-form version hints.
- `setup`: a command list run once when the add-on is installed. Use it to fetch assets
  that are not in Git, compile something, or verify dependencies. It runs with
  `HF_HUB_OFFLINE=1` so it cannot download Hugging Face weights. Use portable interpreter
  names such as `python3` rather than `python`, because the name on PATH varies by machine.
- `models`: a list of model files the add-on loads. Each entry should have `name`,
  `licence` and `size`. These are shown in the install plan so the user knows what they
  are agreeing to before anything is downloaded.

### Tools

`tools` is a list. Each tool needs:

- `id`: a short identifier for this tool. Lowercase letters, numbers, hyphens and
  underscores.
- `label`: the button text in the studio.
- `hint`: one sentence saying what the tool does.
- `inputs`: a list of inputs the user can provide.
- `run`: the command to run, as a list of strings.
- `makes`: what the tool produces. Use `"glb"` for a 3D model file.

### Inputs

Each input has:

- `id`: the placeholder name used in `run`.
- `type`: one of `text`, `number`, `choice`, `toggle`, `picture`, `model`.
- `label`: the label shown beside the field.
- `optional`: set to `true` if the user can leave it blank.

Type-specific fields:

- `number`: `min` and `max` are optional but recommended.
- `choice`: `choices` is a required list of allowed values.
- `toggle`: the value is coerced from `true`/`false`, `1`/`0` or `yes`/`no`.
- `picture` / `model`: the value is a file path. AssetFurnace resolves it relative to the
  current working directory and checks that the file exists before running.

## Placeholders in `run`

AssetFurnace replaces these before calling your script:

- `{out}`: the output folder for this run. Write your result file here.
- `{blender}`: the path to the Blender executable on this machine.
- `{<input-id>}`: the value of the named input, already coerced to the right type.

Any other argument is passed through unchanged.

A small working pattern for a Blender script: clear the scene first, build your mesh,
then export to `{out}/{run-stem}.glb`. Use `bpy.data.meshes.remove(mesh)` and
`bpy.data.materials.remove(material)` to clean up; `bpy.data.block_remove` does not exist
in every Blender version.

## What AssetFurnace writes for you

After a run, AssetFurnace writes a `.provenance.json` sidecar next to the result file.
It records:

- the add-on id, name, repo URL and pinned commit
- the add-on licence
- the add-on's label when it was installed: Verified, Listed or Unknown
- if a model went in, that model's licence and any restrictions, which the result keeps
- the tool id and what it makes
- the actual input values used
- the models the manifest declared

This is the licence record that travels with the file. Do not write your own.

## Testing locally

You can install an add-on straight from a Git URL to test it. It is marked Unknown, because it is not on our list:

```bash
./lab addon add --unreviewed https://github.com/you/pedestal-addon.git@<commit-hash>
```

The URL must end with `@<40-character commit hash>`. The repo name does not have to match the
add-on id: the id comes from `addon.json`. AssetFurnace clones that exact
commit, checks the manifest, runs `setup`, and asks for confirmation before writing
anything.

To test with a local file instead of a remote URL, put a bare Git repo on disk and point
at it the same way:

```bash
./lab addon add --unreviewed /path/to/pedestal-addon.git@<commit-hash>
```

Use a local list file to test the list flow:

```bash
ADDONS_INDEX=/path/to/index.json ./lab addon list
```

The index file is a JSON list of listing objects. Each listing needs `id`, `name`,
`repo`, `commit`, `author`, `summary`, `type`, and can include `links`, `rating` and
`ratings_count`.

## Rules

- The add-on id must not be a path or contain uppercase letters or dots.
- The commit must be pinned to exactly 40 characters.
- The add-on runs code on the user's machine, so be clear about what it fetches and what
  licence it is under.
- Do not download model weights during `setup` unless the user expects it and the install
  plan names them. `HF_HUB_OFFLINE=1` is set during setup.
- The result file must be a GLB written to `{out}` and named `{run-stem}.glb`, where
  `run-stem` is the run name AssetFurnace chooses.
- Keep the manifest honest: `models`, `licence` and `needs` are shown to the user before
  they say yes.

## Getting listed

The list is a public repo, [AssetFurnace/assetfurnace-addons](https://github.com/AssetFurnace/assetfurnace-addons),
with one small file per add-on. It holds no add-on code, only a pointer to your repo and the exact version.

1. Publish your add-on as a public Git repository with a clear licence.
2. Test it by installing it from your own repo (see above).
3. Fork the list repo and add one file, `addons/<your-id>.json`. Copy `addons/pedestal.json`.
4. Open a pull request. Automatic checks run and tell you what to fix.
5. When the checks pass and it is merged, you are **Listed**.
6. Want **Verified**? Say so in the pull request.

A new version of your add-on is a new pull request that changes the `commit` line. Nothing updates by itself.
The list repo's README has the full rules.

## What the labels mean

Every add-on shows one label, in the studio, the command line, the agent tools and on the site.

- **Verified**: we read the code, ran it and checked its licences. Pinned to that exact version. A new
  version is checked again.
- **Listed**: it passed automatic checks (it installs, runs and makes a valid model). We have not read its
  code. Use it at your own risk.
- **Unknown**: not on our list. We have never looked at it. The studio warns you every time.

Add-ons run code on your machine, so read the label before you install.

## How we review an add-on

Listed is automatic. A person (or an agent working for us) is only involved in Verified.

For Verified we read every script that runs, including `setup`; check that it does not reach the network,
write outside its output folder or read your files beyond the inputs you give it, unless it says so; check
the licence of every model and file it uses; and run it on a few test inputs, including a bad one. We check
again on every new version.

We can remove an add-on from the list at any time, for example if its repo disappears or a reported problem
goes unanswered.
