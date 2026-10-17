# Use AssetFurnace from your agent

Claude Code, Codex, Kimi and other agents can drive AssetFurnace through **MCP** (Model
Context Protocol, the standard way agents plug in tools). Your agent can make pictures, 3D
models and animated characters, look at the results, and open them in the studio for you.
Everything still runs on your own machine.

Works on Mac and Linux; Windows is not tested yet. Install AssetFurnace first
([Install](../README.md#install)). The examples below assume the
default folder, `~/image-to-3dlab`. Change the path if you installed somewhere else.

## Let your agent install it

No AssetFurnace yet? Ask your agent: *"Install AssetFurnace from assetfurnace.com and connect it
to yourself."* These are the steps it should follow (agents: run them as written):

1. `curl -fsSL https://assetfurnace.com/install.sh | bash -s -- --dry-run` and tell the user what
   it will do. It installs the code, Python 3.11 and its packages (roughly 1-2 GB), and **no model weights**.
2. After the user says yes: `curl -fsSL https://assetfurnace.com/install.sh | bash -s -- --yes --no-start`
3. Connect, using the matching line below (for Claude Code:
   `claude mcp add assetfurnace -- ~/image-to-3dlab/lab mcp`).
4. Tell the user to start a new session so the tools load. Then `status` and `list_engines`
   show what to set up next, and `install_backend` asks before downloading any weights.

## Connect your agent

**Claude Code**
```bash
claude mcp add assetfurnace -- ~/image-to-3dlab/lab mcp
```

**Codex**
```bash
codex mcp add assetfurnace -- ~/image-to-3dlab/lab mcp
```
or in `~/.codex/config.toml`:
```toml
[mcp_servers.assetfurnace]
command = "/Users/you/image-to-3dlab/lab"
args = ["mcp"]
```

**Kimi Code CLI**
```bash
kimi mcp add --transport stdio assetfurnace -- ~/image-to-3dlab/lab mcp
```

**Anything else** that reads the common JSON config (Claude Desktop, Cursor and many more):
```json
{ "mcpServers": { "assetfurnace": { "command": "/Users/you/image-to-3dlab/lab", "args": ["mcp"] } } }
```

**Clients that want a web address** (for example Muse Code): run
`~/image-to-3dlab/lab mcp --http` and point the client at `http://127.0.0.1:8779/mcp`. It only
listens on this machine.

The studio does not need to be open: the first tool call starts it in the background.

## What your agent can do

| Tool | What it does |
|---|---|
| `status` | What's installed, and what's running now |
| `list_engines` | Every engine, whether it's ready, its licence and download size |
| `list_moves` | The animation moves (idle, walk, run, wave, ...) |
| `create` | Make something new: a picture, one prop, nine props, or an animated character, from an idea, a picture or a GLB |
| `continue_asset` | Take an existing asset further (picture to 3D, rig it, another move), or make a step again as a new version, such as a full-detail Finish |
| `run_status` | How a run is going (it waits, so the agent isn't spamming) |
| `cancel_run` | Stop a run |
| `list_assets` | Your Library |
| `options` | Every knob, as the studio's Advanced panels show them: what each does, its default and when to try it |
| `list_versions` | Every version of one character: its 3D models, Finishes and rigs, raw model included |
| `preview` | A picture of the asset from four sides, or four moments of one of its moves, so the agent can see it and show you |
| `open_in_studio` | Opens the asset in your browser, so you can spin it and play its moves |
| `model_info` | A model's numbers (triangles, vertices, textures and their sizes, bones, moves, file size) and its full path on your computer |
| `show_in_folder` | Opens the folder holding a model in Finder or Explorer, with the file selected |
| `report_bug` | Drafts a bug report (your words, plus version, machine and the last error) and gives you the links to send it. It never sends anything itself |
| `install_backend` | Sets up an engine, but **only after you say yes**: it first tells the agent what it would download, the size and the licence |

When a result looks off, ask your agent to try a knob: it can call `options`, then redo just
that step as a new version, for example *"redo the Finish with Pixel Match's nudge off"*
becomes `continue_asset(asset_id, up_to="finished", again=True, advanced={"nudge": false})`.

Your agent can't delete, hide or rename assets, or paste your Hugging Face token. Those stay
with you, in the studio.

## Try it

> Make me an animated knight character with a wave, then show me what it looks like.

A character takes a while (roughly 15-25 minutes start to finish on our test Mac; yours may differ). Your
agent will check on it and show you the preview when it's done.

## For contributors

The MCP server (`image_to_3dlab/mcp_server.py`) is a thin client of the studio's own API, so
new engines, moves and prompt wording reach agents on their own. Two tests keep it that way:
`tests/test_agent_plan.py` checks the agent's step rules against the studio's `plan.js`, and
`tests/test_mcp_server.py` fails when the studio gains an action that agents can't reach and
that isn't listed as studio-only.
