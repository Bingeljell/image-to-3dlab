---
format: 1920x1080
duration: 29s
mode: autonomous
status: outline
message: Turn a single image into a local, inspectable, reproducible 3D asset.
arc: Future Pacing → Demo Loop → Proof → Capability Cascade → CTA
audience: technical artists, indie game developers, and local AI/3D builders
---

# Image to 3D Lab — visual treatment

## Video direction

Warm-editorial developer-tool film. The visual language is cream paper, warm tile, ink, and one restrained terracotta signal accent, with warm-navy code surfaces reserved for technical proof. EB Garamond carries the main statement; Inter carries explanatory copy; JetBrains Mono carries the index, backend labels, and provenance fields.

The three approved clips are the visual spine: the moss fox supplies the living, labelled-wind experiment; Nikita’s turntable supplies the inspectable 3D result; Nikita’s cheer supplies the early rigging/animation proof. The source images appear only where they clarify the image-to-asset transformation. There is no fake product UI and no claim that the output is finished or game-ready.

Motion is restrained but legible: image scan, panel lift, turntable hold, capability cards entering on beat, and a final lockup. Every experimental lane is explicitly marked `EXPERIMENTAL`. Music is `launch-bed__v01.wav`; the short generated SFX are used as scan, model-lock, and provenance punctuation.

## Frame plan

### Frame 01 — “One image, locally”

- **status:** outline
- **time:** 00:00–00:03
- **duration:** 3s
- **transition_in:** scan-line reveal from a quiet cream ground
- **scene:** Hook / Future Pacing
- **voiceover:** What if one image could become a real 3D asset — locally?
- **poster:** Source fox image on a tile card, with a second dark technical card beginning to resolve as a model silhouette.
- **type:** concept-demo
- **persuasion:** Future Pacing
- **blueprint:** concept-demo-decode-pan
- **asset_candidates:** `assets/moss-fox-alpha_big.png`, `assets/nikita_holding_beer.png`
- **focal:** the fox source image moving from reference to “3D asset” language
- **roles:** hero image, scan marker, opening kicker, caption
- **sfx:** `assets/audio/image-to-3d-lab/scan-whoosh/scan-whoosh__v01.wav` at 00:00.15

The opening should feel like a research note becoming an interface: one large image, one exact promise, one coral spike. The image gets a thin measurement frame and a small `INPUT / IMAGE` label; the right-hand panel is a graphic suggestion of depth, not a fabricated render.

### Frame 02 — “Three local routes”

- **status:** outline
- **time:** 00:03–00:08
- **duration:** 5s
- **transition_in:** image card compresses into the left rail; route cards step in from the right
- **scene:** Product promise / Demo Loop
- **voiceover:** Image to 3D Lab turns a reference into a textured model, with three local backends.
- **poster:** Nikita source image beside three numbered backend lanes and a compact manifest strip.
- **type:** workflow
- **persuasion:** Demo Loop
- **blueprint:** messaging-multi-phrase
- **asset_candidates:** `assets/nikita_holding_beer.png`, `assets/nikita_hero.provenance.json`
- **focal:** “reference → textured model” as the dominant reading path
- **roles:** source card, three route cards, manifest chrome, caption
- **sfx:** `assets/audio/image-to-3d-lab/model-lock/model-lock__v01.wav` at 00:03.20

Use three simple route labels rather than inventing backend names in the film: `LOCAL ROUTE / 01`, `LOCAL ROUTE / 02`, and `LOCAL ROUTE / 03`. The route cards should read as selectable pathways, not as a dashboard. The manifest strip reinforces that the result can be repeated.

### Frame 03 — “Proof beside the result”

- **status:** outline
- **time:** 00:08–00:14
- **duration:** 6s
- **transition_in:** a hard editorial cut into the turntable, followed by a slow hold
- **scene:** Proof / Reproducibility
- **voiceover:** Preview it in Blender. Re-run from a manifest. Keep hashes, settings, and license provenance beside every result.
- **poster:** Nikita turntable video on the left; provenance receipt and `GLB / RE-RUN / HASHED` code surface on the right.
- **type:** proof
- **persuasion:** Proof
- **blueprint:** metric-video-text-pivot
- **asset_candidates:** `assets/nikita_turntable.mp4`, `assets/nikita_hero.provenance.json`, `assets/nikita_hero.glb`
- **focal:** real turntable motion paired with legible provenance fields
- **roles:** video proof, code surface, receipt stamp, caption
- **sfx:** `assets/audio/image-to-3d-lab/provenance-stamp/provenance-stamp__v01.wav` at 00:08.65

This is the trust beat. Keep the turntable large enough to read as a real 3D result, while the receipt is dense enough to communicate reproducibility without pretending to be a full application screenshot. Highlight `hash`, `settings`, and `license` one at a time.

### Frame 04 — “The lab opens up”

- **status:** outline
- **time:** 00:14–00:21
- **duration:** 7s
- **transition_in:** proof card splits into a three-card capability cascade
- **scene:** Capability cascade / Experimental lanes
- **voiceover:** Then go further: multiple views, labeled parts for wind, and early rigging experiments.
- **poster:** Moss fox wind clip, Nikita cheer clip, and a third tile with multi-view / labelled-part notation.
- **type:** comparison
- **persuasion:** Capability Cascade
- **blueprint:** comparison-split-cards
- **asset_candidates:** `assets/moss_fox_wind.mp4`, `assets/nikita_cheers.mp4`, `assets/moss-fox-alpha_big.png`
- **focal:** real motion in the two video cards; experimental badge on each extension lane
- **roles:** video cards, experimental tags, multi-view notation, caption
- **sfx:** none; let the music carry the widening

The cards should arrive in sequence, not as a simultaneous grid dump. The moss fox card gets `LABELLED WIND / EXPERIMENTAL`; the Nikita card gets `RIGGING / EXPERIMENTAL`; the third card gets `MULTI-VIEW / EXPERIMENTAL`. That language keeps the ambition clear without overclaiming.

### Frame 05 — “Inspect. Iterate. Build.”

- **status:** outline
- **time:** 00:21–00:29
- **duration:** 8s
- **transition_in:** capability cards collapse into a single hero plane
- **scene:** Close / CTA
- **voiceover:** From pixels to something you can inspect, iterate, and build with. Image to 3D Lab.
- **poster:** Nikita cheer/turntable hero on a warm-navy field with the product lockup and a small “local / reproducible / inspectable” index.
- **type:** cta
- **persuasion:** CTA
- **blueprint:** cta-orbit-collapse
- **asset_candidates:** `assets/nikita_cheers.mp4`, `assets/nikita_turntable.mp4`, `assets/nikita_hero.provenance.json`
- **focal:** the final lockup and the real animated model behind it
- **roles:** hero video, product name, three-word value line, coral CTA mark
- **sfx:** none; allow a 1.2s musical tail after the lockup resolves

Close with a single terracotta voltage moment: the product name or a thin CTA rule, not both. The final frame should hold long enough for the viewer to remember the name and the three-word value line.

## Audio map

- `assets/audio/image-to-3d-lab/launch-bed/launch-bed__v01.wav` — 29s, medium music model, full-bed ducked under captions.
- `assets/audio/image-to-3d-lab/scan-whoosh/scan-whoosh__v01.wav` — 1s, small SFX model, frame 01.
- `assets/audio/image-to-3d-lab/model-lock/model-lock__v01.wav` — 1s, small SFX model, frame 02.
- `assets/audio/image-to-3d-lab/provenance-stamp/provenance-stamp__v01.wav` — 1s, small SFX model, frame 03.
