# Images in docs

Screenshots and renders that a doc refers to. Kept here rather than in a scratch folder so
a reader months later sees what the text is talking about.

**This repository is deliberately slim** (it was cut from 244 MB to 8.5 MB in September
2026), so images earn their place the same way scripts do:

- Only add an image that makes a claim checkable. A comparison, a defect, a before and
  after. Not decoration.
- **JPEG, at most ~1600px on the long edge, quality around 65.** That lands a three-panel
  viewer comparison near 200-300 KB. Never commit a raw retina PNG screenshot; they run to
  several megabytes each.
- Name by what is being compared, not by the asset or the date:
  `attention-backends-same-seed.jpg`, not `fox-screenshot-2.png`. The asset is an example;
  the comparison is the point.
- macOS screenshot temp folders are cleaned aggressively. Save into this directory
  directly, or the file will be gone within the hour.

Convert with `sips`:

```
sips -Z 1600 -s format jpeg -s formatOptions 65 input.png --out docs/images/name.jpg
```

## Exceptions to "JPEG only"

Two files here are not doc screenshots and follow different rules.

- **`turntable-*.webp`** — an animated WebP in the README. A still cannot show that the
  result is a real model rather than a flattering angle, which is the one thing a reader
  arriving from a link wants to know. Budget **under ~400 KB**: 360px, every 5th frame of
  a 12s turntable (60 frames), quality ~42. Build it with Pillow, not `ffmpeg` — the
  Homebrew build here has no WebP encoder:

  ```python
  frames[0].save(out, save_all=True, append_images=frames[1:],
                 duration=200, loop=0, quality=42, method=6)
  ```

- **`hero-sentence-to-moving-character.webp`** — the README's top picture: a sentence typed,
  then the picture, model, skeleton and moves. Cut from the 0.4.0 reveal video, first 15 s,
  432px wide, 10 fps, quality 50 (~0.7 MB), built with Pillow as above.

- **`social-preview.jpg`** — 1280x640, the card shown when the repo is linked on X, Slack
  or Discord. **It is not referenced by any page.** GitHub only uses it once uploaded under
  *Settings → General → Social preview*; without that, a shared link shows the owner's
  avatar. Kept in the tree so it is versioned and rebuildable.
