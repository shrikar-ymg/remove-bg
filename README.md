---
title: Cutout Studio
emoji: ✂️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Cutout Studio 2.2

A private, local background-removal workspace for high-quality automatic
cutouts and edge-aware corrections. Version 2.2 accelerates the existing
BEN2/BiRefNet/ISNet pipeline automatically and tightens its edge and shadow
finishing. The multi-image workspace, comparison, zoom, preview, and export
tools from 2.0 remain available.

Images and correction data stay on your computer.

## What 2.2 adds

- Hardware acceleration is automatic. The app prefers NVIDIA CUDA when that
  runtime is installed, otherwise DirectML on Windows (NVIDIA, AMD, or Intel),
  AMD ROCm, Apple Core ML, and finally CPU.
- There is no detection button or processor benchmark. The first available
  suitable provider is selected from hardware metadata when a model is loaded
  and is then cached for the session. Known low-power GPU families that run BEN2
  slower than CPU are detected without running a slow inference benchmark.
- Precision mode replaces the full-image closed-form solve with a narrow,
  source-color-aware BEN2 edge pass. This keeps confident model pixels, aligns
  high-contrast contours, and avoids broadly softening the matte.
- The new finisher removes edge color contamination locally and restores detail
  lost while BEN2's fixed 1024px mask is scaled to the source image.
- Optional **Preserve natural shadow** recovers credible soft contact/cast
  shadows as a portable black-alpha layer without baking in the original
  background. It is off by default so normal removal cannot retain background.
- Shadow compositing now works behind weak model-alpha pixels instead of only
  fully transparent pixels, so tinted shadow remnants no longer block recovery.
- Shadow recovery remains conservative on very dark, heavily saturated,
  textured, or border-filling scenes where a single-image estimate is unsafe.
- **Automatic** mode picks the model for each image and is now the default. A
  flat-colour illustration is routed straight to the precision model, which
  measured best or joint-best on every illustration tested. A photograph is
  segmented by two models and the app keeps whichever matte's boundary better
  follows the picture, which is what tells a correct edge from one drawn through
  the middle of a blurred object. On a traffic photograph where the precision
  model washed a second, out-of-focus car into the subject, this cut the stray
  material from 37% to 8% with the subject fully intact, without anyone choosing
  a mode. Photographs therefore run two inferences and take about twice as long;
  illustrations still run one. Where the two scores are within 5% the
  measurement cannot separate them - on a wispy-hair test the leaders were
  within 0.02% - so a near-tie keeps the precision model. The result line
  reports which profile was used, and the four modes remain selectable by hand.
- Comparing two models means loading two of them, so each is released as soon as
  its matte is in hand; the more memory-hungry one runs first, while memory is
  least fragmented. If a model still cannot allocate, it is skipped with a
  warning and the comparison proceeds with the other rather than failing.
- Large images no longer fail with an allocation error. Closed-form matting
  needs about 400 bytes per pixel, so every solve is now bounded by pixel count
  and run on a reduced copy when a region is oversized; GrabCut, used by the
  correction brush, is bounded the same way. A stroke dragged across a
  10-megapixel photograph used to ask for roughly 2 GB in one array and stop
  with "unable to allocate"; it now peaks at about 1.4 GB and finishes in under
  three seconds instead of seventeen. If memory does run out, the request
  reports a capacity problem instead of a raw allocation failure.
- Flat-colour illustrations get their icon and badge fills back. A saliency
  model keeps an icon's outline and glyph but discards the coloured disc between
  them when that disc shares the page's hue. Those discs are enclosed by kept
  pixels, so they are restored, while a soft shadow cast onto the page - which is
  open to the background, never enclosed - is not. A hole whose own colour
  matches the page still shows through, so see-through gaps stay open. The pass
  is gated on a flat, noise-free background and never runs on photographs.

The workspace also includes:

- Add multiple images and process them as a sequential local queue.
- Switch between completed images without losing their edits or history.
- Drag a comparison control to inspect the original against the cutout.
- Fit or zoom the canvas from 4% to 400% for detailed edge work.
- Preview transparency, white, black, or a custom background color.
- Export transparent PNG, lossless WebP, photographic JPG, or embedded PNG SVG.
- Include the selected preview color in PNG, WebP, JPG, or SVG exports.
- Paste an image directly from the clipboard with `Ctrl+V`.
- Use clearer per-file progress, error states, and keyboard shortcuts.

## Removal pipeline

| Mode | Model | Best use |
|---|---|---|
| Precision | BEN2 Base | Hair, fur, glass, soft edges, and difficult subjects |
| Balanced | BiRefNet General Lite | Strong everyday results with a shorter CPU wait |
| Fast | ISNet General | Simple subjects and smooth backgrounds |
| Deep detail | BiRefNet General | A detailed alternative when BEN2 is too selective |

All modes feed a source-color alpha finishing stage. Precision mode uses the
fast color-aware edge finisher and offers optional conservative natural-shadow
recovery. Fast mode alone retains the smooth-background structure-recovery pass.
Very large images are resized to a 12-megapixel processing limit to avoid memory
failures.

## Run

Double-click `start.bat`, or run:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>.

On Windows, `requirements.txt` installs ONNX Runtime DirectML. It uses a
compatible NVIDIA, AMD, or Intel DirectX 12 GPU automatically and falls back to
CPU if no suitable GPU is available. On NVIDIA systems configured with the
CUDA-enabled ONNX Runtime package, CUDA is preferred automatically; no setting
in the app is required. The active or ready backend—and a detected GPU that was
deliberately skipped—appears in the lower-left sidebar and in `GET /api/info`.

The first start may take longer while numerical kernels initialize. Models also
download once and are then cached in `~/.u2net/`: BEN2 Base is roughly 213 MB,
BiRefNet General roughly 1 GB, BiRefNet General Lite roughly 224 MB, and ISNet
roughly 179 MB.

## Use

1. Choose, drop, or paste one or more images.
2. Choose a quality, decide whether to **Preserve natural shadow**, and select
   **Remove background**.
3. Select a completed image from the project queue.
4. Use **Restore** or **Erase** and brush over an area. The app analyzes the
   selection and snaps the correction toward detected object boundaries.
5. Compare against the original, adjust the preview background, and export.

Undo and redo keep up to 12 successful edits per image. Shortcuts include
`Ctrl+Z`, `Ctrl+Shift+Z`, `Ctrl+Y`, `+`, `-`, and `0` to fit the image.

SVG export embeds the edited full-resolution image instead of tracing it, which
preserves photographic detail. JPG cannot store transparency, so a transparent
preview exports on white.

## Local Fast-mode feedback

After correcting a Fast result, select **Save approved Fast correction**. The
app saves the normalized original image, approved alpha mask, and metadata in
`feedback_data/`. The directory is local and ignored by Git. Saving examples
builds a future training set; it does not automatically retrain the model.

## Optional MongoDB storage

With a MongoDB Atlas cluster configured, the app records a metadata entry for
each removal (filename, quality, timings, sizes). **Images are never sent to
MongoDB**; they stay on this machine. Fast-mode feedback is not stored in
MongoDB; it is only saved locally as described above.
Without configuration, or if the cluster is unreachable, the app works exactly
as before.

1. Copy `.env.example` to `.env` (it is ignored by Git).
2. Set `MONGODB_URI` to your Atlas connection string, leaving `<db_password>`
   in place, and put the real password in `MONGODB_PASSWORD`. Special
   characters are URL-encoded for you.
3. In Atlas, allow your IP under **Network Access**.
4. Start the app. The console prints `MongoDB -> cutout_studio connected`.

Data lands in the `removals` collection of the `cutout_studio` database
(`MONGODB_DB` to change it). `GET /api/history?limit=20` returns the latest
removals, and `GET /health` reports the connection state.

## Deploying to Hugging Face Spaces

The repo ships a `Dockerfile` for a free Docker Space (CPU, 16 GB RAM). The
block at the top of this file is the Space's configuration. The models used by
Auto mode are downloaded while the image builds.

Set these in the Space under **Settings → Variables and secrets**:

| Secret | Purpose |
|---|---|
| `MONGODB_URI` | Atlas connection string, with `<db_password>` left in place |
| `MONGODB_PASSWORD` | The database user's password |
| `HISTORY_TOKEN` | Required to read `/api/history`, which lists visitors' filenames |

In Atlas, allow access from `0.0.0.0/0` under **Network Access**, because the
Space's IP address is not fixed. Open the app at its direct
`https://<user>-<space>.hf.space` address; the app's `X-Frame-Options: DENY`
header keeps it from rendering inside the huggingface.co page frame.

## Development

Run the route and export tests without downloading segmentation models:

```powershell
python -m unittest discover -s tests -v
```

Useful local endpoints:

- `GET /health` — runtime status, selected compute backend, model, and app version.
- `GET /api/info` — product version, compute backend, upload limit, and profiles.

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask routes, segmentation, matting, correction, and export processing |
| `db.py` | Optional, fail-soft MongoDB storage for removal history |
| `templates/index.html` | Semantic 2.2 workspace structure |
| `static/styles.css` | Responsive application design and canvas presentation |
| `static/app.js` | Queue, editor, history, comparison, preview, and export behavior |
| `tests/test_app.py` | Route, metadata, security-header, and export tests |
| `requirements.txt` | Python dependencies |
| `start.bat` | Windows launcher |

Set `REMOVE_BG_QUALITY` to `best`, `birefnet`, `balanced`, or `fast` before
starting the server to change the default used by API requests that omit the
quality field.

For troubleshooting only, `REMOVE_BG_PROVIDER` can force `cuda`, `directml`,
`rocm`, `coreml`, or `cpu`. Normal use should leave it unset so selection stays
automatic.
