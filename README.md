# Background Remover

A focused local tool for removing image backgrounds. Best quality uses the
MIT-licensed BEN2 Base confidence-guided matting model, followed by
high-resolution source-color alpha refinement and edge-color cleanup. BiRefNet,
Balanced, and Fast modes remain available because no single model wins on every
kind of subject.

The app intentionally has only two jobs:

1. Create a high-quality automatic cutout while preserving soft hair, fur,
   holes, motion blur, and semi-transparent edges.
2. Let you correct the result with edge-aware Restore and Erase selections,
   including 15-step Undo and Redo history.

Images stay on your computer.

Very large uploads are automatically resized to 12 megapixels before removal.
This prevents memory failures while retaining a high-resolution PNG result.

## Run

Double-click `start.bat`, or run:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

Models download automatically on first use and are cached in `~/.u2net/`.
Best uses the roughly 213 MB BEN2 Base model, BiRefNet uses the roughly 1 GB
BiRefNet General model, Balanced uses the roughly 224 MB BiRefNet General Lite
model, and Fast uses the roughly 179 MB ISNet model. Full BiRefNet can take a
few minutes per image on a CPU.

## Use

1. Choose or drop one image.
2. Choose **Best**, **BiRefNet**, **Balanced**, or **Fast**, then select
   **Remove background**.
3. Select **Restore** and brush over a missing foreground area. The app analyzes
   the selection and restores pixels up to the detected object boundary.
4. Select **Erase** and brush over a background remnant. The app analyzes the
   selection and removes it without blindly applying the full brush shape.
5. Use **Undo** and **Redo** or `Ctrl+Z`, `Ctrl+Shift+Z`, and `Ctrl+Y` to move
   through up to 15 successful brush edits.
6. Select PNG, lossless WebP, or SVG and download the result.

SVG export embeds the edited full-resolution transparent PNG in an SVG
document. This preserves photographic detail instead of reducing it to traced
vector shapes.

The smart brush changes transparency only. Restored RGB pixels come from the
original upload; the app does not sharpen, recolor, or upscale them.

## Improve Fast - ISNet

After correcting a Fast - ISNet result, select **Save final correction locally**.
The app saves the normalized original image, your approved alpha mask, and
metadata in `feedback_data/`. This folder stays on your computer and is ignored
by Git.

Saving examples builds a reliable training set; it does not automatically
retrain ISNet after each image. Once enough varied, corrected examples have
been collected, use them to tune a Fast-mode correction model without teaching
the app from its own mistakes.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Flask backend, segmentation, refinement, and export processing |
| `templates/index.html` | Upload, correction, and download interface |
| `requirements.txt` | Python dependencies |
| `start.bat` | Windows launcher |
| `create_shortcut.bat` | Desktop shortcut creator |

Set `REMOVE_BG_QUALITY` to `best`, `birefnet`, `balanced`, or `fast` before
starting the server to change the default used by API requests that omit the
quality field.
