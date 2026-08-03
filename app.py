"""Focused local background removal with fast segmentation and clean matting."""

import base64
import io
import json
import os
import threading

import cv2
import numpy as np
import onnxruntime as ort
import pooch
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from pymatting import estimate_alpha_cf, estimate_foreground_ml
from rembg import new_session, remove
from scipy import ndimage


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024

QUALITY_PROFILES = {
    "best": {
        "engine": "ben2",
        "model": "BEN2 Base",
        "recover_structures": False,
        "matting_max_side": 1600,
        "label": "BEN2 precision matting",
    },
    "birefnet": {
        "engine": "rembg",
        "model": "birefnet-general",
        "recover_structures": False,
        "matting_max_side": 1400,
        "label": "BiRefNet General",
    },
    "balanced": {
        "engine": "rembg",
        "model": "birefnet-general-lite",
        "recover_structures": False,
        "matting_max_side": 1200,
        "label": "BiRefNet General Lite",
    },
    "fast": {
        "engine": "rembg",
        "model": "isnet-general-use",
        "recover_structures": True,
        "matting_max_side": 1000,
        "label": "ISNet General",
    },
}
DEFAULT_QUALITY = os.environ.get("REMOVE_BG_QUALITY", "best").strip().lower()
if DEFAULT_QUALITY not in QUALITY_PROFILES:
    DEFAULT_QUALITY = "best"
PIPELINE_NAME = "BEN2 confidence matting + high-resolution alpha finishing"
ANALYSIS_MAX_SIDE = 900
GATE_MAX_SIDE = 600
MATTING_MAX_SIDE = 1200
BEN2_MODEL_URL = (
    "https://huggingface.co/PramaLLC/BEN2/resolve/main/BEN2_Base.onnx"
)
BEN2_MODEL_HASH = (
    "sha256:22cea62108ff53b7ccc20f7a008bf30494228d84b1687f29ecbe76936a998101"
)
_sessions = {}
_ben2_session = None
_session_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_session(model_name: str):
    """Load each requested model once and reuse its ONNX session."""
    if model_name not in _sessions:
        with _session_lock:
            if model_name not in _sessions:
                _sessions[model_name] = new_session(model_name)
    return _sessions[model_name]


def get_ben2_session():
    """Load and cache the MIT-licensed BEN2 Base ONNX model."""
    global _ben2_session
    if _ben2_session is None:
        with _session_lock:
            if _ben2_session is None:
                model_home = os.path.expanduser(
                    os.environ.get(
                        "U2NET_HOME",
                        os.path.join(
                            os.environ.get("XDG_DATA_HOME", "~"),
                            ".u2net",
                        ),
                    )
                )
                model_path = pooch.retrieve(
                    url=BEN2_MODEL_URL,
                    known_hash=BEN2_MODEL_HASH,
                    fname="BEN2_Base.onnx",
                    path=model_home,
                    progressbar=True,
                )
                options = ort.SessionOptions()
                options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                available = ort.get_available_providers()
                providers = [
                    provider
                    for provider in (
                        "CUDAExecutionProvider",
                        "DmlExecutionProvider",
                        "CoreMLExecutionProvider",
                        "CPUExecutionProvider",
                    )
                    if provider in available
                ]
                _ben2_session = ort.InferenceSession(
                    model_path,
                    sess_options=options,
                    providers=providers,
                )
    return _ben2_session


def predict_ben2_mask(image: Image.Image) -> Image.Image:
    """Predict a soft foreground matte with BEN2 at its native resolution."""
    original_size = image.size
    prepared = image.convert("RGB").resize(
        (1024, 1024),
        Image.Resampling.BILINEAR,
    )
    tensor = np.asarray(prepared, dtype=np.float32)
    tensor = np.transpose(tensor / 255.0, (2, 0, 1))[None, ...]

    session = get_ben2_session()
    prediction = session.run(
        None,
        {session.get_inputs()[0].name: tensor},
    )[0]
    prediction = np.asarray(prediction, dtype=np.float32).squeeze()
    minimum = float(np.min(prediction))
    maximum = float(np.max(prediction))
    if maximum <= minimum + 1e-8:
        return _empty_mask(original_size)
    prediction = (prediction - minimum) / (maximum - minimum)
    mask = Image.fromarray(
        np.clip(prediction * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    )
    return mask.resize(original_size, Image.Resampling.LANCZOS)


def _empty_mask(size):
    return Image.new("L", size, 0)


def clean_recovered_elements(
    recovered_mask: Image.Image,
    ai_mask: Image.Image,
) -> Image.Image:
    """Reject recovery-only floor fragments without losing upright details."""
    recovered = np.asarray(recovered_mask.convert("L"), dtype=np.uint8)
    ai = np.asarray(ai_mask.convert("L"), dtype=np.uint8)
    height, width = recovered.shape

    recovered_binary = (recovered >= 96).astype(np.uint8)
    vertical_size = max(5, round(height / 90))
    if vertical_size % 2 == 0:
        vertical_size += 1
    vertical_core = cv2.morphologyEx(
        recovered_binary,
        cv2.MORPH_OPEN,
        np.ones((vertical_size, 3), dtype=np.uint8),
    )
    vertical_support = cv2.dilate(
        vertical_core,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    ai_support = cv2.dilate(
        (ai >= 48).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)

    rows = np.arange(height)[:, None]
    lower_area = rows >= round(height * 0.58)
    recovery_is_supported = vertical_support | ai_support
    cleaned = recovered.copy()
    cleaned[lower_area & ~recovery_is_supported] = 0
    cleaned[round(height * 0.84) :] = 0
    return Image.fromarray(cleaned)


def clean_ai_mask(mask: Image.Image) -> Image.Image:
    """Remove detached haze while retaining connected hair and fine details."""
    alpha = np.asarray(mask.convert("L"), dtype=np.uint8).copy()
    possible = (alpha >= 8).astype(np.uint8)
    confident = alpha >= 144
    component_count, labels = cv2.connectedComponents(possible, connectivity=8)
    if component_count <= 1 or not confident.any():
        return _empty_mask(mask.size)

    supported_labels = np.unique(labels[confident])
    supported_labels = supported_labels[supported_labels != 0]
    supported = np.isin(labels, supported_labels)
    alpha[~supported] = 0
    alpha[alpha < 3] = 0
    return Image.fromarray(alpha)


def recovery_is_reliable(mask: Image.Image) -> bool:
    """Identify a useful smooth-background mask before running semantic AI."""
    alpha = np.asarray(mask.convert("L"), dtype=np.uint8)
    recovered_ratio = float(np.mean(alpha >= 64))
    return 0.002 <= recovered_ratio <= 0.85


def combine_masks(
    original: Image.Image,
    ai_mask: Image.Image,
    recovered_mask: Image.Image,
) -> Image.Image:
    """Use smooth-background recovery to prevent AI bridges between objects."""
    ai = np.asarray(ai_mask.convert("L"), dtype=np.uint8)
    recovered = np.asarray(recovered_mask.convert("L"), dtype=np.uint8)
    recovered_area = recovered >= 20
    minimum_area = max(100, round(recovered.size * 0.002))
    if np.count_nonzero(recovered_area) < minimum_area:
        return ai_mask

    recovery_support = cv2.dilate(
        recovered_area.astype(np.uint8),
        np.ones((9, 9), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    ai_confident_support = cv2.dilate(
        (ai >= 224).astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    constrained_ai = ai.copy()
    constrained_ai[~(recovery_support | ai_confident_support)] = 0

    # Recovery is intentionally disabled at the extreme bottom where floor
    # shadows live. Keep the semantic model there so real feet and object bases
    # are not clipped.
    bottom = round(ai.shape[0] * 0.80)
    constrained_ai[bottom:] = ai[bottom:]
    combined = np.maximum(constrained_ai, recovered)

    longest_side = max(original.size)
    scale = min(1.0, GATE_MAX_SIDE / longest_side)
    analysis_size = (
        max(1, round(original.width * scale)),
        max(1, round(original.height * scale)),
    )
    rgb_image = original.convert("RGB")
    rough_image = Image.fromarray(combined)
    recovered_image = Image.fromarray(recovered)
    if analysis_size != original.size:
        rgb_image = rgb_image.resize(analysis_size, Image.Resampling.LANCZOS)
        rough_image = rough_image.resize(analysis_size, Image.Resampling.LANCZOS)
        recovered_image = recovered_image.resize(
            analysis_size,
            Image.Resampling.LANCZOS,
        )

    rgb = cv2.cvtColor(np.asarray(rgb_image), cv2.COLOR_RGB2BGR)
    rough = np.asarray(rough_image, dtype=np.uint8)
    recovered_small = np.asarray(recovered_image, dtype=np.uint8)
    candidate = rough >= 12
    foreground_seed = cv2.erode(
        (recovered_small >= 180).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    ).astype(bool)
    if not foreground_seed.any():
        return Image.fromarray(combined)

    grabcut_mask = np.full(rough.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[candidate] = cv2.GC_PR_FGD
    grabcut_mask[rough <= 2] = cv2.GC_BGD
    grabcut_mask[foreground_seed] = cv2.GC_FGD
    grabcut_mask[[0, -1], :] = cv2.GC_BGD
    grabcut_mask[:, [0, -1]] = cv2.GC_BGD
    try:
        cv2.grabCut(
            rgb,
            grabcut_mask,
            None,
            np.zeros((1, 65), dtype=np.float64),
            np.zeros((1, 65), dtype=np.float64),
            1,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return Image.fromarray(combined)

    keep = np.isin(grabcut_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    keep_image = Image.fromarray((keep * 255).astype(np.uint8))
    if analysis_size != original.size:
        keep_image = keep_image.resize(original.size, Image.Resampling.NEAREST)
    combined[~(np.asarray(keep_image) > 0)] = 0
    return Image.fromarray(combined)


def finish_cutout(
    original: Image.Image,
    combined_mask: Image.Image,
    max_side: int = MATTING_MAX_SIDE,
) -> Image.Image:
    """Refine alpha from source colors and remove colored edge contamination."""
    original_size = original.size
    scale = min(1.0, max_side / max(original_size))
    matte_size = (
        max(1, round(original_size[0] * scale)),
        max(1, round(original_size[1] * scale)),
    )

    rgb_image = original.convert("RGB")
    mask_image = combined_mask.convert("L")
    if matte_size != original_size:
        rgb_image = rgb_image.resize(matte_size, Image.Resampling.LANCZOS)
        mask_image = mask_image.resize(matte_size, Image.Resampling.LANCZOS)

    rgb = np.asarray(rgb_image, dtype=np.float64) / 255.0
    rough_alpha = np.asarray(mask_image, dtype=np.uint8)
    # Keep extremely soft model responses in the unknown band so wispy hair,
    # fur, glass, veils, and motion-blurred edges can be solved from source
    # colors instead of being clipped before matting.
    candidate = rough_alpha >= 4
    if not candidate.any():
        result = original.copy()
        result.putalpha(_empty_mask(original_size))
        return result

    kernel = np.ones((3, 3), dtype=np.uint8)
    candidate_u8 = candidate.astype(np.uint8)
    # Only force genuinely opaque model pixels to alpha 1. Lower thresholds
    # make antialiased edges look thick and destroy semi-transparent detail.
    high_confidence = (rough_alpha >= 235).astype(np.uint8)
    sure_foreground = cv2.erode(
        high_confidence,
        kernel,
        iterations=2,
    ).astype(bool)
    distance = cv2.distanceTransform(high_confidence, cv2.DIST_L2, 5)
    local_maximum = distance >= (
        cv2.dilate(distance, kernel, iterations=1) - 1e-4
    )
    thin_centerline = local_maximum & (distance >= 0.9)
    sure_foreground |= thin_centerline
    possible_foreground = cv2.dilate(
        candidate_u8,
        kernel,
        iterations=1,
    ).astype(bool)

    trimap = np.full(rough_alpha.shape, 0.5, dtype=np.float64)
    trimap[~possible_foreground] = 0.0
    trimap[sure_foreground] = 1.0

    try:
        alpha = estimate_alpha_cf(
            rgb,
            trimap,
            cg_kwargs={"maxiter": 220},
        )
        alpha[~possible_foreground] = 0.0
        alpha[sure_foreground] = 1.0
        alpha[alpha < 0.004] = 0.0
        alpha[alpha > 0.996] = 1.0
    except (ArithmeticError, ValueError):
        alpha = rough_alpha.astype(np.float64) / 255.0

    try:
        foreground = estimate_foreground_ml(
            rgb,
            alpha,
            n_small_iterations=6,
            n_big_iterations=1,
        )
    except (ArithmeticError, ValueError):
        foreground = rgb

    alpha_image = Image.fromarray(
        np.clip(alpha * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    )
    foreground_image = Image.fromarray(
        np.clip(foreground * 255.0, 0, 255).astype(np.uint8),
        mode="RGB",
    )
    if matte_size != original_size:
        alpha_image = alpha_image.resize(original_size, Image.Resampling.LANCZOS)
        foreground_image = foreground_image.resize(
            original_size,
            Image.Resampling.LANCZOS,
        )

    result = np.asarray(original.convert("RGBA")).copy()
    alpha_full = np.asarray(alpha_image, dtype=np.uint8)
    foreground_full = np.asarray(foreground_image, dtype=np.uint8)
    edge_pixels = (alpha_full > 0) & (alpha_full < 252)
    result[edge_pixels, :3] = foreground_full[edge_pixels]

    source_alpha = np.asarray(original.getchannel("A"), dtype=np.uint16)
    final_alpha = (
        alpha_full.astype(np.uint16) * source_alpha // 255
    ).astype(np.uint8)
    result[:, :, 3] = final_alpha
    return Image.fromarray(result)


def matte_refined_target(rgb: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Create a source-aware soft alpha edge around a GrabCut target."""
    binary = target.astype(np.uint8)
    if not binary.any():
        return np.zeros(target.shape, dtype=np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    sure_foreground = cv2.erode(binary, kernel, iterations=1).astype(bool)
    if not sure_foreground.any():
        sure_foreground = target
    outer_edge = cv2.dilate(binary, kernel, iterations=1).astype(bool)

    trimap = np.zeros(target.shape, dtype=np.float64)
    trimap[outer_edge] = 0.5
    trimap[sure_foreground] = 1.0
    try:
        alpha = estimate_alpha_cf(
            rgb.astype(np.float64) / 255.0,
            trimap,
            cg_kwargs={"maxiter": 300},
        )
    except (ArithmeticError, ValueError):
        alpha = binary.astype(np.float64)
    return np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


def smart_refine(
    original: Image.Image,
    current: Image.Image,
    points: list,
    brush_size: int,
    mode: str,
) -> Image.Image:
    """Snap a Restore or Erase stroke to local image boundaries with GrabCut."""
    original = original.convert("RGBA")
    current = current.convert("RGBA")
    if current.size != original.size:
        current = current.resize(original.size, Image.Resampling.LANCZOS)

    width, height = original.size
    parsed_points = []
    for point in points[:5000]:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
        except (TypeError, ValueError):
            continue
        parsed_points.append(
            (min(width - 1, max(0, x)), min(height - 1, max(0, y)))
        )
    if not parsed_points:
        raise ValueError("The refinement stroke is empty.")

    brush_size = min(600, max(4, int(brush_size)))
    selection = np.zeros((height, width), dtype=np.uint8)
    intent_seed = np.zeros_like(selection)
    outer_thickness = brush_size
    seed_thickness = max(2, min(4, round(brush_size * 0.025)))
    path = np.asarray(parsed_points, dtype=np.int32).reshape((-1, 1, 2))

    if len(parsed_points) == 1:
        cv2.circle(selection, parsed_points[0], outer_thickness // 2, 255, -1)
        cv2.circle(intent_seed, parsed_points[0], seed_thickness // 2, 255, -1)
    else:
        cv2.polylines(
            selection,
            [path],
            False,
            255,
            thickness=outer_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.polylines(
            intent_seed,
            [path],
            False,
            255,
            thickness=seed_thickness,
            lineType=cv2.LINE_AA,
        )

    selected_y, selected_x = np.nonzero(selection)
    padding = max(20, brush_size // 2)
    left = max(0, int(selected_x.min()) - padding)
    right = min(width, int(selected_x.max()) + padding + 1)
    top = max(0, int(selected_y.min()) - padding)
    bottom = min(height, int(selected_y.max()) + padding + 1)

    rgb = np.asarray(original.convert("RGB"))
    current_rgba = np.asarray(current).copy()
    roi_rgb = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2BGR)
    roi_selection = selection[top:bottom, left:right] > 0
    roi_seed = intent_seed[top:bottom, left:right] > 0

    work_area = cv2.dilate(
        roi_selection.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=max(1, brush_size // 80),
    ).astype(bool)
    if mode == "restore":
        grabcut_mask = np.full(roi_selection.shape, cv2.GC_BGD, dtype=np.uint8)
        grabcut_mask[work_area] = cv2.GC_PR_BGD
        grabcut_mask[roi_seed] = cv2.GC_FGD
    elif mode == "erase":
        grabcut_mask = np.full(
            roi_selection.shape,
            cv2.GC_PR_FGD,
            dtype=np.uint8,
        )
        grabcut_mask[~work_area] = cv2.GC_BGD
        grabcut_mask[roi_selection] = cv2.GC_PR_BGD
        grabcut_mask[roi_seed] = cv2.GC_BGD
    else:
        raise ValueError("Unknown refinement mode.")

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        roi_rgb,
        grabcut_mask,
        None,
        background_model,
        foreground_model,
        2,
        cv2.GC_INIT_WITH_MASK,
    )

    target_labels = (
        (cv2.GC_FGD, cv2.GC_PR_FGD)
        if mode == "restore"
        else (cv2.GC_BGD, cv2.GC_PR_BGD)
    )
    target = np.isin(grabcut_mask, target_labels)
    target &= roi_selection
    target_alpha = matte_refined_target(
        rgb[top:bottom, left:right],
        target,
    )

    current_alpha = current_rgba[top:bottom, left:right, 3].astype(np.float32)
    target_float = target_alpha.astype(np.float32) / 255.0
    if mode == "restore":
        new_alpha = np.maximum(current_alpha, target_alpha)
        restore_pixels = target_alpha > current_alpha
        original_roi = np.asarray(original)[top:bottom, left:right]
        for channel in range(3):
            channel_data = current_rgba[top:bottom, left:right, channel]
            channel_data[restore_pixels] = original_roi[:, :, channel][restore_pixels]
    elif mode == "erase":
        new_alpha = current_alpha * (1.0 - target_float)
    current_rgba[top:bottom, left:right, 3] = np.clip(
        new_alpha,
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(current_rgba)


def recover_all_elements(image: Image.Image) -> Image.Image:
    """Recover graphic elements that a salient-subject model may ignore.

    Semantic segmentation is deliberately selective. For illustrations on a smooth
    background, that can discard secondary objects and thin connectors. This
    pass learns a smooth polynomial background from edge-connected pixels and
    keeps only structures that differ from that model.

    Complex photographic backgrounds produce a high model residual, which
    disables this pass so normal photo removal still follows ISNet alone.
    """
    original_size = image.size
    longest_side = max(original_size)
    scale = min(1.0, ANALYSIS_MAX_SIDE / longest_side)
    analysis_size = (
        max(1, round(original_size[0] * scale)),
        max(1, round(original_size[1] * scale)),
    )
    analysis = image.convert("RGB")
    if analysis.size != analysis_size:
        analysis = analysis.resize(analysis_size, Image.Resampling.LANCZOS)

    rgb = np.asarray(analysis, dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    if height < 24 or width < 24:
        return _empty_mask(original_size)

    smooth = ndimage.gaussian_filter(rgb, sigma=(1.0, 1.0, 0.0))
    gradient_x = np.stack(
        [ndimage.sobel(smooth[:, :, channel], axis=1) for channel in range(3)],
        axis=2,
    )
    gradient_y = np.stack(
        [ndimage.sobel(smooth[:, :, channel], axis=0) for channel in range(3)],
        axis=2,
    )
    gradient = np.sqrt(np.sum(gradient_x ** 2 + gradient_y ** 2, axis=2))

    border_width = max(2, min(height, width) // 100)
    border_gradient = np.concatenate(
        (
            gradient[:border_width, :].ravel(),
            gradient[-border_width:, :].ravel(),
            gradient[:, :border_width].ravel(),
            gradient[:, -border_width:].ravel(),
        )
    )
    edge_threshold = max(
        0.05,
        float(np.percentile(border_gradient, 95)) * 1.5,
    )

    barrier = gradient > edge_threshold
    barrier = ndimage.binary_closing(barrier, structure=np.ones((3, 3)))
    barrier = ndimage.binary_dilation(barrier, iterations=1)

    seed = np.zeros((height, width), dtype=bool)
    corner_size = max(2, min(height, width) // 128)
    seed[:corner_size, :corner_size] = True
    seed[:corner_size, -corner_size:] = True
    seed[-corner_size:, :corner_size] = True
    seed[-corner_size:, -corner_size:] = True
    background = ndimage.binary_propagation(seed, mask=~barrier)

    background_ratio = float(background.mean())
    if background_ratio < 0.18 or background.sum() < 100:
        return _empty_mask(original_size)

    # Fit a smooth 2D cubic color surface to pixels reached from the corners.
    # Sampling bounds the solve cost without changing the full-size prediction.
    background_y, background_x = np.nonzero(background)
    sample_step = max(1, len(background_x) // 40_000)
    sample_x = background_x[::sample_step].astype(np.float32)
    sample_y = background_y[::sample_step].astype(np.float32)
    normalized_x = (sample_x / max(1, width - 1)) * 2.0 - 1.0
    normalized_y = (sample_y / max(1, height - 1)) * 2.0 - 1.0
    sample_features = np.column_stack(
        (
            np.ones_like(normalized_x),
            normalized_x,
            normalized_y,
            normalized_x ** 2,
            normalized_x * normalized_y,
            normalized_y ** 2,
            normalized_x ** 3,
            (normalized_x ** 2) * normalized_y,
            normalized_x * (normalized_y ** 2),
            normalized_y ** 3,
        )
    )
    sample_colors = rgb[background_y[::sample_step], background_x[::sample_step]]
    coefficients = np.linalg.lstsq(
        sample_features,
        sample_colors,
        rcond=None,
    )[0]

    grid_y, grid_x = np.mgrid[0:height, 0:width]
    normalized_x = (grid_x.astype(np.float32) / max(1, width - 1)) * 2.0 - 1.0
    normalized_y = (grid_y.astype(np.float32) / max(1, height - 1)) * 2.0 - 1.0
    modeled_background = (
        coefficients[0]
        + coefficients[1] * normalized_x[..., None]
        + coefficients[2] * normalized_y[..., None]
        + coefficients[3] * (normalized_x ** 2)[..., None]
        + coefficients[4] * (normalized_x * normalized_y)[..., None]
        + coefficients[5] * (normalized_y ** 2)[..., None]
        + coefficients[6] * (normalized_x ** 3)[..., None]
        + coefficients[7] * ((normalized_x ** 2) * normalized_y)[..., None]
        + coefficients[8] * (normalized_x * (normalized_y ** 2))[..., None]
        + coefficients[9] * (normalized_y ** 3)[..., None]
    )
    polynomial_residual = np.sqrt(
        np.sum((rgb - modeled_background) ** 2, axis=2)
    )

    # A normalized Gaussian reconstruction follows local gradients and floor
    # lighting more closely than the global polynomial. Taking the better fit
    # prevents smooth shadows and lighting bands from becoming foreground.
    reconstruction_sigma = min(48.0, max(10.0, min(height, width) / 26.0))
    background_weight = ndimage.gaussian_filter(
        background.astype(np.float32),
        sigma=reconstruction_sigma,
    )
    local_background = np.empty_like(rgb)
    safe_weight = np.maximum(background_weight, 1e-5)
    for channel in range(3):
        weighted_channel = ndimage.gaussian_filter(
            rgb[:, :, channel] * background,
            sigma=reconstruction_sigma,
        )
        local_background[:, :, channel] = weighted_channel / safe_weight

    local_residual = np.sqrt(
        np.sum((rgb - local_background) ** 2, axis=2)
    )
    residual = np.minimum(polynomial_residual, local_residual)
    residual_threshold = max(
        0.04,
        float(np.percentile(residual[background], 99)) * 1.4,
    )

    # A high residual means this is a complex photographic background rather
    # than a smooth graphic. In that case, BiRefNet remains the safer mask.
    if residual_threshold > 0.22:
        return _empty_mask(original_size)

    local_recovery = (~background) & (
        (residual > residual_threshold)
        | (
            (gradient > edge_threshold)
            & (residual > residual_threshold * 0.5)
        )
    )

    # Local reconstruction can mistake a faint, uniformly colored silhouette
    # for nearby background. Restore global support only for coherent upper
    # elements or tall components. Wide components low in the image are usually
    # floor shadows and deliberately do not receive this fallback.
    polynomial_threshold = max(
        0.04,
        float(np.percentile(polynomial_residual[background], 99)) * 1.4,
    )
    global_recovery = (~background) & (
        (polynomial_residual > polynomial_threshold)
        | (
            (gradient > edge_threshold)
            & (polynomial_residual > polynomial_threshold * 0.5)
        )
    )
    global_recovery = ndimage.binary_opening(
        global_recovery,
        structure=np.ones((3, 3)),
    )
    global_labels, global_count = ndimage.label(global_recovery)
    coherent_global = np.zeros_like(global_recovery)
    if global_count:
        for label_id, bounds in enumerate(
            ndimage.find_objects(global_labels),
            start=1,
        ):
            if bounds is None:
                continue
            y_slice, x_slice = bounds
            component_height = y_slice.stop - y_slice.start
            component_width = x_slice.stop - x_slice.start
            center_y = (y_slice.start + y_slice.stop) / 2.0
            is_upper_element = center_y < height * 0.68
            is_tall_element = component_height > component_width * 1.25
            if is_upper_element or is_tall_element:
                region = global_labels[y_slice, x_slice] == label_id
                coherent_global[y_slice, x_slice] |= region

    recovered = local_recovery | coherent_global
    # Opening removes isolated mask splatter; closing reconnects antialiased
    # strokes and narrow gaps without growing the recovered area.
    recovered = ndimage.binary_opening(
        recovered,
        structure=np.ones((3, 3)),
    )
    recovered = ndimage.binary_closing(
        recovered,
        structure=np.ones((3, 3)),
    )

    labels, component_count = ndimage.label(recovered)
    if component_count:
        component_sizes = np.bincount(labels.ravel())
        minimum_size = max(12, round(width * height * 0.00003))
        keep_component = component_sizes >= minimum_size
        keep_component[0] = False
        recovered = keep_component[labels]

    recovered_mask = Image.fromarray((recovered * 255).astype(np.uint8))
    recovered_mask = recovered_mask.filter(ImageFilter.GaussianBlur(0.6))
    if recovered_mask.size != original_size:
        recovered_mask = recovered_mask.resize(
            original_size,
            Image.Resampling.LANCZOS,
        )
    return recovered_mask


def create_cutout(
    image_bytes: bytes,
    quality: str = DEFAULT_QUALITY,
) -> bytes:
    """Return a transparent PNG using the selected quality profile."""
    if quality not in QUALITY_PROFILES:
        raise ValueError("Choose a valid removal quality.")
    profile = QUALITY_PROFILES[quality]

    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        source = ImageOps.exif_transpose(source)
        original = source.convert("RGBA")

    # BEN2 and BiRefNet provide stronger masks than the previous ISNet default.
    # The legacy structure-recovery pass is retained only for Fast mode, where
    # it can recover graphic elements that the smaller salient-subject model
    # misses. It is deliberately not allowed to overrule the precision models
    # because doing so can restore shadows and smooth background patches.
    recovered_elements = (
        recover_all_elements(original)
        if profile["recover_structures"]
        else None
    )
    with _inference_lock:
        if profile["engine"] == "ben2":
            ai_mask = predict_ben2_mask(original)
        else:
            ai_mask = remove(
                original.convert("RGB"),
                session=get_session(profile["model"]),
                only_mask=True,
                alpha_matting=False,
                post_process_mask=False,
            ).convert("L")

    if ai_mask.size != original.size:
        ai_mask = ai_mask.resize(original.size, Image.Resampling.LANCZOS)

    ai_mask = clean_ai_mask(ai_mask)
    if (
        recovered_elements is not None
        and recovery_is_reliable(recovered_elements)
    ):
        recovered_elements = clean_recovered_elements(
            recovered_elements,
            ai_mask,
        )
        mask = combine_masks(original, ai_mask, recovered_elements)
    else:
        mask = ai_mask
    cutout = finish_cutout(
        original,
        mask,
        max_side=profile["matting_max_side"],
    )
    output = io.BytesIO()
    cutout.save(output, format="PNG", compress_level=4)
    return output.getvalue()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/remove")
def remove_background():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose an image first."}), 400

    quality = request.form.get("quality", DEFAULT_QUALITY).strip().lower()
    if quality not in QUALITY_PROFILES:
        return jsonify({"error": "Choose a valid removal quality."}), 400

    try:
        result = create_cutout(uploaded.read(), quality)
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "That file is not a supported image."}), 400
    except Exception as exc:
        app.logger.exception("Background removal failed")
        return jsonify({"error": f"Background removal failed: {exc}"}), 500

    stem = os.path.splitext(os.path.basename(uploaded.filename))[0] or "image"
    return send_file(
        io.BytesIO(result),
        mimetype="image/png",
        as_attachment=False,
        download_name=f"{stem}-no-bg.png",
    )


@app.post("/refine")
def refine_mask():
    original_file = request.files.get("image")
    current_file = request.files.get("current")
    if original_file is None or current_file is None:
        return jsonify({"error": "The original image and current result are required."}), 400

    try:
        points = json.loads(request.form.get("points", "[]"))
        brush_size = int(request.form.get("brush_size", "48"))
        mode = request.form.get("mode", "erase")
        with Image.open(original_file.stream) as original_source:
            original_source.load()
            original = ImageOps.exif_transpose(original_source).convert("RGBA")
        with Image.open(current_file.stream) as current_source:
            current_source.load()
            current = current_source.convert("RGBA")
        refined = smart_refine(original, current, points, brush_size, mode)
    except (json.JSONDecodeError, UnidentifiedImageError, OSError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc) or "Invalid refinement request."}), 400
    except cv2.error as exc:
        app.logger.exception("Smart refinement failed")
        return jsonify({"error": f"Could not analyze that selection: {exc}"}), 500

    output = io.BytesIO()
    refined.save(output, format="PNG", compress_level=4)
    output.seek(0)
    return send_file(output, mimetype="image/png", download_name="refined.png")


@app.post("/export")
def export_cutout():
    current_file = request.files.get("current")
    output_format = request.form.get("format", "png").lower()
    if current_file is None:
        return jsonify({"error": "The current cutout is required."}), 400
    if output_format not in {"png", "webp", "svg"}:
        return jsonify({"error": "Choose PNG, WebP, or SVG."}), 400

    try:
        with Image.open(current_file.stream) as current_source:
            current_source.load()
            current = current_source.convert("RGBA")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "The current cutout is invalid."}), 400

    stem = os.path.splitext(
        os.path.basename(request.form.get("filename", "image"))
    )[0] or "image"
    output = io.BytesIO()
    if output_format == "webp":
        current.save(output, format="WEBP", lossless=True, method=6)
        mimetype = "image/webp"
    else:
        png = io.BytesIO()
        current.save(png, format="PNG", compress_level=4)
        if output_format == "png":
            output = png
            mimetype = "image/png"
        else:
            encoded = base64.b64encode(png.getvalue()).decode("ascii")
            svg = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{current.width}" height="{current.height}" '
                f'viewBox="0 0 {current.width} {current.height}">\n'
                f'  <image width="{current.width}" height="{current.height}" '
                f'href="data:image/png;base64,{encoded}"/>\n'
                "</svg>\n"
            )
            output = io.BytesIO(svg.encode("utf-8"))
            mimetype = "image/svg+xml"

    output.seek(0)
    return send_file(
        output,
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"{stem}-no-bg.{output_format}",
    )


@app.get("/health")
def health():
    profile = QUALITY_PROFILES[DEFAULT_QUALITY]
    return jsonify(
        {
            "status": "ok",
            "default_quality": DEFAULT_QUALITY,
            "model": profile["model"],
            "pipeline": PIPELINE_NAME,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n  Background Remover")
    print(f"  Open  ->  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
