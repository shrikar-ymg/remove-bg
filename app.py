"""Focused local background removal with fast segmentation and clean matting."""

import base64
import gc
import io
import json
import os
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

try:
    import winreg
except ImportError:  # pragma: no cover - Windows-only standard library module
    winreg = None

import cv2
import numpy as np
import onnxruntime as ort
import pooch
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageColor, ImageFilter, ImageOps, UnidentifiedImageError

# Settings such as MONGODB_URI live in an untracked .env beside this file. It
# never overrides variables that are already set in the real environment.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import db

# PyMatting uses Numba's on-disk cache while it imports. Some Windows Python
# installations put the package in a location where Numba cannot create a cache
# locator, so give it an explicit writable directory before importing PyMatting.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "cutout-studio-numba-cache"),
)
from pymatting import estimate_alpha_cf, estimate_foreground_ml
from rembg import remove
from rembg.sessions import sessions_class
from scipy import ndimage


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024

APP_NAME = "Cutout Studio"
APP_VERSION = "2.2.0"

QUALITY_PROFILES = {
    "best": {
        "engine": "ben2",
        "model": "BEN2 Base",
        "recover_structures": False,
        "matting_max_side": 1800,
        "preserve_shadows": False,
        "label": "BEN2 precision matting",
    },
    "birefnet": {
        "engine": "rembg",
        "model": "birefnet-general",
        "recover_structures": False,
        "matting_max_side": 1400,
        "preserve_shadows": False,
        "label": "BiRefNet General",
    },
    "balanced": {
        "engine": "rembg",
        "model": "birefnet-general-lite",
        "recover_structures": False,
        "matting_max_side": 1200,
        "preserve_shadows": False,
        "label": "BiRefNet General Lite",
    },
    "fast": {
        "engine": "rembg",
        "model": "isnet-general-use",
        "recover_structures": True,
        "matting_max_side": 1000,
        "preserve_shadows": False,
        "label": "ISNet General",
    },
}
AUTO_QUALITY = "auto"
# Photographs are compared between the precision model and the more selective
# lightweight one; those two disagree in the way that matters, and adding a
# third candidate only costs another full inference.
# The lightweight BiRefNet needs a single 800 MB buffer, far more than BEN2's
# working set, so it is tried while memory is least fragmented. If it cannot
# run, the comparison still has BEN2's answer to fall back on.
AUTO_PHOTO_CANDIDATES = ("balanced", "best")
# Scores this close together mean the measurement cannot separate the
# candidates, so the general-purpose model wins instead of sensor noise.
AUTO_TIE_MARGIN = 0.05
AUTO_PREFERRED_QUALITY = "best"
DEFAULT_QUALITY = os.environ.get("REMOVE_BG_QUALITY", AUTO_QUALITY).strip().lower()
if DEFAULT_QUALITY not in QUALITY_PROFILES and DEFAULT_QUALITY != AUTO_QUALITY:
    DEFAULT_QUALITY = AUTO_QUALITY
PIPELINE_NAME = "BEN2 confidence-preserving matting + edge finishing"
ANALYSIS_MAX_SIDE = 900
GATE_MAX_SIDE = 600
MATTING_MAX_SIDE = 1200
MAX_PROCESSING_PIXELS = 12_000_000
# A closed-form matting solve needs about 400 bytes per pixel, so the number
# of pixels handed to it has to be bounded no matter how large the upload or
# how long a correction stroke is.
CF_SOLVE_MAX_PIXELS = 2_000_000
# GrabCut builds a flow graph with several edges per pixel, so the region it
# is given needs its own bound: a stroke dragged across a large photograph
# would otherwise hand it the entire image.
GRABCUT_MAX_PIXELS = 1_200_000
FEEDBACK_DATA_DIR = os.environ.get(
    "REMOVE_BG_FEEDBACK_DIR",
    os.path.join(app.root_path, "feedback_data"),
)
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
_active_providers = {}
_directml_devices_cache = None
_windows_gpu_devices_cache = None

PROVIDER_LABELS = {
    "CUDAExecutionProvider": "NVIDIA CUDA GPU",
    "DmlExecutionProvider": "DirectML GPU",
    "ROCMExecutionProvider": "AMD ROCm GPU",
    "CoreMLExecutionProvider": "Apple Core ML",
    "CPUExecutionProvider": "CPU",
}
PROVIDER_PRIORITY = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)


@app.after_request
def add_response_headers(response):
    """Keep the local interface fresh and apply a conservative browser policy."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )
    if response.mimetype in {"text/html", "text/css", "application/javascript"}:
        response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": "That image is larger than the 60 MB upload limit."}), 413


def directml_devices() -> list[dict]:
    """Read DirectML adapter metadata without loading or benchmarking a model."""
    global _directml_devices_cache
    if _directml_devices_cache is not None:
        return _directml_devices_cache
    devices = []
    try:
        ep_devices = ort.get_ep_devices()
    except (AttributeError, RuntimeError):
        ep_devices = []
    for ep_device in ep_devices:
        if ep_device.ep_name != "DmlExecutionProvider":
            continue
        metadata = dict(ep_device.device.metadata)
        memory_text = str(metadata.get("DxgiVideoMemory", "0"))
        memory_match = re.search(r"\d+", memory_text)
        devices.append(
            {
                "device_id": int(ep_device.ep_options.get("device_id", 0)),
                "name": metadata.get("Description", "DirectML GPU"),
                "vendor": ep_device.device.vendor or "GPU",
                "memory_mb": int(memory_match.group()) if memory_match else 0,
                "discrete": metadata.get("Discrete") == "1",
            }
        )
    _directml_devices_cache = devices
    return devices


def windows_display_devices() -> list[dict]:
    """Find Windows display adapters quickly when the DML wheel is not installed."""
    global _windows_gpu_devices_cache
    if _windows_gpu_devices_cache is not None:
        return _windows_gpu_devices_cache
    if winreg is None:
        _windows_gpu_devices_cache = []
        return _windows_gpu_devices_cache

    devices = []
    class_path = (
        r"SYSTEM\CurrentControlSet\Control\Class"
        r"\{4d36e968-e325-11ce-bfc1-08002be10318}"
    )
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, class_path)
    except OSError:
        _windows_gpu_devices_cache = []
        return _windows_gpu_devices_cache
    with root:
        index = 0
        while True:
            try:
                key_name = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            try:
                key = winreg.OpenKey(root, key_name)
            except OSError:
                continue
            with key:
                try:
                    name = winreg.QueryValueEx(key, "DriverDesc")[0]
                except OSError:
                    continue
                try:
                    vendor = winreg.QueryValueEx(key, "ProviderName")[0]
                except OSError:
                    vendor = "GPU"
                try:
                    memory = winreg.QueryValueEx(
                        key,
                        "HardwareInformation.MemorySize",
                    )[0]
                    if isinstance(memory, bytes):
                        memory = int.from_bytes(memory[:8], "little")
                    memory_mb = int(memory) // (1024 * 1024)
                except (OSError, TypeError, ValueError):
                    memory_mb = 0
                devices.append(
                    {
                        "device_id": None,
                        "name": str(name),
                        "vendor": str(vendor),
                        "memory_mb": memory_mb,
                        "discrete": any(
                            token in f"{vendor} {name}".lower()
                            for token in ("nvidia", "radeon", "intel arc")
                        ),
                    }
                )
    _windows_gpu_devices_cache = devices
    return devices


def preferred_directml_device() -> dict | None:
    devices = directml_devices()
    if not devices:
        return None

    def rank(device):
        name = device["name"].lower()
        family_bonus = 0
        if any(token in name for token in (" rtx ", " gtx ", "radeon rx", "intel arc")):
            family_bonus = 2
        elif device["discrete"]:
            family_bonus = 1
        return family_bonus, device["memory_mb"]

    return max(devices, key=rank)


def detected_gpu() -> dict | None:
    """Return the best GPU description available without any inference work."""
    dml_device = preferred_directml_device()
    if dml_device:
        return dml_device
    devices = windows_display_devices()
    if not devices:
        return None
    return max(
        devices,
        key=lambda device: (
            device["discrete"],
            device["memory_mb"],
        ),
    )


def directml_is_suitable(device: dict | None) -> bool:
    """Avoid known low-power adapters that run BEN2 slower than the CPU."""
    if device is None:
        # Older DirectML builds do not expose adapter metadata. Keep backward
        # compatibility and let session creation decide whether DML works.
        return True
    name = device["name"].lower()
    low_power_family = (
        "geforce mx" in name
        or "geforce gt " in name
        or "intel(r) uhd" in name
        or "intel uhd" in name
        or "intel(r) hd graphics" in name
    )
    too_little_dedicated_memory = device["discrete"] and 0 < device["memory_mb"] < 1024
    return not (low_power_family or too_little_dedicated_memory)


def available_provider_priority() -> list[str]:
    """Return usable ONNX providers in automatic preference order."""
    available = set(ort.get_available_providers())
    requested = os.environ.get("REMOVE_BG_PROVIDER", "auto").strip().lower()
    aliases = {
        "cuda": "CUDAExecutionProvider",
        "directml": "DmlExecutionProvider",
        "dml": "DmlExecutionProvider",
        "rocm": "ROCMExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }
    forced = aliases.get(requested)
    if forced in available:
        return [forced, *(p for p in PROVIDER_PRIORITY if p in available and p != forced)]
    providers = [provider for provider in PROVIDER_PRIORITY if provider in available]
    if (
        "DmlExecutionProvider" in providers
        and not directml_is_suitable(preferred_directml_device())
    ):
        providers.remove("DmlExecutionProvider")
    return providers


def provider_chain(primary: str):
    """Build one accelerated provider plus a reliable CPU fallback."""
    if primary == "CUDAExecutionProvider":
        providers = [
            (
                primary,
                {
                    # Heuristic convolution selection avoids a long first-run
                    # benchmark on consumer NVIDIA cards.
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "arena_extend_strategy": "kSameAsRequested",
                    "do_copy_in_default_stream": "1",
                },
            )
        ]
    elif primary == "DmlExecutionProvider":
        device = preferred_directml_device()
        device_id = device["device_id"] if device else 0
        providers = [(primary, {"device_id": str(device_id)})]
    else:
        providers = [primary]
    if primary != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    return providers


def session_options(primary: str) -> ort.SessionOptions:
    """Create options compatible with the selected execution provider."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if primary == "DmlExecutionProvider":
        # Required by ONNX Runtime's DirectML execution provider.
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    threads = os.environ.get("OMP_NUM_THREADS")
    if threads:
        options.inter_op_num_threads = int(threads)
        options.intra_op_num_threads = int(threads)
    return options


def _record_active_provider(model_name: str, session) -> str:
    active = next(
        (
            provider
            for provider in session.get_providers()
            if provider in PROVIDER_LABELS
        ),
        "CPUExecutionProvider",
    )
    _active_providers[model_name] = active
    return active


def prepare_provider(primary: str) -> None:
    if primary == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
        # Finds CUDA/cuDNN supplied by PyTorch or ONNX Runtime's optional
        # NVIDIA packages without requiring PATH changes.
        ort.preload_dlls()


def create_ort_session(model_path: str, model_name: str):
    """Create a model session on the fastest working local provider."""
    errors = []
    for primary in available_provider_priority() or ["CPUExecutionProvider"]:
        try:
            prepare_provider(primary)
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options(primary),
                providers=provider_chain(primary),
            )
            active = _record_active_provider(model_name, session)
            if primary == "CPUExecutionProvider" or active == primary:
                return session
            errors.append(f"{primary} initialized as {active}")
        except Exception as exc:
            errors.append(f"{primary}: {exc}")
            app.logger.warning(
                "Could not initialize %s for %s; trying the next provider: %s",
                primary,
                model_name,
                exc,
            )
    raise RuntimeError(
        f"No ONNX Runtime provider could load {model_name}: " + "; ".join(errors)
    )


def get_session(model_name: str):
    """Load and cache a rembg model on the fastest working local provider."""
    if model_name not in _sessions:
        with _session_lock:
            if model_name not in _sessions:
                session_class = next(
                    (
                        candidate
                        for candidate in sessions_class
                        if candidate.name() == model_name
                    ),
                    None,
                )
                if session_class is None:
                    raise ValueError(f"No session class found for model '{model_name}'")

                errors = []
                for primary in available_provider_priority() or ["CPUExecutionProvider"]:
                    try:
                        prepare_provider(primary)
                        candidate = session_class(
                            model_name,
                            session_options(primary),
                            providers=provider_chain(primary),
                        )
                        active = _record_active_provider(
                            model_name,
                            candidate.inner_session,
                        )
                        if primary == "CPUExecutionProvider" or active == primary:
                            _sessions[model_name] = candidate
                            break
                        errors.append(f"{primary} initialized as {active}")
                    except Exception as exc:
                        errors.append(f"{primary}: {exc}")
                        app.logger.warning(
                            "Could not initialize %s for %s; trying the next provider: %s",
                            primary,
                            model_name,
                            exc,
                        )
                if model_name not in _sessions:
                    raise RuntimeError(
                        f"No ONNX Runtime provider could load {model_name}: "
                        + "; ".join(errors)
                    )
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
                _ben2_session = create_ort_session(model_path, "BEN2 Base")
    return _ben2_session


def runtime_backend_info() -> dict:
    """Describe automatic acceleration without loading a model or benchmarking."""
    priority = available_provider_priority() or ["CPUExecutionProvider"]
    configured = priority[0]
    active = _active_providers.get("BEN2 Base")
    provider = active or configured
    accelerated = provider != "CPUExecutionProvider"
    dml_device = preferred_directml_device()
    gpu = detected_gpu()
    if provider == "DmlExecutionProvider" and dml_device:
        label = f"DirectML · {dml_device['name']}"
    else:
        label = PROVIDER_LABELS.get(provider, provider)
    if active:
        detail = f"Active: {label}"
    elif accelerated:
        detail = f"Ready: {label}"
    elif dml_device and not directml_is_suitable(dml_device):
        detail = f"CPU selected · {dml_device['name']} detected"
    elif gpu:
        detail = f"CPU fallback · {gpu['name']} detected"
    else:
        detail = "CPU fallback · install a GPU runtime to accelerate"
    return {
        "automatic": True,
        "accelerated": accelerated,
        "active": active is not None,
        "provider": provider,
        "label": label,
        "detail": detail,
        "detected_gpu": gpu["name"] if gpu else None,
    }


def predict_ben2_mask(image: Image.Image) -> Image.Image:
    """Predict a soft foreground matte with BEN2 at its native resolution."""
    original_size = image.size
    prepared = image.convert("RGB").resize(
        (1024, 1024),
        # Match BEN2's official torchvision Resize preprocessing.
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
    # Match the official ONNX implementation: bilinear output scaling avoids
    # the ringing that Lanczos can introduce around a high-contrast mask.
    return mask.resize(original_size, Image.Resampling.BILINEAR)


def _empty_mask(size):
    return Image.new("L", size, 0)


def limit_processing_size(
    image: Image.Image,
    max_pixels: int = MAX_PROCESSING_PIXELS,
) -> Image.Image:
    """Keep decoded uploads within a safe memory budget for mask processing."""
    width, height = image.size
    pixel_count = width * height
    if pixel_count <= max_pixels:
        return image

    scale = (max_pixels / pixel_count) ** 0.5
    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    return image.resize(resized_size, Image.Resampling.LANCZOS)


def feedback_example_count() -> int:
    """Return the number of locally saved Fast-mode training examples."""
    if not os.path.isdir(FEEDBACK_DATA_DIR):
        return 0
    return sum(
        entry.is_dir()
        for entry in os.scandir(FEEDBACK_DATA_DIR)
    )


def background_profile(image: Image.Image) -> dict | None:
    """Describe a flat, graphic background if this image has one.

    A saliency model reports what looks like a subject, which is the wrong
    question for a flat-colour illustration: there, anything that is not the
    background colour is content, including icons and badges the model treats as
    decoration. Detecting that case exactly is what makes it safe to key the
    background by colour instead of trusting the model alone.

    The test is deliberately narrow. The border has to be one tight colour
    cluster, and the areas matching it must be free of photographic noise, so
    photographs - even ones shot against a plain wall - do not qualify.
    """
    sample = image.convert("RGB")
    sample.thumbnail((512, 512), Image.Resampling.LANCZOS)
    pixels = np.asarray(sample, dtype=np.float32)
    height, width = pixels.shape[:2]
    if height < 32 or width < 32:
        return None

    border = np.ones((height, width), dtype=bool)
    border[
        round(height * 0.06) : round(height * 0.94),
        round(width * 0.06) : round(width * 0.94),
    ] = False
    border_pixels = pixels[border].reshape(-1, 3)
    color = np.median(border_pixels, axis=0)

    flatness = float(
        (np.linalg.norm(border_pixels - color, axis=1) < 24.0).mean()
    )
    distance = np.linalg.norm(pixels - color, axis=2)
    matching = distance < 24.0
    if matching.sum() < 64:
        return None
    gray = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    detail = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3))
    noise = float(np.percentile(detail[matching], 75))

    # Measured across the sample set: flat illustrations sit at 69-98% border
    # flatness with noise under 9, while photographs reach at most 57% flatness
    # and 12 or more noise. Both conditions must hold.
    if flatness < 0.65 or noise > 10.0:
        return None
    return {"color": color, "flatness": flatness, "noise": noise}


def fill_graphic_interiors(
    image: Image.Image,
    mask: Image.Image,
    profile: dict,
) -> Image.Image:
    """Restore icon and badge fills a saliency model treated as decoration.

    On a flat-colour illustration the model reliably keeps an icon's outline and
    glyph but discards the coloured disc between them, because that disc shares
    the page's hue. Those discs are *enclosed* by kept pixels, which is what
    separates them from the soft shadows the artwork casts onto the background:
    a shadow is open to the background and is never enclosed.

    A hole is filled only when its own colour is decisively unlike the
    background, so genuine see-through gaps - the space inside a handle, the gap
    between two figures - keep showing through.
    """
    alpha = np.asarray(mask.convert("L"), dtype=np.uint8)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    kept = alpha >= 128
    if not kept.any():
        return mask

    holes = ndimage.binary_fill_holes(kept) & ~kept
    if not holes.any():
        return mask

    # A compressed source needs more headroom before a difference is real.
    tolerance = max(20.0, profile["noise"] * 2.6)
    labels, count = ndimage.label(holes)
    filled = alpha.copy()
    for index, bounds in enumerate(ndimage.find_objects(labels), start=1):
        if bounds is None:
            continue
        region = labels[bounds] == index
        area = int(np.count_nonzero(region))
        if area < 12:
            continue
        patch = rgb[bounds][region]
        distance = float(
            np.median(np.linalg.norm(patch - profile["color"], axis=1))
        )
        if distance <= tolerance:
            continue  # the hole shows the page through, so it stays open
        window = filled[bounds]
        window[region] = 255
        filled[bounds] = window
    return Image.fromarray(filled, mode="L")


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


def find_screen_panel_holes(mask: np.ndarray) -> np.ndarray:
    """Recover missing upper screen panels without filling ordinary gaps."""
    support = mask >= 12
    if not support.any():
        return np.zeros(support.shape, dtype=bool)

    holes = ndimage.binary_fill_holes(support) & ~support
    labels, component_count = ndimage.label(holes)
    if not component_count:
        return holes

    height, width = support.shape
    minimum_width = max(16, round(width * 0.05))
    maximum_height = max(10, round(height * 0.12))
    maximum_bottom = round(height * 0.58)
    panels = np.zeros_like(holes)
    for label_id, bounds in enumerate(ndimage.find_objects(labels), start=1):
        if bounds is None:
            continue
        y_slice, x_slice = bounds
        panel_height = y_slice.stop - y_slice.start
        panel_width = x_slice.stop - x_slice.start
        aspect_ratio = panel_width / max(1, panel_height)
        if (
            panel_width >= minimum_width
            and panel_height <= maximum_height
            and y_slice.stop <= maximum_bottom
            and aspect_ratio >= 1.5
        ):
            panels[y_slice, x_slice] |= labels[y_slice, x_slice] == label_id
    return panels


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
    screen_panels = find_screen_panel_holes(rough)
    candidate = rough >= 12
    foreground_seed = cv2.erode(
        (recovered_small >= 180).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    ).astype(bool)
    if not foreground_seed.any():
        return ai_mask

    grabcut_mask = np.full(rough.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[candidate] = cv2.GC_PR_FGD
    grabcut_mask[rough <= 2] = cv2.GC_BGD
    grabcut_mask[foreground_seed] = cv2.GC_FGD
    grabcut_mask[screen_panels] = cv2.GC_FGD
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
        return ai_mask

    keep = np.isin(grabcut_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    keep_image = Image.fromarray((keep * 255).astype(np.uint8))
    panel_image = Image.fromarray(
        screen_panels.astype(np.uint8) * 255,
    )
    if analysis_size != original.size:
        keep_image = keep_image.resize(original.size, Image.Resampling.NEAREST)
        panel_image = panel_image.resize(
            original.size,
            Image.Resampling.NEAREST,
        )
    combined_image = Image.fromarray(combined)
    panel_combined = Image.composite(
        Image.new("L", original.size, 255),
        combined_image,
        panel_image,
    )
    gated_recovery = Image.composite(
        panel_combined,
        _empty_mask(original.size),
        keep_image,
    )
    # Recovery is an additive correction. It must never erase a pixel BEN2
    # already selected, even if the color gate disagrees with the model.
    safely_combined = np.maximum(
        ai,
        np.asarray(gated_recovery.convert("L"), dtype=np.uint8),
    )
    return Image.fromarray(safely_combined, mode="L")


def solve_alpha_bounded(
    rgb: np.ndarray,
    trimap: np.ndarray,
    maxiter: int = 260,
    max_pixels: int = CF_SOLVE_MAX_PIXELS,
) -> np.ndarray:
    """Closed-form alpha matting with a bounded memory footprint.

    The solver builds a sparse Laplacian holding 25 entries per pixel in two
    8-byte arrays, so it needs roughly 400 bytes for every pixel handed to it:
    a 10-megapixel photograph asks for about 4 GB and fails on an ordinary
    machine. Alpha is smooth at the scale the solver contributes, so an oversized
    problem is solved on a reduced copy and scaled back rather than refused.
    """
    height, width = trimap.shape[:2]
    pixels = height * width
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        reduced = (max(8, round(width * scale)), max(8, round(height * scale)))
        # Resize first and widen to double precision afterwards: converting a
        # 10-megapixel region before reducing it would itself cost 250 MB.
        rgb_input = cv2.resize(
            np.ascontiguousarray(rgb, dtype=np.float32),
            reduced,
            interpolation=cv2.INTER_AREA,
        ).astype(np.float64)
        # Nearest keeps the trimap's three exact levels; anything smoother would
        # invent unknown pixels along the known/unknown boundary.
        trimap_input = cv2.resize(
            np.ascontiguousarray(trimap, dtype=np.float32),
            reduced,
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float64)
    else:
        rgb_input = np.asarray(rgb, dtype=np.float64)
        trimap_input = np.asarray(trimap, dtype=np.float64)
    # Resampling can leave a value a hair outside the unit range.
    np.clip(rgb_input, 0.0, 1.0, out=rgb_input)
    np.clip(trimap_input, 0.0, 1.0, out=trimap_input)

    try:
        solved = estimate_alpha_cf(
            rgb_input,
            trimap_input,
            cg_kwargs={"maxiter": maxiter},
        )
    except (ArithmeticError, ValueError, MemoryError):
        # Falling back to the trimap's own committed values keeps the caller
        # working; it loses the colour-aware refinement, nothing else.
        return np.clip(np.asarray(trimap, dtype=np.float64), 0.0, 1.0)

    if solved.shape[:2] != (height, width):
        solved = cv2.resize(
            solved.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float64)
    return solved


def refine_precision_matte(
    rgb: np.ndarray,
    model_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine BEN2's transition using local source colors in a narrow ROI.

    BEN2 already supplies a strong soft matte. A full closed-form solve over the
    entire image is both slow and prone to widening a correct model boundary.
    This pass estimates nearby foreground/background colors with normalized
    Gaussian filters, uses them only where the color separation is reliable,
    and leaves intentional low-contrast transparency intact.
    """
    alpha = model_alpha.astype(np.float32, copy=True)
    foreground = rgb.astype(np.float32, copy=True)
    candidate_y, candidate_x = np.nonzero(alpha > 0.003)
    if not len(candidate_x):
        return alpha, foreground

    height, width = alpha.shape
    sigma = min(7.0, max(2.0, min(height, width) * 0.003))
    padding = max(12, round(sigma * 4))
    left = max(0, int(candidate_x.min()) - padding)
    right = min(width, int(candidate_x.max()) + padding + 1)
    top = max(0, int(candidate_y.min()) - padding)
    bottom = min(height, int(candidate_y.max()) + padding + 1)

    alpha_roi = alpha[top:bottom, left:right]
    rgb_roi = rgb[top:bottom, left:right].astype(np.float32, copy=False)
    foreground_weight = np.clip((alpha_roi - 0.25) / 0.60, 0.0, 1.0) ** 2
    background_weight = np.clip((0.75 - alpha_roi) / 0.60, 0.0, 1.0) ** 2
    foreground_denominator = cv2.GaussianBlur(
        foreground_weight,
        (0, 0),
        sigma,
    ) + 1e-4
    background_denominator = cv2.GaussianBlur(
        background_weight,
        (0, 0),
        sigma,
    ) + 1e-4

    local_foreground = np.empty_like(rgb_roi)
    local_background = np.empty_like(rgb_roi)
    for channel in range(3):
        local_foreground[:, :, channel] = cv2.GaussianBlur(
            rgb_roi[:, :, channel] * foreground_weight,
            (0, 0),
            sigma,
        ) / foreground_denominator
        local_background[:, :, channel] = cv2.GaussianBlur(
            rgb_roi[:, :, channel] * background_weight,
            (0, 0),
            sigma,
        ) / background_denominator

    color_direction = local_foreground - local_background
    color_distance_squared = np.sum(color_direction * color_direction, axis=2)
    color_alpha = np.sum(
        (rgb_roi - local_background) * color_direction,
        axis=2,
    ) / np.maximum(color_distance_squared, 1e-4)
    color_alpha = np.clip(color_alpha, 0.0, 1.0)

    uncertainty = np.clip(1.0 - np.abs(alpha_roi - 0.5) * 2.0, 0.0, 1.0)
    color_contrast = np.clip(
        (np.sqrt(color_distance_squared) - 0.035) / 0.20,
        0.0,
        1.0,
    )
    # Both color estimates must have real nearby seed support. This prevents a
    # broad glass/veil gradient from looking like a high-contrast edge merely
    # because one normalized blur has almost no samples in its window.
    seed_support = np.clip(
        np.minimum(foreground_denominator, background_denominator) / 0.025,
        0.0,
        1.0,
    )
    color_contrast *= seed_support
    solver_weight = 0.82 * uncertainty * color_contrast
    refined = alpha_roi * (1.0 - solver_weight) + color_alpha * solver_weight

    # Tighten definite, high-contrast contours without converting hair, glass,
    # motion blur, or other low-contrast translucency into a hard binary edge.
    local_range = (
        cv2.dilate(alpha_roi, np.ones((3, 3), dtype=np.uint8))
        - cv2.erode(alpha_roi, np.ones((3, 3), dtype=np.uint8))
    )
    smoothstep = refined * refined * (3.0 - 2.0 * refined)
    shape_weight = 0.58 * color_contrast * np.clip(local_range / 0.08, 0.0, 1.0)
    refined = refined * (1.0 - shape_weight) + smoothstep * shape_weight
    alpha_blur = cv2.GaussianBlur(refined, (0, 0), 0.55)
    refined += (
        0.42
        * (refined - alpha_blur)
        * np.clip(local_range / 0.06, 0.0, 1.0)
    )
    refined = np.clip(refined, 0.0, 1.0)
    refined[alpha_roi <= 0.006] = 0.0
    refined[alpha_roi >= 0.994] = 1.0
    alpha[top:bottom, left:right] = refined

    # Remove the old background color from translucent boundary pixels. The
    # local foreground estimate bounds the unmixing operation and prevents the
    # bright/dark color spikes common in naive alpha division.
    safe_alpha = np.maximum(refined[:, :, None], 0.08)
    unmixed = (
        rgb_roi - (1.0 - refined[:, :, None]) * local_background
    ) / safe_alpha
    unmixed = np.clip(unmixed, 0.0, 1.0)
    decontaminated = 0.72 * unmixed + 0.28 * local_foreground
    color_weight = np.clip(
        (1.0 - refined) * 0.86 * color_contrast,
        0.0,
        0.82,
    )
    color_weight[(refined <= 0.01) | (refined >= 0.995)] = 0.0
    foreground[top:bottom, left:right] = (
        rgb_roi * (1.0 - color_weight[:, :, None])
        + decontaminated * color_weight[:, :, None]
    )
    return alpha, np.clip(foreground, 0.0, 1.0)


def finish_cutout(
    original: Image.Image,
    combined_mask: Image.Image,
    max_side: int = MATTING_MAX_SIDE,
    precision: bool = False,
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
    model_alpha = rough_alpha.astype(np.float64) / 255.0
    if not np.any(rough_alpha >= 4):
        result = original.copy()
        result.putalpha(_empty_mask(original_size))
        return result

    if precision:
        alpha, foreground = refine_precision_matte(
            rgb.astype(np.float32),
            model_alpha.astype(np.float32),
        )
    else:
        alpha = None
        foreground = None

    kernel = np.ones((3, 3), dtype=np.uint8)
    # BEN2 already predicts a useful soft mask. Treat only its genuinely
    # uncertain pixels as a matting problem; the old broad unknown band let the
    # closed-form solver soften large parts of an otherwise crisp model edge.
    high_confidence = (rough_alpha >= 246).astype(np.uint8)
    sure_foreground = cv2.erode(
        high_confidence,
        kernel,
        iterations=1,
    ).astype(bool)
    distance = cv2.distanceTransform(high_confidence, cv2.DIST_L2, 5)
    local_maximum = distance >= (
        cv2.dilate(distance, kernel, iterations=1) - 1e-4
    )
    thin_centerline = local_maximum & (distance >= 0.95)
    sure_foreground |= thin_centerline
    sure_background = rough_alpha <= 4

    trimap = np.full(rough_alpha.shape, 0.5, dtype=np.float64)
    trimap[sure_background] = 0.0
    trimap[sure_foreground] = 1.0

    if not precision:
        try:
            solved_alpha = solve_alpha_bounded(rgb, trimap, maxiter=260)
            unknown = ~(sure_background | sure_foreground)
            # Trust source-color matting most near the ambiguous 50% contour,
            # but retain more of the model near transparent/opaque endpoints.
            confidence = np.abs(model_alpha - 0.5) * 2.0
            solver_weight = np.clip(0.78 - 0.5 * confidence, 0.28, 0.78)
            alpha = model_alpha.copy()
            alpha[unknown] = (
                model_alpha[unknown] * (1.0 - solver_weight[unknown])
                + solved_alpha[unknown] * solver_weight[unknown]
            )
            alpha[sure_background] = 0.0
            alpha[sure_foreground] = 1.0
        except (ArithmeticError, ValueError):
            alpha = model_alpha

    # Recover high-frequency alpha detail lost by 1024px model output scaling.
    # The local-range gate prevents the unsharp pass from creating texture in
    # flat translucent areas such as glass or motion blur.
    alpha32 = alpha.astype(np.float32)
    blurred_alpha = cv2.GaussianBlur(alpha32, (0, 0), 0.65)
    local_max = cv2.dilate(alpha32, kernel)
    local_min = cv2.erode(alpha32, kernel)
    structured_edge = (local_max - local_min) >= 0.035
    alpha[structured_edge] = np.clip(
        alpha32[structured_edge]
        + 0.62 * (alpha32[structured_edge] - blurred_alpha[structured_edge]),
        0.0,
        1.0,
    )
    alpha[alpha < 0.006] = 0.0
    alpha[alpha > 0.994] = 1.0

    if not precision:
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
        alpha_image = alpha_image.resize(original_size, Image.Resampling.BICUBIC)
        foreground_image = foreground_image.resize(
            original_size,
            Image.Resampling.LANCZOS,
        )

    result = np.asarray(original.convert("RGBA")).copy()
    alpha_full = np.asarray(alpha_image, dtype=np.uint8)
    foreground_full = np.asarray(foreground_image, dtype=np.uint8)
    edge_pixels = (alpha_full > 0) & (alpha_full < 250)
    # Foreground estimation removes background-color contamination, but
    # replacing every edge RGB value can make the visible boundary look soft.
    # Blend it into the source in proportion to transparency instead.
    blend = np.clip(
        (1.0 - alpha_full.astype(np.float32) / 255.0) * 0.78,
        0.0,
        0.72,
    )
    for channel in range(3):
        source_channel = result[:, :, channel].astype(np.float32)
        refined_channel = foreground_full[:, :, channel].astype(np.float32)
        mixed = source_channel * (1.0 - blend) + refined_channel * blend
        result[:, :, channel][edge_pixels] = np.clip(
            mixed[edge_pixels],
            0,
            255,
        ).astype(np.uint8)

    source_alpha = np.asarray(original.getchannel("A"), dtype=np.uint16)
    final_alpha = (
        alpha_full.astype(np.uint16) * source_alpha // 255
    ).astype(np.uint8)
    result[:, :, 3] = final_alpha
    return Image.fromarray(result)


def detect_natural_shadow(
    original: Image.Image,
    subject_alpha: Image.Image,
    max_side: int = 1000,
) -> Image.Image:
    """Estimate a conservative contact/cast-shadow alpha near the subject.

    A single image cannot identify every shadow reliably. This pass therefore
    keeps only smooth luminance deficits close to the lower part of the subject,
    with similar local chroma and component contact. Complex or dark background
    texture is intentionally rejected instead of being restored as foreground.
    """
    original_size = original.size
    scale = min(1.0, max_side / max(original_size))
    analysis_size = (
        max(1, round(original_size[0] * scale)),
        max(1, round(original_size[1] * scale)),
    )
    rgb_image = original.convert("RGB")
    alpha_image = subject_alpha.convert("L")
    if analysis_size != original_size:
        rgb_image = rgb_image.resize(analysis_size, Image.Resampling.LANCZOS)
        alpha_image = alpha_image.resize(analysis_size, Image.Resampling.BILINEAR)

    rgb = np.asarray(rgb_image, dtype=np.uint8)
    subject = np.asarray(alpha_image, dtype=np.uint8) >= 96
    height, width = subject.shape
    subject_pixels = int(np.count_nonzero(subject))
    if subject_pixels < 24 or subject_pixels > subject.size * 0.78:
        return _empty_mask(original_size)

    selected_y, selected_x = np.nonzero(subject)
    top, bottom = int(selected_y.min()), int(selected_y.max())
    subject_height = max(1, bottom - top + 1)
    if bottom >= height - max(2, round(height * 0.008)):
        return _empty_mask(original_size)

    def odd(value, minimum, maximum):
        value = min(maximum, max(minimum, int(round(value))))
        return value if value % 2 else value + 1

    zone_width = odd(width * 0.28, 31, 241)
    zone_height = odd(height * 0.22, 25, 161)
    zone_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (zone_width, zone_height),
    )
    near_subject = cv2.dilate(subject.astype(np.uint8), zone_kernel).astype(bool)
    rows = np.arange(height)[:, None]
    # Ground shadows originate at the support/contact region. Searching around
    # the whole lower half of a portrait can incorrectly classify ordinary
    # background lighting as shadow, so stay close to the subject base.
    lower_subject_area = rows >= bottom - round(subject_height * 0.12)
    shadow_zone = near_subject & lower_subject_area

    exclusion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    subject_exclusion = cv2.dilate(
        subject.astype(np.uint8),
        exclusion_kernel,
        iterations=1,
    ).astype(bool)
    shadow_zone &= ~subject_exclusion

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance = lab[:, :, 0]
    # The closing footprint must be wider than a typical soft shadow band;
    # smaller kernels follow the dark region instead of estimating the lit
    # surface that would exist beneath it.
    close_size = odd(min(height, width) * 0.34, 31, 151)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size),
    )
    bright_envelope = cv2.dilate(luminance, close_kernel)
    lit_luminance = cv2.GaussianBlur(
        bright_envelope,
        (0, 0),
        max(1.2, close_size / 18.0),
    )
    # On smooth studio/tabletop backgrounds, a quadratic surface fitted to the
    # brighter nearby pixels provides the missing luminance under broad cast
    # shadows that even a large local maximum filter cannot span.
    background_samples = near_subject & ~subject_exclusion
    background_is_smooth = False
    if np.count_nonzero(background_samples) >= 100:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        median_saturation = float(np.median(hsv[:, :, 1][background_samples]))
        median_luminance = float(np.median(luminance[background_samples]))
        # A portable black-alpha layer also works on moderately colored
        # surfaces because real shadows primarily reduce luminance. Reject only
        # very saturated or dark scenes where a single-image estimate is too
        # ambiguous to move onto a different background safely.
        if median_saturation > 180.0 or median_luminance < 72.0:
            return _empty_mask(original_size)
        fit_y, fit_x = np.nonzero(background_samples)
        sample_step = max(1, len(fit_x) // 30_000)
        fit_x = fit_x[::sample_step]
        fit_y = fit_y[::sample_step]
        normalized_x = fit_x.astype(np.float32) / max(1, width - 1)
        normalized_y = fit_y.astype(np.float32) / max(1, height - 1)
        features = np.column_stack(
            (
                np.ones_like(normalized_x),
                normalized_x,
                normalized_y,
                normalized_x ** 2,
                normalized_x * normalized_y,
                normalized_y ** 2,
            )
        )
        sample_luminance = luminance[fit_y, fit_x]
        active_samples = np.ones(len(fit_x), dtype=bool)
        coefficients = None
        # Iteratively remove dark outliers (the possible shadow) while keeping
        # gradual lighting/background gradients in the fitted surface.
        for _ in range(3):
            if np.count_nonzero(active_samples) < 80:
                break
            coefficients = np.linalg.lstsq(
                features[active_samples],
                sample_luminance[active_samples],
                rcond=None,
            )[0]
            residual = sample_luminance - features @ coefficients
            centered = residual - np.median(residual[active_samples])
            robust_scale = max(
                1.5,
                float(np.median(np.abs(centered[active_samples]))) * 1.4826,
            )
            active_samples = residual >= -max(4.0, robust_scale * 1.1)
        if coefficients is None or np.count_nonzero(active_samples) < 80:
            coefficients = np.zeros(6, dtype=np.float32)
            active_samples = np.ones(len(fit_x), dtype=bool)
        fitted_samples = features @ coefficients
        fit_error = float(
            np.percentile(
                np.abs(
                    fitted_samples[active_samples]
                    - sample_luminance[active_samples]
                ),
                85,
            )
        )
        if fit_error <= 11.5:
            background_is_smooth = True
            grid_y, grid_x = np.mgrid[0:height, 0:width]
            grid_x = grid_x.astype(np.float32) / max(1, width - 1)
            grid_y = grid_y.astype(np.float32) / max(1, height - 1)
            fitted_luminance = (
                coefficients[0]
                + coefficients[1] * grid_x
                + coefficients[2] * grid_y
                + coefficients[3] * grid_x ** 2
                + coefficients[4] * grid_x * grid_y
                + coefficients[5] * grid_y ** 2
            )
            lit_luminance = np.maximum(lit_luminance, fitted_luminance)
    deficit = np.clip(
        (lit_luminance - luminance) / np.maximum(lit_luminance, 24.0),
        0.0,
        1.0,
    )

    chroma_sigma = max(2.0, close_size / 8.0)
    local_a = cv2.GaussianBlur(lab[:, :, 1], (0, 0), chroma_sigma)
    local_b = cv2.GaussianBlur(lab[:, :, 2], (0, 0), chroma_sigma)
    chroma_delta = np.hypot(lab[:, :, 1] - local_a, lab[:, :, 2] - local_b)
    gradient_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gradient_x, gradient_y) / 1020.0

    candidate = (
        shadow_zone
        & background_is_smooth
        & (lit_luminance >= 38.0)
        & (deficit >= 0.025)
        & (deficit <= 0.58)
        & (chroma_delta <= 24.0)
        & (gradient <= 0.30)
    )
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)

    contact_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 9))
    contact_ring = cv2.dilate(
        subject.astype(np.uint8),
        contact_kernel,
    ).astype(bool) & ~subject_exclusion
    component_count, labels = cv2.connectedComponents(
        candidate.astype(np.uint8),
        connectivity=8,
    )
    keep = np.zeros(candidate.shape, dtype=bool)
    minimum_area = max(10, round(subject.size * 0.000025))
    for component_id in range(1, component_count):
        component = labels == component_id
        area = int(np.count_nonzero(component))
        if area < minimum_area:
            continue
        touches_subject = np.any(component & contact_ring)
        strong_cast = float(np.mean(deficit[component])) >= 0.10
        if touches_subject or strong_cast:
            keep |= component

    shadow = np.zeros(candidate.shape, dtype=np.float32)
    shadow[keep] = np.clip(
        (deficit[keep] - 0.018) * 1.02,
        0.0,
        0.52,
    )
    shadow = cv2.GaussianBlur(shadow, (0, 0), 1.15)
    shadow[subject_exclusion] = 0.0
    shadow[shadow < 0.008] = 0.0
    shadow_image = Image.fromarray(
        np.clip(shadow * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    )
    if analysis_size != original_size:
        shadow_image = shadow_image.resize(
            original_size,
            Image.Resampling.BILINEAR,
        )
    return shadow_image


def preserve_natural_shadow(
    original: Image.Image,
    cutout: Image.Image,
    shadow_image: Image.Image | None = None,
) -> Image.Image:
    """Composite a portable dark shadow behind the finished subject."""
    cutout = cutout.convert("RGBA")
    if shadow_image is None:
        shadow_image = detect_natural_shadow(original, cutout.getchannel("A"))
    shadow_alpha = np.asarray(shadow_image, dtype=np.uint8)
    if not np.any(shadow_alpha):
        return cutout

    result = np.asarray(cutout, dtype=np.uint8).copy()
    subject_alpha = result[:, :, 3].astype(np.float32) / 255.0
    shadow = shadow_alpha.astype(np.float32) / 255.0
    shadow_behind = shadow * (1.0 - subject_alpha)
    combined_alpha = subject_alpha + shadow_behind
    active = combined_alpha > 1e-5

    # Straight-alpha black reproduces a luminance deficit on light or colored
    # exports. Source-over compositing keeps every opaque subject pixel exactly
    # unchanged while also rescuing shadows where BEN2 left a weak, tinted
    # response instead of an entirely transparent pixel.
    subject_rgb = result[:, :, :3].astype(np.float32) / 255.0
    combined_rgb = np.zeros_like(subject_rgb)
    combined_rgb[active] = (
        subject_rgb[active]
        * subject_alpha[active, None]
        / combined_alpha[active, None]
    )
    result[:, :, :3] = np.clip(combined_rgb * 255.0, 0, 255).astype(np.uint8)
    result[:, :, 3] = np.clip(combined_alpha * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGBA")


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

    trimap = np.zeros(target.shape, dtype=np.float32)
    trimap[outer_edge] = 0.5
    trimap[sure_foreground] = 1.0
    # Single precision here: the solver widens whatever it keeps, and a
    # full-image stroke would otherwise hold a 250 MB double-precision copy.
    alpha = solve_alpha_bounded(
        rgb.astype(np.float32) / np.float32(255.0),
        trimap,
        maxiter=300,
    )
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
        # Brush coordinates come from the image currently shown in the editor.
        # The automatic cutout may be reduced to the processing-size limit, so
        # resizing the edited result back to the upload's dimensions makes a
        # stroke land at the wrong place.  Keep the editor/result dimensions as
        # the working coordinate space and bring the source image to that size.
        original = original.resize(current.size, Image.Resampling.LANCZOS)

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
    # Segment at a reduced scale when the stroke covers a large region. What
    # GrabCut contributes is the boundary's location at object scale, which
    # survives the reduction; the alpha edge is rebuilt at full scale below.
    roi_pixels = grabcut_mask.shape[0] * grabcut_mask.shape[1]
    if roi_pixels > GRABCUT_MAX_PIXELS:
        cut_scale = (GRABCUT_MAX_PIXELS / roi_pixels) ** 0.5
        cut_size = (
            max(16, round(grabcut_mask.shape[1] * cut_scale)),
            max(16, round(grabcut_mask.shape[0] * cut_scale)),
        )
        small_rgb = cv2.resize(roi_rgb, cut_size, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(
            grabcut_mask,
            cut_size,
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.grabCut(
            small_rgb,
            small_mask,
            None,
            background_model,
            foreground_model,
            2,
            cv2.GC_INIT_WITH_MASK,
        )
        grabcut_mask = cv2.resize(
            small_mask,
            (grabcut_mask.shape[1], grabcut_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
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
    disables this pass so normal photo removal stays with the semantic model.
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
    # than a smooth graphic. In that case, the semantic model remains safer.
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
    # Do not open the mask here: at the Fast model's analysis resolution, a
    # 3x3 opening deletes legitimate dots, dashed lines, and thin UI edges.
    # Closing still reconnects antialiased strokes without erasing them.
    recovered = ndimage.binary_closing(
        recovered,
        structure=np.ones((3, 3)),
    )

    labels, component_count = ndimage.label(recovered)
    if component_count:
        component_sizes = np.bincount(labels.ravel())
        minimum_size = max(12, round(width * height * 0.00003))
        fine_detail_size = max(3, round(width * height * 0.000006))
        keep_component = component_sizes >= minimum_size
        for label_id in range(1, component_count + 1):
            component_size = component_sizes[label_id]
            if component_size < fine_detail_size or keep_component[label_id]:
                continue
            component_residual = residual[labels == label_id]
            # Preserve compact, high-contrast details while discarding weak
            # smooth-background speckles that recovery can occasionally add.
            if float(np.percentile(component_residual, 75)) >= (
                residual_threshold * 1.35
            ):
                keep_component[label_id] = True
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


def release_session(quality: str) -> None:
    """Drop a cached model session and its runtime arena.

    Comparing two models means loading two of them. A segmentation model holds
    its weights plus an execution arena sized for its own activations - the
    lightweight BiRefNet alone asks for an 800 MB buffer - so on an 8 GB machine
    keeping the first one resident makes the second fail to allocate. Releasing
    each candidate as soon as its matte is in hand keeps the peak at one model.
    """
    global _ben2_session
    profile = QUALITY_PROFILES.get(quality)
    with _session_lock:
        if profile is not None and profile["engine"] == "ben2":
            _ben2_session = None
        elif profile is not None:
            _sessions.pop(profile["model"], None)
    gc.collect()


def predict_model_mask(image: Image.Image, quality: str) -> Image.Image:
    """Run one quality profile's segmentation model and return its raw matte."""
    profile = QUALITY_PROFILES[quality]
    with _inference_lock:
        if profile["engine"] == "ben2":
            mask = predict_ben2_mask(image)
        else:
            mask = remove(
                image.convert("RGB"),
                session=get_session(profile["model"]),
                only_mask=True,
                alpha_matting=False,
                post_process_mask=False,
            ).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return mask


def mask_edge_agreement(image: Image.Image, mask: Image.Image) -> float:
    """How much of a matte's own boundary lies on a real edge in the photograph.

    A model that has washed a neighbouring object into the subject has to end
    its matte somewhere the picture does not change - through the middle of a
    blurred car, say - while a correct matte ends where the image does. Scoring
    that is what lets the app compare two models' answers without a human
    looking at them.

    Measured on a traffic photograph where one model kept a second vehicle and
    another did not, this ranked the correct model first by a 34% margin.
    """
    scale = min(1.0, 1200.0 / max(image.size))
    size = (
        max(16, round(image.width * scale)),
        max(16, round(image.height * scale)),
    )
    gray = np.asarray(
        image.convert("L").resize(size, Image.Resampling.BOX),
        dtype=np.float32,
    )
    gradient = np.hypot(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    # Normalising by a high percentile keeps the score comparable between a
    # high-contrast photograph and a soft one.
    reference = float(np.percentile(gradient, 99))
    if reference <= 1e-6:
        return 0.0
    gradient = np.clip(gradient / reference, 0.0, 1.0)

    small_mask = np.asarray(
        mask.convert("L").resize(size, Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    kept = (small_mask > 128).astype(np.uint8)
    if not kept.any() or kept.all():
        return 0.0
    boundary = cv2.morphologyEx(
        kept,
        cv2.MORPH_GRADIENT,
        np.ones((5, 5), dtype=np.uint8),
    ) > 0
    if not boundary.any():
        return 0.0
    return float(gradient[boundary].mean())


def choose_automatic_mask(image: Image.Image) -> tuple:
    """Pick the model for this image, and return its matte and the choice.

    Two things are decided here. A flat-colour illustration is routed straight
    to the precision model, which measured best or joint-best on every
    illustration tested, and needs no comparison. A photograph is segmented by
    two models and the one whose boundary better follows the picture is kept.

    Near-identical scores mean the metric cannot tell the candidates apart -
    on a wispy-hair test the two leaders were within 0.02% - so a tie is
    resolved in favour of the precision model rather than by noise.
    """
    graphic = background_profile(image)
    if graphic is not None:
        # One model is enough here, but if it cannot run the request should
        # still produce a cutout rather than fail.
        for quality in (AUTO_PREFERRED_QUALITY, *AUTO_PHOTO_CANDIDATES):
            try:
                return quality, predict_model_mask(image, quality), graphic
            except Exception as error:
                app.logger.warning(
                    "Automatic selection skipped %s on a graphic: %s",
                    quality,
                    error,
                )
                release_session(quality)
        raise RuntimeError(
            "No segmentation model could run on this image. "
            "Close other applications and try again."
        )

    scored = []
    failures = []
    for quality in AUTO_PHOTO_CANDIDATES:
        try:
            mask = predict_model_mask(image, quality)
        except Exception as error:  # a model that cannot run is
            # simply not a candidate; the comparison proceeds with the rest.
            app.logger.warning(
                "Automatic selection skipped %s: %s", quality, error
            )
            failures.append(quality)
            release_session(quality)
            continue
        scored.append((mask_edge_agreement(image, mask), quality, mask))
        # Free this model before loading the next one.
        release_session(quality)

    if not scored:
        raise RuntimeError(
            "No segmentation model could run on this image. "
            "Close other applications and try again."
        )
    scored.sort(key=lambda entry: entry[0], reverse=True)
    if len(scored) == 1:
        return scored[0][1], scored[0][2], None

    best_score, best_quality, best_mask = scored[0]
    if len(scored) > 1:
        runner_score, runner_quality, runner_mask = scored[1]
        margin = (best_score - runner_score) / max(runner_score, 1e-6)
        if margin < AUTO_TIE_MARGIN:
            for _, quality, mask in scored:
                if quality == AUTO_PREFERRED_QUALITY:
                    return quality, mask, None
    return best_quality, best_mask, None


def create_cutout(
    image_bytes: bytes,
    quality: str = DEFAULT_QUALITY,
    preserve_shadows: bool | None = None,
) -> tuple:
    """Return a transparent PNG and the quality profile that produced it."""
    requested = quality
    if quality != AUTO_QUALITY and quality not in QUALITY_PROFILES:
        raise ValueError("Choose a valid removal quality.")

    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        source = ImageOps.exif_transpose(source)
        original = limit_processing_size(source).convert("RGBA")

    graphic_background = None
    automatic_mask = None
    if requested == AUTO_QUALITY:
        quality, automatic_mask, graphic_background = choose_automatic_mask(
            original
        )

    profile = QUALITY_PROFILES[quality]
    if preserve_shadows is None:
        preserve_shadows = profile["preserve_shadows"]

    # The legacy structure-recovery pass remains limited to Fast mode. It can
    # rescue simple graphic details missed by ISNet, but must not overrule BEN2
    # or BiRefNet because it can restore smooth background patches and shadows.
    recovered_elements = (
        recover_all_elements(original)
        if profile["recover_structures"]
        else None
    )
    # Automatic selection has already run the winning model; running it again
    # would double the wait for no benefit.
    ai_mask = (
        automatic_mask
        if automatic_mask is not None
        else predict_model_mask(original, quality)
    )
    ai_mask = clean_ai_mask(ai_mask)

    # A flat-colour illustration is not a saliency problem: everything that is
    # not the background colour is content. Keying it recovers icons and badges
    # the model discards as decoration, and the gate keeps photographs out.
    if graphic_background is None:
        graphic_background = background_profile(original)
    if graphic_background is not None:
        ai_mask = fill_graphic_interiors(original, ai_mask, graphic_background)

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
    shadow_mask = (
        detect_natural_shadow(original, mask)
        if preserve_shadows
        else None
    )
    cutout = finish_cutout(
        original,
        mask,
        max_side=profile["matting_max_side"],
        precision=profile["engine"] == "ben2",
    )
    if preserve_shadows:
        cutout = preserve_natural_shadow(
            original,
            cutout,
            shadow_image=shadow_mask,
        )
    output = io.BytesIO()
    cutout.save(output, format="PNG", compress_level=4)
    return output.getvalue(), quality


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/info")
def api_info():
    """Expose stable product and model details for local clients."""
    return jsonify(
        {
            "name": APP_NAME,
            "version": APP_VERSION,
            "default_quality": DEFAULT_QUALITY,
            "max_upload_bytes": app.config["MAX_CONTENT_LENGTH"],
            "runtime": runtime_backend_info(),
            "automatic_quality": AUTO_QUALITY,
            "automatic_candidates": list(AUTO_PHOTO_CANDIDATES),
            "quality_profiles": {
                key: {
                    "label": profile["label"],
                    "model": profile["model"],
                    "preserve_shadows": profile["preserve_shadows"],
                }
                for key, profile in QUALITY_PROFILES.items()
            },
        }
    )


@app.post("/remove")
def remove_background():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose an image first."}), 400

    quality = request.form.get("quality", DEFAULT_QUALITY).strip().lower()
    if quality not in QUALITY_PROFILES and quality != AUTO_QUALITY:
        return jsonify({"error": "Choose a valid removal quality."}), 400
    shadow_value = request.form.get("preserve_shadows")
    if shadow_value is None:
        preserve_shadows = (
            QUALITY_PROFILES[quality]["preserve_shadows"]
            if quality in QUALITY_PROFILES
            else None
        )
    elif shadow_value.strip().lower() in {"1", "true", "yes", "on"}:
        preserve_shadows = True
    elif shadow_value.strip().lower() in {"0", "false", "no", "off"}:
        preserve_shadows = False
    else:
        return jsonify({"error": "Choose a valid shadow preservation option."}), 400

    image_bytes = uploaded.read()
    started = time.perf_counter()
    try:
        result, chosen_quality = create_cutout(
            image_bytes,
            quality,
            preserve_shadows=preserve_shadows,
        )
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "That file is not a supported image."}), 400
    except MemoryError:
        app.logger.exception("Background removal ran out of memory")
        return jsonify(
            {
                "error": (
                    "This image needed more memory than is available. Close other "
                    "applications, or try a smaller copy of the image."
                )
            }
        ), 507
    except Exception as exc:
        app.logger.exception("Background removal failed")
        return jsonify({"error": f"Background removal failed: {exc}"}), 500

    db.record_removal(
        filename=os.path.basename(uploaded.filename),
        requested_quality=quality,
        quality=chosen_quality,
        preserve_shadows=preserve_shadows,
        input_bytes=len(image_bytes),
        output_bytes=len(result),
        duration_ms=round((time.perf_counter() - started) * 1000),
        app_version=APP_VERSION,
    )

    stem = os.path.splitext(os.path.basename(uploaded.filename))[0] or "image"
    response = send_file(
        io.BytesIO(result),
        mimetype="image/png",
        as_attachment=False,
        download_name=f"{stem}-no-bg.png",
    )
    # The interface shows which profile Auto settled on for this image.
    response.headers["X-Cutout-Quality"] = chosen_quality
    return response


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


@app.post("/feedback")
def save_feedback():
    """Store an explicitly approved Fast-mode alpha mask for later training."""
    original_file = request.files.get("image")
    current_file = request.files.get("current")
    quality = request.form.get("quality", "").strip().lower()
    if original_file is None or current_file is None:
        return jsonify({"error": "The original image and final cutout are required."}), 400
    if quality != "fast":
        return jsonify({"error": "Feedback collection is available for Fast - ISNet only."}), 400

    try:
        with Image.open(original_file.stream) as original_source:
            original_source.load()
            original = limit_processing_size(
                ImageOps.exif_transpose(original_source)
            ).convert("RGB")
        with Image.open(current_file.stream) as current_source:
            current_source.load()
            current = current_source.convert("RGBA")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "The original image or final cutout is invalid."}), 400

    if current.size != original.size:
        current = current.resize(original.size, Image.Resampling.LANCZOS)

    example_id = uuid.uuid4().hex
    example_dir = os.path.join(FEEDBACK_DATA_DIR, example_id)
    try:
        os.makedirs(example_dir, exist_ok=False)
        original.save(
            os.path.join(example_dir, "original.png"),
            format="PNG",
            compress_level=4,
        )
        current.getchannel("A").save(
            os.path.join(example_dir, "alpha.png"),
            format="PNG",
            compress_level=4,
        )
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "original_filename": os.path.basename(original_file.filename),
            "quality": quality,
            "size": {"width": current.width, "height": current.height},
        }
        with open(
            os.path.join(example_dir, "metadata.json"),
            "w",
            encoding="utf-8",
        ) as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
    except OSError:
        app.logger.exception("Could not save Fast-mode feedback")
        return jsonify({"error": "Could not save the local feedback example."}), 500

    return jsonify(
        {
            "saved": True,
            "examples": feedback_example_count(),
        }
    )


@app.post("/export")
def export_cutout():
    current_file = request.files.get("current")
    output_format = request.form.get("format", "png").lower()
    if current_file is None:
        return jsonify({"error": "The current cutout is required."}), 400
    if output_format == "jpeg":
        output_format = "jpg"
    if output_format not in {"png", "webp", "jpg", "svg"}:
        return jsonify({"error": "Choose PNG, WebP, JPG, or SVG."}), 400

    try:
        with Image.open(current_file.stream) as current_source:
            current_source.load()
            current = current_source.convert("RGBA")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "The current cutout is invalid."}), 400

    background = request.form.get("background", "transparent").strip().lower()
    background_colors = {
        "white": (255, 255, 255),
        "black": (9, 10, 11),
    }
    if background == "custom":
        try:
            custom_color = request.form.get("background_color", "#ffffff")
            if not custom_color.startswith("#") or len(custom_color) not in {4, 7}:
                raise ValueError
            background_colors["custom"] = ImageColor.getrgb(custom_color)
        except ValueError:
            return jsonify({"error": "Choose a valid custom background color."}), 400
    elif background not in {"transparent", *background_colors}:
        return jsonify({"error": "Choose a valid export background."}), 400

    # JPEG cannot represent transparency, so use white when the transparent
    # preview is selected. Other formats retain alpha unless a color was chosen.
    export_image = current
    if background != "transparent" or output_format == "jpg":
        color = background_colors.get(background, background_colors["white"])
        backdrop = Image.new("RGBA", current.size, (*color, 255))
        export_image = Image.alpha_composite(backdrop, current).convert("RGB")

    stem = os.path.splitext(
        os.path.basename(request.form.get("filename", "image"))
    )[0] or "image"
    output = io.BytesIO()
    if output_format == "webp":
        export_image.save(output, format="WEBP", lossless=True, method=6)
        mimetype = "image/webp"
    elif output_format == "jpg":
        export_image.save(
            output,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=True,
        )
        mimetype = "image/jpeg"
    else:
        png = io.BytesIO()
        export_image.save(png, format="PNG", compress_level=4)
        if output_format == "png":
            output = png
            mimetype = "image/png"
        else:
            encoded = base64.b64encode(png.getvalue()).decode("ascii")
            svg = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{export_image.width}" height="{export_image.height}" '
                f'viewBox="0 0 {export_image.width} {export_image.height}">\n'
                f'  <image width="{export_image.width}" height="{export_image.height}" '
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


@app.get("/api/history")
def api_history():
    """List the most recent removals recorded in MongoDB."""
    if not db.configured():
        return jsonify({"error": "MongoDB is not configured."}), 503
    try:
        limit = min(100, max(1, int(request.args.get("limit", "20"))))
    except ValueError:
        return jsonify({"error": "limit must be a whole number."}), 400
    rows = db.recent_removals(limit)
    if rows is None:
        return jsonify({"error": "MongoDB is not reachable."}), 503
    return jsonify({"removals": rows})


@app.get("/health")
def health():
    # The default can be automatic selection, which has no single model until
    # an image decides it.
    profile = QUALITY_PROFILES.get(DEFAULT_QUALITY)
    return jsonify(
        {
            "status": "ok",
            "name": APP_NAME,
            "version": APP_VERSION,
            "default_quality": DEFAULT_QUALITY,
            "model": (
                profile["model"]
                if profile is not None
                else "selected per image from "
                + ", ".join(
                    QUALITY_PROFILES[candidate]["model"]
                    for candidate in AUTO_PHOTO_CANDIDATES
                )
            ),
            "pipeline": PIPELINE_NAME,
            "runtime": runtime_backend_info(),
            "database": db.status(),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  {APP_NAME} {APP_VERSION}")
    print(f"  Compute -> {runtime_backend_info()['detail']}")
    if db.configured():
        state = "connected" if db.ping() else "unreachable (continuing without it)"
        print(f"  MongoDB -> {db.database_name()} {state}")
    print(f"  Open  ->  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
