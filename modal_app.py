"""Deploy Cutout Studio to Modal:  modal deploy modal_app.py

The Flask app is served unchanged. Modal builds the image once (models baked
in), starts a container on the first request, and stops it again after it has
been idle, so nothing is billed while nobody is using the app.

Create the secret first (see the README):  modal secret create cutout-studio ...
"""

import modal

APP_DIR = "/root/app"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # onnxruntime needs the OpenMP runtime, which the slim image leaves out.
    .apt_install("libgomp1")
    .pip_install_from_requirements("requirements.txt")
    # Bake the models Auto mode uses into the image, so a cold start does not
    # download them. The other profiles are fetched on first use.
    .add_local_file("app.py", f"{APP_DIR}/app.py", copy=True)
    .add_local_file("db.py", f"{APP_DIR}/db.py", copy=True)
    .env(
        {
            "PYTHONPATH": APP_DIR,
            # Match the reserved cores; otherwise onnxruntime sizes its thread
            # pool from the whole host and the threads fight each other.
            "OMP_NUM_THREADS": "4",
        }
    )
    .run_commands(
        "python -c \"import os, pooch; from app import BEN2_MODEL_URL, BEN2_MODEL_HASH; "
        "pooch.retrieve(url=BEN2_MODEL_URL, known_hash=BEN2_MODEL_HASH, "
        "fname='BEN2_Base.onnx', path=os.path.expanduser('~/.u2net'))\"",
        "python -c \"from rembg import new_session; new_session('birefnet-general-lite')\"",
    )
    # PyMatting compiles its numeric kernels with Numba the first time it is
    # imported, which took over a minute in every fresh container. Compile them
    # once here and keep the cache in the image. "generic" makes the compiled
    # code portable, so the cache is valid on whichever CPU Modal starts us on.
    .env(
        {
            "NUMBA_CACHE_DIR": "/root/numba_cache",
            "NUMBA_CPU_NAME": "generic",
        }
    )
    .run_commands(
        "python -c \"import numpy as np; "
        "from pymatting import estimate_alpha_cf, estimate_foreground_ml; "
        "rgb = np.random.rand(48, 48, 3); tri = np.full((48, 48), 0.5); "
        "tri[:12] = 0.0; tri[36:] = 1.0; "
        "alpha = estimate_alpha_cf(rgb, tri); estimate_foreground_ml(rgb, alpha)\"",
    )
    # Keep both Auto-mode models loaded instead of reloading them for every
    # request. Measured on this container: 15-23% faster, byte-identical output.
    # Kept last so changing it does not rebuild the model layers above.
    .env({"REMOVE_BG_KEEP_MODELS": "1"})
    .add_local_dir(
        ".",
        APP_DIR,
        ignore=[
            ".git",
            ".env",
            ".env.*",
            "**/__pycache__",
            "tests",
            "feedback_data",
            "worklog.txt",
            ".vscode",
            ".claude",
            "*.bat",
            "modal_app.py",
            "Dockerfile",
            ".dockerignore",
        ],
    )
)

# Modal's CLI looks for this object by the name "app". It is unrelated to the
# Flask module of the same name imported inside web().
app = modal.App("cutout-studio", image=image)


@app.function(
    cpu=4,
    # With both models kept loaded a request settles at 11-20 GB. Modal bills
    # the higher of the request and actual use and does not cap memory unless
    # asked, so ask for what is really needed and let the scheduler place it.
    memory=12288,
    secrets=[modal.Secret.from_name("cutout-studio")],
    # Long enough for a batch of images to share one start, short because an
    # idle container is the largest cost of a single visit.
    scaledown_window=60,
    timeout=900,
)
# The app serialises model inference itself, so extra concurrency only lets the
# interface and small requests stay responsive while an image is processing.
@modal.concurrent(max_inputs=2)
# The public URL is https://<workspace>--<label>.modal.run
@modal.wsgi_app(label="cut-out-studio")
def web():
    from app import app as flask_app

    return flask_app
