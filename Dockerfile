FROM python:3.12-slim

# onnxruntime needs the OpenMP runtime, which the slim image leaves out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the models the Auto mode uses into the image, so a restart does not
# download them again. The other profiles are fetched on first use.
COPY --chown=user app.py db.py ./
RUN python -c "import os, pooch; from app import BEN2_MODEL_URL, BEN2_MODEL_HASH; pooch.retrieve(url=BEN2_MODEL_URL, known_hash=BEN2_MODEL_HASH, fname='BEN2_Base.onnx', path=os.path.expanduser('~/.u2net'))" \
    && python -c "from rembg import new_session; new_session('birefnet-general-lite')"

COPY --chown=user . .

EXPOSE 7860
# One worker keeps a single copy of the models in memory; threads let the
# interface stay responsive while an image is being processed.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "300"]
