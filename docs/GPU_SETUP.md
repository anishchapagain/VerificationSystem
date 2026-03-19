# GPU Setup Guide — 8GB VRAM / 24GB RAM

> This guide walks you through installing CUDA PyTorch, generating weights,
> starting the API, and training — in the correct order.

---

## Your Hardware Profile

| Resource | You Have | Required | Headroom |
|----------|----------|----------|----------|
| GPU VRAM | 8 GB | ~650 MB (training) / ~130 MB (inference) | Huge |
| RAM | 24 GB | ~5 GB (full stack) | ~19 GB free |
| Training time | — | ~45–75 min (GPU) vs ~12 hr (CPU) | Fast |

---

## Step 1 — Verify CUDA is Available

```bash
nvidia-smi
```

Note the **CUDA Version** in the top-right corner (e.g. 12.1, 11.8).

---

## Step 2 — Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

---

## Step 3 — Install PyTorch with CUDA

**Do NOT install torch from requirements.txt.** Use the correct CUDA wheel:

```bash
# For CUDA 12.1 (most common on drivers released after Jan 2024)
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older drivers)
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

Expected output: `CUDA: True | NVIDIA GeForce RTX XXXX`

---

## Step 4 — Install FAISS GPU

```bash
pip install faiss-gpu==1.7.4
```

Verify:
```bash
python -c "import faiss; print('FAISS GPUs:', faiss.get_num_gpus())"
# Expected: FAISS GPUs: 1
```

---

## Step 5 — Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 6 — Configure Environment

```bash
cp .env.example .env
```

The defaults work for your setup. Key values:
```
DEVICE=cuda
MODEL_WEIGHTS_PATH=weights/siamese_best.pt
MATCH_THRESHOLD=0.85
```

---

## Step 7 — Start PostgreSQL

```bash
docker run -d \
  --name sig-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=signature_db \
  -p 5432:5432 \
  postgres:16-alpine
```

---

## Step 8 — Generate Model Weights (Immediate API Access)

This creates a valid `.pt` checkpoint from random initialisation so the
API can start immediately. Verification scores will be random until
you replace this with trained weights (Step 10).

```bash
python scripts/generate_weights.py
```

Output:
```
weights/siamese_best.pt created successfully!
FastAPI server can now load and run inference.
Verification SCORES are random until training completes.
```

This file is ~130 MB on disk.

---

## Step 9 — Start the API

```bash
uvicorn backend.main:app --reload --port 8000
```

Check the health endpoint:
```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "database": "connected",
  "model_loaded": true
}
```

The API is now fully operational. You can register signatures and call
/verify — the scores just won't be meaningful yet.

---

## Step 10 — Obtain Training Data (CEDAR Dataset)

```bash
python scripts/download_cedar.py   # Prints download instructions
```

CEDAR is free for non-commercial research:
- **URL:** http://www.cedar.buffalo.edu/NIJ/data/
- **Size:** ~7 MB
- **Contents:** 55 signers × (24 genuine + 24 forged)

After downloading `cedar.zip`:
```bash
python scripts/download_cedar.py --cedar_zip /path/to/cedar.zip
```

This organises the images into `data/processed/genuine/` and `data/processed/forged/`.

---

## Step 11 — Train the Model

```bash
python -m ml.train \
  --data_dir data/processed \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.0001 \
  --output weights/siamese_best.pt \
  --amp \
  --workers 6
```

### What happens during training

| Phase | What you see |
|-------|-------------|
| Startup | GPU name, VRAM, parameter count |
| Per-epoch | train_loss, val_loss, EER, accuracy, FAR, FRR, time |
| Best checkpoint | "Best checkpoint saved | EER=0.XXXX" |
| Early stopping | Stops automatically if EER stops improving |
| Completion | Final EER and path to saved weights |

### Expected output on your 8GB GPU

```
Epoch   1/100 | train_loss=0.4821 | val_loss=0.4612 | EER=0.3201 | acc=0.6812 | time=32.1s
Epoch   5/100 | train_loss=0.2341 | val_loss=0.2198 | EER=0.1854 | acc=0.8211 | time=31.4s
  Best checkpoint saved | EER=0.1854
...
Epoch  40/100 | train_loss=0.0821 | val_loss=0.0934 | EER=0.0512 | acc=0.9421 | time=30.8s
  Best checkpoint saved | EER=0.0512
...
Training complete | best EER=0.0421 | saved to weights/siamese_best.pt
```

**Total training time: ~45–75 minutes on your GPU.**

### VRAM usage during training

| Batch size | VRAM used |
|------------|-----------|
| 32 | ~350 MB |
| 64 (recommended) | ~650 MB |
| 128 | ~1.2 GB |
| 256 | ~2.4 GB |

All of these comfortably fit your 8GB card.

---

## Step 12 — Restart API with Trained Weights

Training saves to the same path (`weights/siamese_best.pt`) that the API
reads from. Just restart the server:

```bash
# Stop the server (Ctrl+C) and restart
uvicorn backend.main:app --reload --port 8000
```

The API now loads the trained checkpoint. Verification scores are real.

---

## Step 13 — Run the Streamlit Frontend

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501

---

## Verify GPU is Used for Inference

Add this check to confirm the API is using your GPU:

```bash
python -c "
import torch
from backend.models.siamese_net import ModelManager
mgr = ModelManager('weights/siamese_best.pt')
mgr.load()
print(mgr.device_info)
"
```

Expected:
```python
{
  'device': 'cuda',
  'amp': True,
  'gpu_name': 'NVIDIA GeForce RTX XXXX',
  'vram_total_gb': 8.0,
  'vram_used_gb': 0.132
}
```

---

## Performance on Your Hardware

| Operation | CPU time | Your GPU time |
|-----------|----------|---------------|
| Load model | 2–4 s | 1–2 s |
| Preprocess image | 5–20 ms | 5–20 ms (CPU) |
| Extract embedding | 80–150 ms | 3–8 ms |
| Match (10 refs) | <1 ms | <1 ms |
| **Full /verify** | **100–200 ms** | **15–40 ms** |

**Training:** 12 hours (CPU) → **~60 minutes (your GPU)**

---

## Troubleshooting

**`CUDA: False`**
- NVIDIA driver not installed. Run: `sudo apt install nvidia-driver-535`
- Wrong CUDA version. Check `nvidia-smi` and reinstall PyTorch with matching URL.

**`torch.cuda.OutOfMemoryError`**
- Reduce `--batch_size` to 32.
- Add `torch.cuda.empty_cache()` before training.

**`faiss-gpu not found`**
- Run: `pip install faiss-gpu==1.7.4`
- If pip fails: `conda install -c conda-forge faiss-gpu`

**API returns `503 model not loaded`**
- Run `python scripts/generate_weights.py` first.
- Check `weights/siamese_best.pt` exists.

**Training loss is NaN with AMP**
- The GradScaler handles this automatically. If it persists, add `--amp false` to disable AMP.
