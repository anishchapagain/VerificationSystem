# Signature Verifier

> AI-powered handwritten signature registration and verification system.
> Built with **FastAPI**, **PostgreSQL**, **PyTorch Siamese Networks**, and **Streamlit**.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Environment Variables](#environment-variables)
- [Docker Deployment](#docker-deployment)
- [Training the ML Model](#training-the-ml-model)
- [Running Tests](#running-tests)
- [Roadmap](#roadmap)

---

## Overview

Signature Verifier accepts a handwritten signature as an image or video, preprocesses it through a multi-step OpenCV pipeline, extracts a 512-dimensional embedding vector using a trained Siamese Neural Network, and compares it against stored reference signatures using cosine similarity. The result is a `MATCH` or `NO MATCH` verdict with a confidence score and a full audit log.

The system is designed for real-world use in **banking**, **legal**, **healthcare**, and **government** workflows where physical document signatures need automated verification.

---

## Features

| Feature | Details |
|---------|---------|
| **Image verification** | PNG, JPG, BMP, TIFF |
| **Video verification** | MP4, AVI, MOV — sharpest frame auto-selected |
| **Ensemble voting** | Multiple video frames voted on for robustness |
| **Confidence scores** | Per-reference cosine similarity breakdown |
| **Adjustable threshold** | Tune FAR/FRR trade-off via `.env` |
| **Audit log** | Every verification stored with timestamp + score |
| **User management** | Register, login (JWT), profile |
| **REST API** | FastAPI with OpenAPI docs at `/docs` |
| **Streamlit UI** | 3-page web interface (Register / Verify / History) |
| **PostgreSQL** | Async SQLAlchemy with Alembic migrations |
| **Docker** | Full stack docker-compose in one command |
| **Structured logs** | Loguru with rotating files + optional JSON sink |
| **OOP throughout** | Every component is a class with docstrings |
| **Test suite** | pytest with unit + integration tests |

---

## Tech Stack

| Layer | Technology | Version |
|-------|------------|---------|
| API Framework | FastAPI | 0.115 |
| ASGI Server | Uvicorn | 0.30 |
| Frontend | Streamlit | 1.37 |
| ML Framework | PyTorch | 2.4 |
| Computer Vision | OpenCV | 4.10 |
| Vector Search | FAISS | 1.8 |
| Database ORM | SQLAlchemy (async) | 2.0 |
| Database | PostgreSQL | 16 |
| DB Driver | asyncpg | 0.29 |
| Migrations | Alembic | 1.13 |
| Auth | python-jose (JWT) + passlib (bcrypt) | — |
| Validation | Pydantic v2 | 2.8 |
| Logging | Loguru | 0.7 |
| Testing | pytest + pytest-asyncio | — |
| Containerisation | Docker + docker-compose | — |

---

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ (or use Docker)
- Git

### 1. Clone

```bash
git clone https://github.com/your-org/signature-verifier.git
cd signature-verifier
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set DATABASE_URL and SECRET_KEY at minimum
```

### 4. Start PostgreSQL

```bash
# Using Docker (easiest):
docker run -d \
  --name sig-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=signature_db \
  -p 5432:5432 \
  postgres:16-alpine
```

### 5. Start the API

```bash
uvicorn backend.main:app --reload --port 8000
```

Tables are created automatically on first startup.
Visit **http://localhost:8000/docs** for the interactive API documentation.

### 6. Start the Streamlit frontend

```bash
streamlit run frontend/app.py
```

Visit **http://localhost:8501**

### 7. Train or obtain model weights

Until you train the model (see [Training the ML Model](#training-the-ml-model)), the
`/verify` endpoint will return `503 — model not loaded`. You can:

- Train using the CEDAR dataset (~2h on Google Colab free T4 GPU)
- Use a pre-trained checkpoint if provided by your team
- For development, the register endpoint works without the model loaded

---

## Project Structure

```
signature-verifier/
│
├── backend/                        # FastAPI application
│   ├── main.py                     # App factory, lifespan, middleware
│   ├── config.py                   # Pydantic Settings (all env vars)
│   ├── core/
│   │   ├── logger.py               # Loguru setup — 3 sinks
│   │   └── exceptions.py           # 13 typed domain exceptions
│   ├── db/
│   │   ├── database.py             # Async engine, session, get_db dependency
│   │   ├── models.py               # ORM: User, Signature, MatchLog
│   │   └── crud.py                 # UserCRUD, SignatureCRUD, MatchLogCRUD
│   ├── models/
│   │   └── siamese_net.py          # ConvBlock, SignatureEncoder, SiameseNetwork, ModelManager
│   ├── services/
│   │   ├── preprocessor.py         # 8-step OpenCV pipeline
│   │   ├── matcher.py              # Cosine similarity + ensemble voting
│   │   ├── video_handler.py        # Frame extraction + Laplacian sharpness
│   │   └── auth.py                 # bcrypt + JWT
│   ├── routers/
│   │   ├── signature.py            # /api/signatures/* endpoints
│   │   ├── users.py                # /api/users/* endpoints
│   │   └── health.py               # /health endpoint
│   ├── schemas/
│   │   └── signature.py            # Pydantic request/response models
│   └── vector_store/
│       └── faiss_index.py          # Thread-safe FAISS IndexFlatIP wrapper
│
├── frontend/                       # Streamlit application
│   ├── app.py                      # Home page + sidebar config
│   └── pages/
│       ├── 1_Register.py           # Register reference signature
│       ├── 2_Verify.py             # Verify query signature
│       └── 3_History.py            # Audit log viewer
│
├── ml/                             # Model training pipeline
│   ├── train.py                    # Training loop with early stopping
│   ├── dataset.py                  # SignaturePairDataset (genuine/forged pairs)
│   ├── losses.py                   # ContrastiveLoss
│   └── evaluate.py                 # EER, accuracy_at_threshold
│
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── test_preprocessor.py        # Preprocessor unit tests
│   ├── test_matcher.py             # Matcher unit tests
│   └── test_api.py                 # API integration tests
│
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
│
├── weights/                        # Trained .pt checkpoint (git-ignored)
├── storage/                        # Uploaded images (git-ignored)
│   ├── signatures/
│   └── embeddings/
├── logs/                           # Rotating log files (git-ignored)
├── .env.example
├── requirements.txt
├── README.md
└── TECHNICAL.md
```

---

## API Documentation

Interactive docs are available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Endpoint Summary

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — database + model status |
| `POST` | `/api/users/register` | Create a new user account |
| `POST` | `/api/users/login` | Authenticate and receive JWT |
| `GET` | `/api/users/me/{user_id}` | Get user profile |
| `POST` | `/api/signatures/register` | Upload and store a reference signature |
| `POST` | `/api/signatures/verify` | Verify a query signature (image or video) |
| `GET` | `/api/signatures/{user_id}` | List all reference signatures for a user |
| `DELETE` | `/api/signatures/{signature_id}` | Soft-delete a reference signature |
| `GET` | `/api/signatures/history/{user_id}` | Paginated verification audit log |

### Example: Register a Signature

```bash
curl -X POST http://localhost:8000/api/signatures/register \
  -F "file=@/path/to/signature.png" \
  -F "user_id=1" \
  -F "label=Primary"
```

### Example: Verify a Signature

```bash
curl -X POST http://localhost:8000/api/signatures/verify \
  -F "file=@/path/to/query.png" \
  -F "user_id=1"
```

Response:
```json
{
  "user_id": 1,
  "verdict": true,
  "verdict_label": "MATCH",
  "score": 0.923456,
  "confidence": "High",
  "threshold_used": 0.85,
  "best_match_signature_id": 3,
  "source": "image",
  "score_breakdown": [
    {"signature_id": 3, "score": 0.923456},
    {"signature_id": 1, "score": 0.712100}
  ],
  "match_log_id": 44,
  "processed_at": "2024-11-01T10:25:00Z"
}
```

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/signature_db` | ✅ | PostgreSQL async DSN |
| `SECRET_KEY` | `changeme` | ✅ | JWT signing secret (use a long random string in production) |
| `MODEL_WEIGHTS_PATH` | `weights/siamese_best.pt` | ✅ | Path to trained Siamese checkpoint |
| `MATCH_THRESHOLD` | `0.85` | — | Cosine similarity cutoff for MATCH |
| `EMBEDDING_DIM` | `512` | — | Must match training config |
| `LOG_LEVEL` | `INFO` | — | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `DEBUG` | `false` | — | FastAPI debug mode + SQL echo |
| `VIDEO_FRAME_STRIDE` | `5` | — | Process every Nth video frame |
| `VIDEO_TOP_FRAMES` | `3` | — | Use top K sharpest frames for ensemble |
| `TOKEN_EXPIRE_MINUTES` | `60` | — | JWT lifetime in minutes |

---

## Docker Deployment

```bash
cd docker
docker compose up --build
```

This starts three services:
- `postgres` — PostgreSQL 16 on port 5432
- `backend` — FastAPI on port 8000
- `frontend` — Streamlit on port 8501

Persistent volumes are created for:
- PostgreSQL data
- Uploaded signature images (`./storage`)
- Model weights (`./weights`)
- Log files (`./logs`)

---

## Training the ML Model

The Siamese Network must be trained before the `/verify` endpoint works.

### Step 1: Obtain a dataset

Recommended: **CEDAR Signature Dataset**
- 55 subjects × 24 genuine + 24 forged signatures
- Download from: http://www.cedar.buffalo.edu/NIJ/data/

Expected directory layout after preprocessing:
```
data/processed/
├── genuine/
│   ├── user_001_sig_01.png
│   └── ...
└── forged/
    ├── user_001_forg_01.png
    └── ...
```

### Step 2: Train

```bash
python -m ml.train \
  --data_dir data/processed \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.0001 \
  --output weights/siamese_best.pt
```

Alternatively, use Google Colab (free T4 GPU):
1. Upload the `ml/` directory and `backend/models/siamese_net.py`
2. Mount Google Drive for persistent weight storage
3. Run `!python ml/train.py --data_dir /content/data --epochs 100`
4. Download `siamese_best.pt` to `weights/`

### Expected Training Metrics

| Metric | Expected Value |
|--------|---------------|
| EER (Equal Error Rate) | 3–8% |
| Val accuracy at 0.85 threshold | 90–95% |
| Training time (GPU) | ~2h on T4 |
| Training time (CPU) | ~12h |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Individual test files
pytest tests/test_preprocessor.py -v
pytest tests/test_matcher.py -v
pytest tests/test_api.py -v --asyncio-mode=auto

# With coverage
pytest tests/ --cov=backend --cov-report=html
```

---

## Roadmap

- [ ] FAISS index sync with DB on startup
- [ ] Alembic migration scripts
- [ ] JWT middleware for protected routes
- [ ] Admin dashboard (user management)
- [ ] Online signature support (stylus stroke data)
- [ ] Forgery detection confidence score
- [ ] Webhook notifications on failed verifications
- [ ] S3/GCS storage backend for signature images
- [ ] Prometheus metrics endpoint

---

Note: This is a work in progress, and being done using Vibe Coding.
