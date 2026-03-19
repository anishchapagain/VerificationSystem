# Technical Reference — Signature Verifier

> Version: 1.0.0 | Python 3.11 | FastAPI 0.115 | PostgreSQL 16 | PyTorch 2.4

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [How the System Works — End to End](#2-how-the-system-works--end-to-end)
3. [ML Models & Algorithms Used](#3-ml-models--algorithms-used)
4. [Replaceable Components](#4-replaceable-components)
5. [Module Reference](#5-module-reference)
6. [Class & Method API](#6-class--method-api)
7. [Database Schema](#7-database-schema)
8. [Preprocessing Pipeline Deep Dive](#8-preprocessing-pipeline-deep-dive)
9. [Matching Algorithm](#9-matching-algorithm)
10. [Video Processing](#10-video-processing)
11. [API Contracts](#11-api-contracts)
12. [Configuration Reference](#12-configuration-reference)
13. [Error Handling Architecture](#13-error-handling-architecture)
14. [Logging Architecture](#14-logging-architecture)
15. [Security Design](#15-security-design)
16. [Performance Characteristics](#16-performance-characteristics)
17. [Known Limitations](#17-known-limitations)

---

## 1. System Architecture

```
╔════════════════════════════════════════════════════╗
║           STREAMLIT FRONTEND  (port 8501)          ║
║  app.py → pages/1_Register  2_Verify  3_History    ║
╚══════════════════╦═════════════════════════════════╝
                   ║  HTTP multipart/form-data
                   ║  JSON responses
╔══════════════════╩═════════════════════════════════╗
║           FASTAPI BACKEND  (port 8000)             ║
║                                                    ║
║  main.py                                           ║
║    ├── CORSMiddleware                              ║
║    ├── Request logging middleware                  ║
║    ├── Domain exception handler                    ║
║    └── Generic exception handler                  ║
║                                                    ║
║  routers/                                          ║
║    ├── health.py      GET /health                  ║
║    ├── users.py       POST /api/users/*            ║
║    └── signature.py   POST /api/signatures/*       ║
║                                                    ║
║  services/                    models/              ║
║    ├── preprocessor.py          └── siamese_net.py ║
║    ├── matcher.py             vector_store/        ║
║    ├── video_handler.py         └── faiss_index.py ║
║    └── auth.py                                     ║
╚════════╦═════════════════════════╦════════════════╝
         ║                         ║
╔════════╩════════╗     ╔══════════╩══════════════╗
║   PostgreSQL 16  ║     ║   File Storage           ║
║   (asyncpg)      ║     ║   storage/signatures/    ║
║                  ║     ║   storage/embeddings/    ║
║  users           ║     ║   weights/               ║
║  signatures      ║     ╚═════════════════════════╝
║  match_logs      ║
╚══════════════════╝
```

### Design Principles

- **OOP throughout** — every component is a class with typed attributes, docstrings, and private helpers.
- **Domain exceptions** — no raw SQLAlchemy or OpenCV exceptions leak to the API layer.
- **Dependency injection** — FastAPI `Depends()` wires services into routes; nothing is a module-level singleton except the FAISS index.
- **Async I/O** — all database operations are async (asyncpg + SQLAlchemy 2.0 async). File I/O is synchronous (acceptable for current load; can be moved to a thread pool with `run_in_executor` if needed).
- **Separation of concerns** — routers validate input and build responses; services implement logic; CRUD handles persistence; models define structure.

---

## 2. How the System Works — End to End

### Registration Flow

```
User uploads image via Streamlit
  │
  ▼
POST /api/signatures/register (multipart form)
  │
  ├─ [1] Validate file extension and size
  ├─ [2] Save raw image to storage/signatures/{uuid}.ext
  │
  ├─ [3] SignaturePreprocessor.run(file_path)
  │         → Grayscale → Denoise → Binarize (Otsu)
  │         → Morphology → Crop → Resize(256×128) → Normalize
  │         Returns: float32 array (128, 256) in [0,1]
  │
  ├─ [4] ModelManager.extract_embedding(array)
  │         → Add batch+channel dims: (1,1,128,256)
  │         → SignatureEncoder.forward() → FC → L2 Normalize
  │         Returns: float32 array (512,) on unit sphere
  │
  ├─ [5] SignatureCRUD.create(db, user_id, file_path, embedding)
  │         → embedding.tobytes() stored in BYTEA column
  │         → Returns Signature ORM record
  │
  └─ [6] Return SignatureRegisterResponse (201)
```

### Verification Flow

```
User uploads image or video via Streamlit
  │
  ▼
POST /api/signatures/verify (multipart form)
  │
  ├─ [1] Detect modality: image vs video (by file extension)
  ├─ [2] Save query file to storage/signatures/queries/{uuid}.ext
  │
  ├─ [3a] If IMAGE:
  │         SignaturePreprocessor.run() → embedding (1 vector)
  │
  ├─ [3b] If VIDEO:
  │         VideoSignatureExtractor.extract()
  │           → cv2.VideoCapture reads every 5th frame
  │           → Laplacian variance scores each frame
  │           → Top 3 sharpest frames selected
  │           → Each frame preprocessed → 3 embeddings
  │
  ├─ [4] SignatureCRUD.get_embeddings_by_user(db, user_id)
  │         → Loads all Signature records for user
  │         → Deserialises embedding bytes back to float32 arrays
  │
  ├─ [5a] If IMAGE (1 embedding):
  │         SignatureMatcher.match(query, references, user_id)
  │           → sklearn cosine_similarity(query, all_refs)
  │           → max_score = highest cosine similarity
  │           → verdict = (max_score >= threshold)
  │
  ├─ [5b] If VIDEO (N embeddings):
  │         SignatureMatcher.ensemble_match(embeddings, refs, user_id)
  │           → Runs match() for each frame's embedding
  │           → Majority vote across frame verdicts
  │           → Best-scoring frame's score used
  │
  ├─ [6] MatchLogCRUD.create(db, ...)
  │         → Writes immutable audit record to match_logs table
  │
  └─ [7] Return VerifyResponse with score, verdict, breakdown (200)
```

---

## 3. ML Models & Algorithms Used

### 3.1 Siamese Neural Network (Primary Model)

**What it is:** A twin convolutional network trained to produce embedding vectors for signature images such that genuine signatures cluster together on the unit sphere, while forgeries are pushed far apart.

**Architecture — `SignatureEncoder`:**

```
Input: (B, 1, 128, 256)  — batch × channel × H × W

ConvBlock 1:  Conv2d(1→32,   k=3, pad=1) → BN → ReLU → MaxPool(2)  → (B, 32,  64, 128)
ConvBlock 2:  Conv2d(32→64,  k=3, pad=1) → BN → ReLU → MaxPool(2)  → (B, 64,  32,  64)
ConvBlock 3:  Conv2d(64→128, k=3, pad=1) → BN → ReLU → MaxPool(2)  → (B, 128, 16,  32)
ConvBlock 4:  Conv2d(128→256,k=3, pad=1) → BN → ReLU → MaxPool(2)  → (B, 256,  8,  16)

Flatten:      (B, 256×8×16) = (B, 32768)
FC1:          Linear(32768 → 1024) → ReLU → Dropout(0.5)
FC2:          Linear(1024 → 512)
L2 Norm:      F.normalize(dim=1)                                    → (B, 512)
```

Total trainable parameters: ~34 million.

**Why L2 normalisation?**
Placing all embeddings on the unit sphere means cosine similarity = dot product, which is faster to compute and bounded in [-1, 1]. FAISS IndexFlatIP exploits this directly.

**Training objective — Contrastive Loss:**

```
d  = ||emb_1 - emb_2||_2       (Euclidean distance between pair)
label = 1  →  genuine pair (same person)
label = 0  →  impostor pair (forgery or different person)

L = label × d²  +  (1 - label) × max(margin - d, 0)²
```

This pushes genuine pairs to distance 0 and impostor pairs beyond `margin` (default 1.0).

**Training datasets:**
| Dataset | Subjects | Genuine/subject | Forged/subject |
|---------|----------|-----------------|----------------|
| CEDAR | 55 | 24 | 24 |
| SigNet | 150 | 30 | 20 |
| GPDS-960 | 960 | 24 | 30 |

**Benchmark performance on CEDAR:**
| Metric | This Architecture | SOTA |
|--------|------------------|------|
| EER | ~4–6% | ~2% |
| Accuracy @0.85 | ~92% | ~97% |

---

### 3.2 Cosine Similarity (Matching Algorithm)

**What it is:** A vector similarity metric that measures the angle between two L2-normalised embedding vectors.

```
cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)
                        = a · b   (when both are L2-normalised)
```

Range: [-1, 1] where 1 = identical direction, 0 = orthogonal, -1 = opposite.

**Implementation:** `sklearn.metrics.pairwise.cosine_similarity` — handles edge cases and is vectorised for batch computation.

**Why not Euclidean distance?**
Cosine similarity is invariant to vector magnitude, making it more robust to small variations in pen pressure and scan exposure. It also maps naturally to a threshold in [0, 1].

---

### 3.3 FAISS IndexFlatIP (Vector Search)

**What it is:** Facebook AI Similarity Search — a library for fast nearest-neighbour lookup over dense vector sets.

**Index type used:** `IndexFlatIP` (Inner Product / cosine for L2-normalised vectors).

This is an exact search index — no approximation. For up to ~100k signatures, exact search is fast enough (<10ms). For millions of signatures, swap to `IndexIVFFlat` or `IndexHNSW`.

**Why FAISS over pure numpy?**
For large user bases, FAISS leverages SIMD CPU instructions (AVX2) for ~50× faster batch similarity search vs. numpy loops.

---

### 3.4 Otsu's Thresholding (Preprocessing)

**What it is:** An automatic global threshold selection algorithm that minimises intra-class variance between the foreground (ink) and background (paper) pixel intensity distributions.

```python
_, binary = cv2.threshold(
    blurred, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
```

Otsu's method is used because it requires zero configuration — the optimal threshold is computed per-image. This handles variability in scanner exposure, paper colour, and pen ink darkness.

---

### 3.5 Laplacian Variance (Video Sharpness)

**What it is:** The variance of the Laplacian-filtered image. High variance indicates strong edges = sharp image.

```python
sharpness = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
```

Used to automatically select the best frame from a video. Motion-blurred frames have very low Laplacian variance and are discarded.

---

### 3.6 No LLMs/SLMs Used

This system deliberately does **not** use large language models. All intelligence comes from:

1. A domain-specific **Siamese CNN** trained on signature data.
2. Classic **computer vision** (OpenCV, Laplacian, Otsu).
3. **Geometric similarity** (cosine similarity, FAISS).

This makes the system:
- **Fast**: ~50–200ms per verification (no LLM inference latency).
- **Private**: runs fully on-premise, no data leaves the server.
- **Predictable**: deterministic outputs for the same input.
- **Lightweight**: the full model is ~130MB on disk.

---

## 4. Replaceable Components

The OOP design and dependency injection make every component swappable without touching other layers.

### Replace the ML Model

The `ModelManager` class in `backend/models/siamese_net.py` is the only coupling point between the ML model and the rest of the system. To replace the Siamese Network:

1. Create a new model class with a `extract_embedding(array: np.ndarray) -> np.ndarray` method.
2. Update `ModelManager.load()` and `ModelManager.extract_embedding()`.
3. No other files need changes.

**Possible replacements:**

| Alternative | When to use | Notes |
|-------------|-------------|-------|
| **EfficientNet-B0** (fine-tuned) | When you have >1000 training samples per user | Higher accuracy, 5× larger model |
| **ResNet-18** (fine-tuned) | General purpose upgrade | Well-supported, easy to fine-tune |
| **ViT-Small** (Vision Transformer) | State-of-the-art accuracy needed | Requires more training data |
| **CLIP image encoder** (OpenAI) | Zero-shot, no retraining needed | Lower accuracy without fine-tuning |
| **SigNet** (pretrained) | Drop-in offline signature specialist | Best option for production without training |
| **Writer-independent DTW** | No ML, classical approach | Lower accuracy, no training needed |

To use a pre-trained HuggingFace model instead:

```python
# In backend/models/siamese_net.py — replace ModelManager.extract_embedding:
from transformers import AutoFeatureExtractor, AutoModel
import torch

class HuggingFaceModelManager:
    def __init__(self, model_name: str = "microsoft/resnet-18"):
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def extract_embedding(self, image_array: np.ndarray) -> np.ndarray:
        # Convert to PIL, extract features, return pooled output
        from PIL import Image
        pil = Image.fromarray((image_array * 255).astype(np.uint8))
        inputs = self.extractor(images=pil, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.pooler_output.squeeze().numpy()
        return embedding / np.linalg.norm(embedding)
```

---

### Replace the Vector Store

`backend/vector_store/faiss_index.py` wraps FAISS with a stable interface. To replace:

| Alternative | When to use |
|-------------|-------------|
| **ChromaDB** | When you need metadata filtering + persistence out of the box |
| **Qdrant** | Production-grade vector DB with cloud option |
| **Pinecone** | Fully managed, no infra |
| **pgvector** | When you want vectors inside PostgreSQL (simplest stack) |
| **Pure numpy** | Development only, < 1000 signatures |

The `SignatureMatcher` class does not call FAISS directly — it receives pre-loaded embeddings from `SignatureCRUD.get_embeddings_by_user()`. So the vector store is only used for fast lookup at scale; the core matching logic is independent.

---

### Replace the Preprocessing Pipeline

`SignaturePreprocessor` can be extended or replaced step by step:

| Step | Alternative |
|------|-------------|
| Otsu threshold | Adaptive threshold (`cv2.THRESH_ADAPTIVE_MEAN`) for uneven lighting |
| Gaussian blur | Bilateral filter — preserves edges while denoising |
| Morphological cleanup | Thinning algorithms (Zhang-Suen) for stroke-level analysis |
| Resize | Aspect-ratio preserving resize with zero-padding |
| Entire pipeline | Deep learning-based preprocessing (U-Net segmentation) |

---

### Replace the Database

SQLAlchemy's ORM abstracts the database entirely. To switch from PostgreSQL:

- **SQLite** (development): change `DATABASE_URL` to `sqlite+aiosqlite:///./db.sqlite3` and add `aiosqlite` to requirements.
- **MySQL**: change to `mysql+aiomysql://...`.
- **MongoDB**: replace `db/models.py` with Motor ODM models and `db/crud.py` with Motor queries.

---

### Replace the Frontend

The Streamlit frontend is stateless — it only makes HTTP calls to the FastAPI backend. It can be replaced with any frontend that speaks HTTP:

- React / Vue / Angular SPA
- Mobile app (React Native, Flutter)
- CLI tool (`httpx` or `requests`)
- Another Streamlit app in a different style

---

## 5. Module Reference

| Module | Class | Responsibility |
|--------|-------|---------------|
| `backend/main.py` | `create_app()` | App factory, middleware, exception handlers, lifespan |
| `backend/config.py` | `Settings` | Pydantic Settings — single source of config truth |
| `backend/core/logger.py` | — | `setup_logger()`, `get_logger()` |
| `backend/core/exceptions.py` | 13 exception classes | Typed domain errors with HTTP status codes |
| `backend/db/database.py` | — | Engine, session factory, `get_db`, `init_db`, `check_db_health` |
| `backend/db/models.py` | `User`, `Signature`, `MatchLog` | SQLAlchemy ORM definitions |
| `backend/db/crud.py` | `UserCRUD`, `SignatureCRUD`, `MatchLogCRUD` | All database read/write logic |
| `backend/models/siamese_net.py` | `ConvBlock`, `SignatureEncoder`, `SiameseNetwork`, `ModelManager` | CNN architecture + inference |
| `backend/services/preprocessor.py` | `SignaturePreprocessor`, `PreprocessingResult` | 8-step OpenCV pipeline |
| `backend/services/matcher.py` | `SignatureMatcher`, `MatchResult` | Cosine similarity + ensemble voting |
| `backend/services/video_handler.py` | `VideoSignatureExtractor`, `FrameCandidate` | Frame sampling + sharpness scoring |
| `backend/services/auth.py` | `AuthService` | bcrypt hashing + JWT sign/verify |
| `backend/vector_store/faiss_index.py` | `FAISSIndexManager` | Thread-safe FAISS IndexFlatIP |
| `backend/routers/signature.py` | Router | Register, verify, list, delete, history endpoints |
| `backend/routers/users.py` | Router | Register user, login, profile |
| `backend/routers/health.py` | Router | `/health` liveness probe |
| `backend/schemas/signature.py` | 10 Pydantic models | Request/response contracts |
| `ml/train.py` | — | Training loop, early stopping, checkpoint saving |
| `ml/dataset.py` | `SignaturePairDataset` | Genuine/forged pair generation |
| `ml/losses.py` | `ContrastiveLoss` | Siamese training loss |
| `ml/evaluate.py` | — | `compute_eer()`, `accuracy_at_threshold()` |

---

## 6. Class & Method API

### `SignaturePreprocessor`

```python
SignaturePreprocessor(
    target_width: int = 256,
    target_height: int = 128,
    blur_kernel: Tuple[int, int] = (5, 5),
    morph_kernel_size: int = 3,
    debug: bool = False,
)

.run(image_path: str) -> PreprocessingResult
    # Full pipeline from file path. Validates extension first.

.run_from_array(array: np.ndarray) -> PreprocessingResult
    # Full pipeline from in-memory BGR or grayscale array.
    # Used internally by VideoSignatureExtractor.

# Each step callable independently for unit testing:
._to_grayscale(image) -> np.ndarray     # BGR→Gray
._denoise(gray) -> np.ndarray           # Gaussian blur
._binarize(blurred) -> np.ndarray       # Otsu → binary
._morphological_cleanup(binary)         # Close + Open
._crop_to_signature(binary)             # Contour bounding box
._resize(image)                         # Target dimensions
._normalize(image)                      # uint8→float32 [0,1]
```

---

### `ModelManager`

```python
ModelManager(
    weights_path: str,        # Path to .pt checkpoint
    embedding_dim: int = 512, # Must match training config
    device: Optional[str] = None,  # "cuda" | "cpu" | auto
)

.load() -> None
    # Load checkpoint weights. Raises ModelNotLoadedError if file missing.

.extract_embedding(image_array: np.ndarray) -> np.ndarray
    # image_array: float32 (H, W) in [0,1]
    # Returns: float32 (512,) L2-normalised embedding

.is_loaded -> bool  # property
```

---

### `SignatureMatcher`

```python
SignatureMatcher(threshold: float = 0.85)

.match(
    query_embedding: np.ndarray,                          # (512,)
    reference_embeddings: List[Tuple[int, np.ndarray]],  # [(sig_id, emb), ...]
    user_id: int,
) -> MatchResult

.ensemble_match(
    query_embeddings: List[np.ndarray],                  # One per video frame
    reference_embeddings: List[Tuple[int, np.ndarray]],
    user_id: int,
) -> MatchResult
    # Majority vote: verdict = more-than-half frames vote MATCH
```

**MatchResult:**
```python
@dataclass
class MatchResult:
    score: float           # Best cosine similarity [0.0–1.0]
    verdict: bool          # True = MATCH
    best_sig_id: int       # DB pk of closest reference
    threshold_used: float
    all_scores: dict       # {sig_id: score} for all references
    confidence: str        # property → "Very High" | "High" | "Medium" | "Low" | "Very Low"
```

---

### `VideoSignatureExtractor`

```python
VideoSignatureExtractor(
    stride: int = 5,          # Sample every Nth frame
    top_frames: int = 3,      # Use K sharpest
    preprocessor: SignaturePreprocessor = None,  # Injected or auto-created
)

.extract(video_path: str) -> List[PreprocessingResult]
    # Returns one PreprocessingResult per selected frame.
    # Raises NoUsableFrameError if all frames below sharpness threshold.

MIN_SHARPNESS_THRESHOLD = 50.0  # Laplacian variance cutoff
```

---

### CRUD Classes

All methods are `@staticmethod async`:

```python
# UserCRUD
UserCRUD.create(db, name, email, hashed_password) -> User
UserCRUD.get_by_id(db, user_id) -> Optional[User]
UserCRUD.get_by_email(db, email) -> Optional[User]
UserCRUD.deactivate(db, user_id) -> None

# SignatureCRUD
SignatureCRUD.create(db, user_id, file_path, embedding, faiss_id, label) -> Signature
SignatureCRUD.get_by_user(db, user_id) -> List[Signature]
SignatureCRUD.get_by_id(db, sig_id) -> Optional[Signature]
SignatureCRUD.get_embeddings_by_user(db, user_id) -> List[Tuple[int, np.ndarray]]
SignatureCRUD.soft_delete(db, sig_id) -> None
SignatureCRUD.update_faiss_id(db, sig_id, faiss_id) -> None

# MatchLogCRUD
MatchLogCRUD.create(db, user_id, query_path, score, threshold_used, verdict, source, best_match_id) -> MatchLog
MatchLogCRUD.get_by_user(db, user_id, limit, offset) -> List[MatchLog]
```

---

## 7. Database Schema

```sql
-- ─── users ────────────────────────────────────────────────────────────────
CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(120) NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active       BOOLEAN DEFAULT TRUE NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at      TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX idx_users_email ON users(email);

-- ─── signatures ───────────────────────────────────────────────────────────
CREATE TABLE signatures (
    id          SERIAL PRIMARY KEY,
    user_id     INT REFERENCES users(id) ON DELETE CASCADE NOT NULL,
    label       VARCHAR(100),
    file_path   TEXT NOT NULL,
    embedding   BYTEA NOT NULL,          -- np.float32.tobytes() — 512×4=2048 bytes
    faiss_id    INT,                     -- Row index in FAISS flat index
    is_active   BOOLEAN DEFAULT TRUE NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX idx_signatures_user_id  ON signatures(user_id);
CREATE INDEX idx_signatures_faiss_id ON signatures(faiss_id);

-- ─── match_logs ───────────────────────────────────────────────────────────
CREATE TABLE match_logs (
    id              SERIAL PRIMARY KEY,
    user_id         INT REFERENCES users(id)      ON DELETE CASCADE  NOT NULL,
    best_match_id   INT REFERENCES signatures(id) ON DELETE SET NULL,
    query_path      TEXT NOT NULL,
    score           FLOAT NOT NULL,
    threshold_used  FLOAT NOT NULL,
    verdict         BOOLEAN NOT NULL,
    source          VARCHAR(20) DEFAULT 'image' NOT NULL,  -- 'image' | 'video'
    created_at      TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX idx_match_logs_user_id    ON match_logs(user_id);
CREATE INDEX idx_match_logs_created_at ON match_logs(created_at);
```

**Embedding serialisation:**
```python
# Write to DB:
embedding.astype(np.float32).tobytes()         # 512 × 4 bytes = 2 KB

# Read from DB:
np.frombuffer(row.embedding, dtype=np.float32) # → shape (512,)
```

---

## 8. Preprocessing Pipeline Deep Dive

Each step is a discrete private method on `SignaturePreprocessor`, callable independently for unit testing.

| Step | Function | Input | Output | OpenCV Call |
|------|----------|-------|--------|-------------|
| 1. Load | `_load` | file path | BGR (H,W,3) uint8 | `cv2.imread` |
| 2. Grayscale | `_to_grayscale` | BGR (H,W,3) | Gray (H,W) uint8 | `cv2.cvtColor(BGR2GRAY)` |
| 3. Denoise | `_denoise` | Gray (H,W) | Gray (H,W) | `cv2.GaussianBlur(5,5)` |
| 4. Binarize | `_binarize` | Gray (H,W) | Binary (H,W) 0/255 | `cv2.threshold(THRESH_OTSU\|INV)` |
| 5. Morphology | `_morphological_cleanup` | Binary | Cleaned binary | `morphologyEx(CLOSE)` then `(OPEN)` |
| 6. Crop | `_crop_to_signature` | Binary | Cropped (variable) | `findContours` → `boundingRect` |
| 7. Resize | `_resize` | Cropped | (128, 256) uint8 | `cv2.resize(INTER_AREA)` |
| 8. Normalize | `_normalize` | uint8 [0,255] | float32 [0,1] | `array / 255.0` |

**Why these specific choices:**
- **Gaussian blur (5×5)** — Removes high-frequency noise without blurring signature strokes excessively. Larger kernels (9×9) would lose thin strokes.
- **Otsu + THRESH_INV** — Produces white-on-black (ink=255, paper=0). The model was trained on this convention.
- **MORPH_CLOSE then MORPH_OPEN** — Close fills small gaps within strokes; Open then removes isolated noise pixels that survived binarisation.
- **INTER_AREA resize** — Best interpolation for downscaling. Avoids aliasing artefacts that `INTER_LINEAR` introduces.

---

## 9. Matching Algorithm

### Cosine Similarity

For L2-normalised embeddings, cosine similarity equals the dot product:

```
cos(emb_q, emb_r) = emb_q · emb_r   (since ||emb_q|| = ||emb_r|| = 1)
```

The matcher computes this for all reference embeddings simultaneously using a matrix multiplication:

```python
# query_2d: (1, 512)
# ref_matrix: (N, 512)
scores = sklearn.metrics.pairwise.cosine_similarity(query_2d, ref_matrix)
# scores: (1, N) → take [0] → (N,)
```

### Threshold Selection

The default threshold `0.85` was chosen empirically on the CEDAR dataset at the operating point where `FAR ≈ FRR ≈ 5%`.

To tune the threshold for your deployment:
```python
from ml.evaluate import accuracy_at_threshold
metrics = accuracy_at_threshold(labels, scores, threshold=0.85)
# Returns: {accuracy, precision, recall, f1, far, frr}
```

### Multi-Reference Strategy

When a user has N reference signatures, the system uses a **1-vs-all-max** strategy:
- Compare the query against every reference.
- The final score is the maximum cosine similarity found.
- This improves recall (catches more genuine signatures) at the cost of slightly lower precision.

---

## 10. Video Processing

```
VideoSignatureExtractor.extract(video_path)
  │
  ├─ cv2.VideoCapture opens the file
  │
  ├─ For every Nth frame (stride=5):
  │   ├─ cv2.Laplacian(gray, CV_64F).var() → sharpness score
  │   ├─ If sharpness >= 50.0 → keep as FrameCandidate
  │   └─ If sharpness < 50.0 → discard (motion blur)
  │
  ├─ Sort candidates by sharpness descending
  ├─ Select top K=3 frames
  │
  ├─ For each selected frame:
  │   └─ SignaturePreprocessor.run_from_array(frame) → PreprocessingResult
  │
  └─ Return [PreprocessingResult × K]
```

The caller (`verify_signature` route) then:
1. Extracts one embedding per result.
2. Calls `SignatureMatcher.ensemble_match()`.
3. Majority vote determines final verdict.

---

## 11. API Contracts

### POST `/api/signatures/register`

**Input** (multipart/form-data):
```
file     : UploadFile  (PNG/JPG/BMP/TIFF, max 10 MB)
user_id  : int (>0)
label    : str (optional, max 100 chars)
```

**Output** (201):
```json
{
  "signature_id": 12,
  "user_id": 3,
  "label": "Primary",
  "file_path": "storage/signatures/abc123de.png",
  "faiss_id": null,
  "created_at": "2024-11-01T10:22:00Z",
  "message": "Signature registered successfully."
}
```

---

### POST `/api/signatures/verify`

**Input** (multipart/form-data):
```
file     : UploadFile  (image or video)
user_id  : int (>0)
```

**Output** (200):
```json
{
  "user_id": 3,
  "verdict": true,
  "verdict_label": "MATCH",
  "score": 0.923456,
  "confidence": "High",
  "threshold_used": 0.85,
  "best_match_signature_id": 12,
  "source": "image",
  "score_breakdown": [
    {"signature_id": 12, "score": 0.923456},
    {"signature_id": 7,  "score": 0.712100}
  ],
  "match_log_id": 44,
  "processed_at": "2024-11-01T10:25:00Z"
}
```

**Confidence labels:**
| Score | Label |
|-------|-------|
| ≥ 0.95 | Very High |
| ≥ 0.90 | High |
| ≥ 0.85 | Medium |
| ≥ 0.70 | Low |
| < 0.70 | Very Low |

---

### GET `/health`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected",
  "model_loaded": true,
  "uptime_seconds": 3600.4,
  "timestamp": "2024-11-01T11:22:00Z"
}
```

`status` is `"degraded"` if database is unreachable or model weights are missing.

---

## 12. Configuration Reference

All configuration is in `backend/config.py` (Pydantic `BaseSettings`). Values are loaded from environment variables or `.env` file with the following precedence: environment variable > `.env` file > default.

| Variable | Type | Default | Notes |
|----------|------|---------|-------|
| `APP_NAME` | str | `"Signature Verifier API"` | Display name |
| `APP_VERSION` | str | `"1.0.0"` | Semantic version |
| `DEBUG` | bool | `false` | Enables SQL echo and verbose errors |
| `LOG_LEVEL` | str | `INFO` | Loguru level |
| `DATABASE_URL` | str | postgres+asyncpg://... | Full DSN with asyncpg driver |
| `MODEL_WEIGHTS_PATH` | str | `weights/siamese_best.pt` | Path to .pt checkpoint |
| `EMBEDDING_DIM` | int | `512` | Must match training |
| `MATCH_THRESHOLD` | float | `0.85` | Tune via ROC analysis |
| `SIGNATURE_STORAGE_PATH` | str | `storage/signatures` | Auto-created |
| `FAISS_INDEX_PATH` | str | `storage/embeddings/index.faiss` | Auto-created |
| `ALLOWED_ORIGINS` | list | `["http://localhost:8501"]` | CORS origins |
| `SECRET_KEY` | str | `changeme` | **Must change in production** |
| `ALGORITHM` | str | `HS256` | JWT signing algorithm |
| `TOKEN_EXPIRE_MINUTES` | int | `60` | JWT lifetime |
| `VIDEO_FRAME_STRIDE` | int | `5` | Frame sampling rate |
| `VIDEO_TOP_FRAMES` | int | `3` | Ensemble frame count |
| `IMAGE_TARGET_WIDTH` | int | `256` | Preprocessor output |
| `IMAGE_TARGET_HEIGHT` | int | `128` | Preprocessor output |
| `MAX_FILE_SIZE_MB` | int | `10` | Upload size limit |

---

## 13. Error Handling Architecture

### Exception Hierarchy

```
SignatureVerifierError (base)
├── status_code: int      (HTTP hint)
├── message: str          (user-facing)
└── detail: Optional[str] (for logging)

Subclasses:
  ImageLoadError           → 422  (file unreadable)
  InvalidImageFormatError  → 415  (wrong extension)
  ImagePreprocessingError  → 422  (pipeline step failed)
  VideoLoadError           → 422  (video unreadable)
  NoUsableFrameError       → 422  (all frames blurry)
  ModelNotLoadedError      → 503  (weights not loaded)
  EmbeddingExtractionError → 500  (inference failed)
  VectorStoreError         → 500  (FAISS operation failed)
  NoReferenceSignatureError→ 404  (user has no references)
  DatabaseError            → 500  (SQLAlchemy error)
  RecordNotFoundError      → 404  (entity not in DB)
  UserAlreadyExistsError   → 409  (duplicate email)
  AuthenticationError      → 401  (invalid JWT)
```

### Three-Level Handling

**Level 1 — Service layer:** catches raw library errors and wraps them:
```python
try:
    result = cv2.threshold(...)
except Exception as exc:
    raise ImagePreprocessingError("binarize", detail=str(exc)) from exc
```

**Level 2 — Router layer:** converts domain exceptions to HTTPException:
```python
except NoReferenceSignatureError as exc:
    raise HTTPException(status_code=404, detail=exc.message)
except Exception as exc:
    log.exception(...)
    raise HTTPException(status_code=500, detail="Internal server error.")
```

**Level 3 — App-level handlers:** safety net in `main.py` for anything that escapes the routers:
```python
@app.exception_handler(SignatureVerifierError)
async def domain_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})
```

---

## 14. Logging Architecture

Three Loguru sinks configured in `backend/core/logger.py`:

| Sink | Destination | Level | Format | Rotation |
|------|-------------|-------|--------|----------|
| Console | stderr | configurable | Colored human-readable | — |
| App log | `logs/app_YYYY-MM-DD.log` | configurable | Plain text | 10 MB / 7 days |
| Error log | `logs/errors_YYYY-MM-DD.log` | ERROR+ | Plain text | 5 MB / 30 days |

**Named loggers:**
```python
log = get_logger("preprocessor")   # bound context
log.info("Pipeline complete | file={name} | shape={shape}")
```

**Log levels in use:**
- `DEBUG` — frame sharpness scores, DB session lifecycle, embedding norms
- `INFO` — every successful registration/verification, model load, app start/stop
- `WARNING` — recoverable issues (blurry frame skipped, empty FAISS index)
- `ERROR` — operation failed (preprocessing error, DB rollback)
- `CRITICAL` — startup failure (DB init, model critical error)

---

## 15. Security Design

| Concern | Implementation |
|---------|---------------|
| Password storage | bcrypt (work factor 12) via passlib |
| Authentication | JWT HS256, configurable expiry |
| CORS | Explicit origin whitelist via CORSMiddleware |
| File upload validation | Extension whitelist + size limit |
| SQL injection | SQLAlchemy ORM with parameterised queries — no raw SQL |
| Sensitive data in logs | `diagnose=False` on file sinks — local variables not logged |
| Secret management | All secrets in `.env`, never committed |

**Production hardening checklist:**
- [ ] Set `SECRET_KEY` to a 64+ character random string
- [ ] Set `DEBUG=false`
- [ ] Restrict `ALLOWED_ORIGINS` to your domain
- [ ] Use HTTPS (terminate TLS at nginx/Caddy reverse proxy)
- [ ] Enable PostgreSQL SSL (`?ssl=require` in DSN)
- [ ] Rotate JWT secret periodically

---

## 16. Performance Characteristics

| Operation | Typical Latency | Bottleneck |
|-----------|----------------|-----------|
| Preprocessing (image) | 5–20 ms | OpenCV I/O |
| Embedding extraction (CPU) | 30–120 ms | FC layers |
| Embedding extraction (GPU) | 2–8 ms | Data transfer |
| Cosine similarity (100 refs) | <1 ms | Vectorised numpy |
| FAISS search (10k refs) | <5 ms | SIMD |
| PostgreSQL query | 1–5 ms | Network |
| Full verification (CPU) | 50–200 ms | Model inference |
| Full verification (GPU) | 15–50 ms | Model inference |

**Scaling notes:**
- FAISS `IndexFlatIP` is exact search — fine for <500k vectors. Above that, switch to `IndexIVFFlat`.
- The model is stateless — multiple Uvicorn workers can share the same `.pt` file safely.
- Database connection pool (`pool_size=10, max_overflow=20`) handles ~30 concurrent requests.

---

## 17. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| Model not pre-trained | Cannot verify until training completes | Use Google Colab for free GPU training |
| FAISS index not synced to DB on restart | Cold-start FAISS index is empty | Call `FAISSIndexManager.rebuild()` with all DB embeddings on startup |
| Skilled forgery detection | Trained forgers can fool the system | Lower threshold, collect forgery training data |
| Poor image quality | Accuracy drops significantly below 300 DPI | Add image quality check before processing |
| Very small signatures | Crop step may over-crop | Increase padding in `_crop_to_signature` |
| JWT not enforced | All endpoints are currently unprotected | Add `Depends(get_current_user)` to routes |
| Alembic migrations not scaffolded | `create_all()` used on startup | Run `alembic init` and write migration scripts |
