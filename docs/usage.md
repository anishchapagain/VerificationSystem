# use.md — Complete Usage Guide

> Signature Verification System — Step by step from zero to live verification.

---

## What This File Covers

1. [How the System Works — In Plain Terms](#1-how-the-system-works--in-plain-terms)
2. [One Time Setup — Install and Configure](#2-one-time-setup--install-and-configure)
3. [Collecting Signature Images](#3-collecting-signature-images)
4. [Organising Images for Training](#4-organising-images-for-training)
5. [Training the Model](#5-training-the-model)
6. [Starting the System](#6-starting-the-system)
7. [Enrolling a Customer](#7-enrolling-a-customer)
8. [Verifying a Signature](#8-verifying-a-signature)
9. [Checking Verification History](#9-checking-verification-history)
10. [Using the Streamlit Web Interface](#10-using-the-streamlit-web-interface)
11. [Quick Reference — All Commands](#11-quick-reference--all-commands)
12. [Troubleshooting](#12-troubleshooting)
13. [Real World Case — Mr. X, New Customer](#13-real-world-case--mr-x-new-customer)

---

## 1. How the System Works — In Plain Terms

The system has two completely separate activities. Understanding this distinction
is the most important thing before you start.

```
ACTIVITY 1 — TRAINING  (done once, before deployment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You collect signature images from real people.
You run a training command.
The system learns what makes signatures genuine vs forged.
A model file (weights/siamese_best.pt) is saved to disk.
This takes about 1 hour on a GPU or overnight on CPU.
This never needs to happen again unless you want to improve accuracy.


ACTIVITY 2 — DAILY USAGE  (ongoing, through the API)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Customer visits the bank for the first time.
Staff uploads the customer's genuine signature → stored as reference.

Next time a document arrives with that customer's signature:
Staff uploads the signature from the document.
System compares it against the stored reference.
Returns: MATCH or NO MATCH with a score.

No training happens during daily usage.
The model simply applies what it learned during training.
```

---

## 2. One Time Setup — Install and Configure

**Only needs to be done once on your machine.**

### Step 1 — Install Visual C++ Redistributable (Windows only)

Download and install, then restart your terminal:
```
https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### Step 2 — Open Terminal in Project Folder

```powershell
cd D:\PythonProjects\signatureverifier
```

### Step 3 — Create and Activate Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your prompt.

### Step 4 — Install PyTorch

**GPU machine (recommended):**
```powershell
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```powershell
pip install torch==2.4.0 torchvision==0.19.0
```

Verify:
```powershell
python -c "import torch; print('PyTorch OK:', torch.__version__)"
```

### Step 5 — Install FAISS

**GPU machine:**
```powershell
pip install faiss-gpu==1.7.4
```

**CPU only:**
```powershell
pip install faiss-cpu==1.8.0
```

### Step 6 — Install Remaining Dependencies

```powershell
pip install -r requirements.txt
```

### Step 7 — Configure Environment

```powershell
copy .env.example .env
```

Open `.env` and set your PostgreSQL password:
```
DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/signature_db
DEVICE=cuda       # change to cpu if no GPU
MATCH_THRESHOLD=0.85
```

### Step 8 — Verify PostgreSQL is Running

```powershell
pg_isready -U postgres
# Expected: localhost:5432 - accepting connections
```

If not running, open Windows Services and start `postgresql-x64-16`.

---

## 3. Collecting Signature Images

Before training, you need to collect signature images from real people.
These images are used only for training — not for the daily API workflow.

### What You Need

| Item | Minimum | Recommended |
|------|---------|-------------|
| Number of people | 5 | 20 or more |
| Genuine signatures per person | 10 | 15 to 20 |
| Forged signatures per person | 5 | 10 |
| Total images | 75 | 500 or more |
| Collection period | 3 days | 2 to 3 weeks |

### Genuine Signatures

- Ask each person to sign naturally on plain white paper
- Use a black ballpoint pen
- Collect across **multiple days** — do not collect all in one session
- Scan at 300 DPI or photograph from directly above with even lighting
- Save as PNG

> Collecting across multiple days is important. Signatures vary naturally
> from day to day. If you collect all 15 in one sitting they look too similar
> and the model learns a rigid template that fails in real use.

### Forged Signatures

- Show someone a genuine signature sample
- Ask them to imitate it as closely as possible
- Collect 5 to 10 attempts per person
- These teach the model what an imitation looks like

### Image Quality Requirements

| Property | Required |
|----------|----------|
| Paper | Plain white, unlined, no watermark |
| Ink | Black or dark blue pen |
| DPI | 300 minimum |
| Format | PNG |
| Background | Clean white — no shadows, no lines |
| Size | Crop to signature area, leave small white border |

---

## 4. Organising Images for Training

The training script expects images in a specific folder structure.
Create these folders in your project directory:

```
signatureverifier/
└── data/
    └── processed/
        ├── genuine/
        │   ├── user_001_sig_01.png
        │   ├── user_001_sig_02.png
        │   ├── user_001_sig_03.png
        │   ├── user_002_sig_01.png
        │   ├── user_002_sig_02.png
        │   └── ...
        └── forged/
            ├── user_001_forg_01.png
            ├── user_001_forg_02.png
            ├── user_002_forg_01.png
            └── ...
```

### Naming Rules

```
Genuine:  user_XXX_sig_YY.png
Forged:   user_XXX_forg_YY.png

Where:
  XXX = three digit user number  (001, 002, 003 ...)
  YY  = two digit image number   (01, 02, 03 ...)
```

**The user number must match between genuine and forged files.**
The system uses this prefix to pair each person's genuine signatures
with their corresponding forgeries during training.

### Create the Folders (Windows)

```powershell
mkdir data\processed\genuine
mkdir data\processed\forged
```

### Place Your Images

Copy your collected signature images into the correct folders
following the naming convention above.

Verify the count:
```powershell
# Count genuine images
(Get-ChildItem data\processed\genuine\*.png).Count

# Count forged images
(Get-ChildItem data\processed\forged\*.png).Count
```

---

## 5. Training the Model

Training teaches the model to distinguish genuine signatures from forgeries.
You only need to do this once. The trained model is saved as a file and
the API uses it for all future verifications.

### Before Training — Generate Placeholder Weights

This lets you start and test the API immediately while training runs:

```powershell
python scripts/generate_weights.py
```

Output:
```
✅  weights/siamese_best.pt created successfully!
⚠️  Verification SCORES are random until training completes.
```

### Run Training

**GPU machine (recommended — takes about 1 hour):**
```powershell
python -m ml.train ^
    --data_dir data/processed ^
    --epochs 100 ^
    --batch_size 64 ^
    --output weights/siamese_best.pt
```

**CPU only (takes 10 to 18 hours — run overnight):**
```powershell
python -m ml.train ^
    --data_dir data/processed ^
    --epochs 100 ^
    --batch_size 16 ^
    --workers 2 ^
    --output weights/siamese_best.pt
```

### What You See During Training

```
============================================================
  Signature Verifier  v2.0.0
  Device      : cuda
  GPU         : NVIDIA GeForce RTX XXXX
  VRAM        : 8.0 GB
  AMP FP16    : True
  Batch size  : 64
  Epochs      : 100
============================================================
Dataset | train=680 pairs | val=120 pairs
Model parameters: 34,xxx,xxx

Epoch   1/100 | train_loss=0.4821 | val_loss=0.4612 | EER=0.3201 | acc=0.6812 | time=32.1s
Epoch   5/100 | train_loss=0.2341 | val_loss=0.2198 | EER=0.1854 | acc=0.8211 | time=31.4s
  ✅ Best checkpoint saved | EER=0.1854 | path=weights/siamese_best.pt
...
Epoch  40/100 | train_loss=0.0821 | val_loss=0.0934 | EER=0.0512 | acc=0.9421 | time=30.8s
  ✅ Best checkpoint saved | EER=0.0512
...
Training complete | best EER=0.0421 | saved to weights/siamese_best.pt
```

### What the Numbers Mean

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| train_loss | How well the model fits training data | Decreasing over epochs |
| val_loss | How well it generalises to unseen data | Decreasing, close to train_loss |
| EER | Equal Error Rate — lower is better | Below 0.08 (8%) |
| acc | Accuracy at the 0.85 threshold | Above 0.90 (90%) |

Training stops automatically when the model stops improving
(after 15 epochs without improvement — early stopping).

### After Training

Training saves the best model automatically to `weights/siamese_best.pt`.
This file is now your trained model. The API will load it on next startup.

> If you are upgrading from placeholder weights, restart the API after
> training completes so it loads the real trained weights.

---

## 6. Starting the System

### Every Time You Work with the System

**Terminal 1 — Start the API:**
```powershell
cd D:\PythonProjects\signatureverifier
.venv\Scripts\activate
python backend/main.py
```

Wait for these lines:
```
Database 'signature_db' already exists.
All tables created / verified.
Model loaded | device=cuda | params=34,xxx,xxx
Startup complete — ready to accept requests.
  Swagger UI  : http://localhost:8000/docs
  Health      : http://localhost:8000/health
  Frontend    : http://localhost:8501
```

**Terminal 2 — Start the Frontend:**
```powershell
cd D:\PythonProjects\signatureverifier
.venv\Scripts\activate
streamlit run frontend/app.py
```

### Verify Everything is Working

```powershell
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

## 7. Enrolling a Customer

Enrolment stores a customer's genuine signature as their reference.
This must be done before any verification can happen for that customer.

**Enrolment happens once per customer.**
After enrolment, the customer can be verified any number of times
without uploading their reference again.

### Step 1 — Create a User Account for the Customer

Using Swagger UI at `http://localhost:8000/docs`:

Click `POST /api/users/register` → Try it out → fill in:
```json
{
  "name": "Ram Bahadur Thapa",
  "email": "ram.thapa@example.com",
  "password": "SecurePass123!"
}
```

Response:
```json
{
  "id": 1,
  "name": "Ram Bahadur Thapa",
  "email": "ram.thapa@example.com",
  "created_at": "2026-03-16T10:00:00Z"
}
```

Note the `id` — you will use it for all future operations for this customer.

Using curl:
```powershell
curl -X POST http://localhost:8000/api/users/register ^
  -H "Content-Type: application/json" ^
  -d "{\"name\": \"Ram Bahadur Thapa\", \"email\": \"ram@example.com\", \"password\": \"Pass123!\"}"
```

---

### Step 2 — Upload the Reference Signature

Click `POST /api/signatures/register` → Try it out → fill in:
```
file     →  Choose File  →  select the customer's signature image
user_id  →  1            →  the id from Step 1
label    →  Primary      →  optional label
```

Using curl:
```powershell
curl -X POST http://localhost:8000/api/signatures/register ^
  -F "file=@C:\signatures\ram_genuine_01.png" ^
  -F "user_id=1" ^
  -F "label=Primary"
```

Response:
```json
{
  "signature_id": 1,
  "user_id": 1,
  "label": "Primary",
  "file_path": "storage/signatures/abc123de.png",
  "message": "Signature registered successfully."
}
```

The customer is now enrolled. Their signature is stored in the database
and ready to be verified against.

### Enrol Multiple Reference Signatures (Recommended)

For better accuracy, enrol 3 to 5 signatures collected on different days.
Repeat Step 2 for each signature using the same `user_id`.

```powershell
# First reference
curl -X POST http://localhost:8000/api/signatures/register ^
  -F "file=@C:\signatures\ram_sig_day1.png" ^
  -F "user_id=1" ^
  -F "label=Day 1"

# Second reference (different day)
curl -X POST http://localhost:8000/api/signatures/register ^
  -F "file=@C:\signatures\ram_sig_day2.png" ^
  -F "user_id=1" ^
  -F "label=Day 2"

# Third reference
curl -X POST http://localhost:8000/api/signatures/register ^
  -F "file=@C:\signatures\ram_sig_day3.png" ^
  -F "user_id=1" ^
  -F "label=Day 3"
```

---

## 8. Verifying a Signature

When a document arrives with a signature, upload that signature to verify
it against the enrolled reference for that customer.

### What You Need

- The signature image from the document (scan or photograph)
- The `user_id` of the customer the document claims to be signed by

### Run Verification

Click `POST /api/signatures/verify` → Try it out → fill in:
```
file     →  Choose File  →  select the signature from the document
user_id  →  1            →  the customer's id
```

Using curl:
```powershell
curl -X POST http://localhost:8000/api/signatures/verify ^
  -F "file=@C:\signatures\document_signature.png" ^
  -F "user_id=1"
```

### Response

```json
{
  "user_id": 1,
  "verdict": true,
  "verdict_label": "MATCH",
  "score": 0.923456,
  "confidence": "High",
  "threshold_used": 0.85,
  "best_match_signature_id": 2,
  "source": "image",
  "score_breakdown": [
    { "signature_id": 1, "score": 0.891200 },
    { "signature_id": 2, "score": 0.923456 },
    { "signature_id": 3, "score": 0.876100 }
  ],
  "match_log_id": 44,
  "processed_at": "2026-03-16T10:25:00Z"
}
```

### Reading the Result

| Field | What It Means |
|-------|---------------|
| `verdict` | `true` = MATCH, `false` = NO MATCH |
| `verdict_label` | Plain text MATCH or NO MATCH |
| `score` | Similarity score 0.0 to 1.0 — higher means more similar |
| `confidence` | Very High / High / Medium / Low / Very Low |
| `threshold_used` | Score must meet or exceed this to return MATCH |
| `score_breakdown` | Score against each stored reference individually |
| `match_log_id` | ID of the audit log entry — use for compliance records |

### Confidence Guide

| Score Range | Confidence | What to Do |
|-------------|------------|------------|
| 0.95 – 1.0 | Very High | Proceed with confidence |
| 0.90 – 0.94 | High | Proceed |
| 0.85 – 0.89 | Medium | Proceed — consider manual glance |
| 0.70 – 0.84 | Low | NO MATCH — send to officer review |
| Below 0.70 | Very Low | NO MATCH — flag as suspected forgery |

### Adjusting the Threshold

The default threshold is `0.85`. You can adjust it in `.env`:

```
# Stricter — fewer false acceptances, more rejections
MATCH_THRESHOLD=0.90

# More lenient — fewer false rejections, more acceptances
MATCH_THRESHOLD=0.80
```

Restart the API after changing the threshold.

---

## 9. Checking Verification History

Every verification is logged automatically. Retrieve the full history
for a customer at any time.

```powershell
curl http://localhost:8000/api/signatures/history/1?limit=50
```

Or in Swagger UI: `GET /api/signatures/history/{user_id}`

Response:
```json
{
  "user_id": 1,
  "total_returned": 3,
  "logs": [
    {
      "id": 44,
      "score": 0.923456,
      "verdict": true,
      "verdict_label": "MATCH",
      "threshold_used": 0.85,
      "source": "image",
      "created_at": "2026-03-16T10:25:00Z"
    },
    {
      "id": 38,
      "score": 0.312100,
      "verdict": false,
      "verdict_label": "NO MATCH",
      "threshold_used": 0.85,
      "source": "image",
      "created_at": "2026-03-15T14:10:00Z"
    }
  ]
}
```

### List All Enrolled Signatures for a Customer

```powershell
curl http://localhost:8000/api/signatures/1
```

### Delete a Reference Signature

If a signature needs to be replaced (damaged image, customer request):

```powershell
curl -X DELETE http://localhost:8000/api/signatures/{signature_id}
```

This soft-deletes the signature — it is deactivated but the record
is retained in the database for audit purposes.

---

## 10. Using the Streamlit Web Interface

The Streamlit frontend provides a simple click-and-upload interface
for staff who prefer not to use Swagger or curl.

Open: `http://localhost:8501`

### Page 1 — Register

1. Set your User ID in the sidebar
2. Click **Upload Signature Image**
3. Select the signature PNG file
4. Add a label (optional — e.g. "Primary")
5. Click **Register Signature**
6. Confirmation shows the stored signature ID

### Page 2 — Verify

1. Set your User ID in the sidebar
2. Click **Upload Query Signature**
3. Select the signature from the document
4. Click **Verify Now**
5. Result shows:
   - Large MATCH ✅ or NO MATCH ❌
   - Score and confidence level
   - Score bar for each stored reference

### Page 3 — History

1. Set your User ID in the sidebar
2. Click **Load History**
3. See all past verifications with scores, verdicts, and timestamps

---

## 11. Quick Reference — All Commands

### Setup (once)

```powershell
# Activate environment
.venv\Scripts\activate

# Install PyTorch GPU
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install FAISS GPU
pip install faiss-gpu==1.7.4

# Install all other packages
pip install -r requirements.txt

# Configure environment
copy .env.example .env
```

### Training (once)

```powershell
# Create placeholder weights so API works immediately
python scripts/generate_weights.py

# Train on your images (GPU — ~1 hour)
python -m ml.train --data_dir data/processed --epochs 100 --batch_size 64

# Train on your images (CPU — run overnight)
python -m ml.train --data_dir data/processed --epochs 100 --batch_size 16 --workers 2
```

### Start the System (every session)

```powershell
# Terminal 1 — API
python backend/main.py

# Terminal 2 — Frontend
streamlit run frontend/app.py
```

### Enrolment

```powershell
# Create user
curl -X POST http://localhost:8000/api/users/register ^
  -H "Content-Type: application/json" ^
  -d "{\"name\": \"Name\", \"email\": \"email@example.com\", \"password\": \"Pass123!\"}"

# Register reference signature
curl -X POST http://localhost:8000/api/signatures/register ^
  -F "file=@C:\path\to\signature.png" ^
  -F "user_id=1"
```

### Verification

```powershell
# Verify a signature
curl -X POST http://localhost:8000/api/signatures/verify ^
  -F "file=@C:\path\to\document_signature.png" ^
  -F "user_id=1"

# Check health
curl http://localhost:8000/health

# View history
curl http://localhost:8000/api/signatures/history/1
```

### URLs

| Service | URL |
|---------|-----|
| API Swagger UI | http://localhost:8000/docs |
| API ReDoc | http://localhost:8000/redoc |
| Health Check | http://localhost:8000/health |
| Streamlit UI | http://localhost:8501 |

---

## 12. Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `fbgemm.dll not found` | Missing Visual C++ Redistributable | Install from https://aka.ms/vs/17/release/vc_redist.x64.exe and restart terminal |
| `import torch` fails | Wrong PyTorch version | Reinstall with correct CUDA URL |
| `503 model not loaded` | Weights file missing | Run `python scripts/generate_weights.py` |
| `database connection refused` | PostgreSQL not running | Open Windows Services, start postgresql-x64-16 |
| `module not found` | Virtual env not active | Run `.venv\Scripts\activate` |
| `port 8000 in use` | Another process on that port | Kill the process or edit port in `backend/main.py` |
| Score always random | Placeholder weights still loaded | Run training, then restart the API |
| Score low for genuine sig | Poor image quality | Rescan at 300 DPI, ensure plain white background |
| NO MATCH for genuine sig | Threshold too strict | Lower `MATCH_THRESHOLD` in `.env` to 0.80 |
| MATCH for forged sig | Threshold too lenient | Raise `MATCH_THRESHOLD` in `.env` to 0.90 |
| Training loss not decreasing | Too few images | Collect more signatures — minimum 75, ideally 500+ |
| GPU out of memory during training | Batch size too large | Reduce `--batch_size` to 32 or 16 |

---

## 13. Real World Case — Mr. X, New Customer

This section walks through the exact flow for a brand new customer who
has never used the system before. It covers every step from first visit
to ongoing verification.

---

### The Important Distinction — What the Model Knows vs What the System Knows

Before anything else, understand this clearly:

```
The model was trained on Person A, B, C, D, E ... signatures.
Mr. X is a completely new person.
The model has NEVER seen Mr. X's signature during training.

Can the system still verify Mr. X?   YES.

Why?
The model did not learn WHO specific people are.
It learned HOW TO COMPARE any two signatures.
That skill works on Mr. X even though he never appeared in training data.
```

---

### Phase 1 — Mr. X Visits the Bank for the First Time

There is no verdict possible yet. The system has no reference for Mr. X.
The only thing that can happen at this stage is enrolment.

```
Mr. X arrives at the bank counter
          │
          ▼
Staff creates a user account for Mr. X

  POST /api/users/register
  {
    "name": "Mr. X",
    "email": "mrx@example.com",
    "password": "SecurePass123!"
  }

  Response: { "id": 7, "name": "Mr. X" }
  Note the id = 7  — this is Mr. X's permanent identifier

          │
          ▼
Staff asks Mr. X to sign on plain white paper
Staff scans the signature at 300 DPI, saves as PNG
Staff uploads it to the system

  POST /api/signatures/register
  file    = mr_x_signature.png
  user_id = 7

          │
          ▼
System runs internally:

  1. Image saved to  storage/signatures/abc123.png
  2. Preprocessor cleans the image
     → grayscale → denoise → binarise → crop → resize → normalise
  3. Model converts it to a 512-number embedding
     [ 0.231, -0.814, 0.442, 0.119, ... 512 numbers total ]
  4. Embedding saved to PostgreSQL — signatures table
  5. Embedding indexed in FAISS for fast future retrieval

  Response: { "signature_id": 12, "message": "Signature registered successfully." }

          │
          ▼
Mr. X is now enrolled.
His reference embedding is stored.
No verdict was generated — this was enrolment only.
No training happened — the model was not touched.
```

---

### Phase 2 — Mr. X Submits a Cheque Three Weeks Later

Now a verdict is possible because a reference exists from Phase 1.

```
Mr. X presents a cheque at the bank
          │
          ▼
Staff looks at the cheque and scans just the signature portion
Staff uploads it to the system with Mr. X's user_id

  POST /api/signatures/verify
  file    = cheque_signature.png
  user_id = 7

          │
          ▼
System runs internally:

  ┌────────────────────────────────────────────────────┐
  │  QUERY SIDE — the new signature from the cheque    │
  │                                                    │
  │  1. Image saved to storage/signatures/queries/     │
  │  2. Preprocessor cleans the image                  │
  │     (same 8-step pipeline as enrolment)            │
  │  3. Model converts it to a 512-number embedding    │
  │     [ 0.198, -0.801, 0.461, 0.134, ... ]           │
  └────────────────────────────────────────────────────┘
          │
          ▼
  ┌────────────────────────────────────────────────────┐
  │  REFERENCE SIDE — stored at enrolment              │
  │                                                    │
  │  4. System loads Mr. X's stored embedding          │
  │     from PostgreSQL for user_id = 7                │
  │     [ 0.231, -0.814, 0.442, 0.119, ... ]           │
  └────────────────────────────────────────────────────┘
          │
          ▼
  ┌────────────────────────────────────────────────────┐
  │  COMPARISON                                        │
  │                                                    │
  │  5. Cosine similarity computed between the two     │
  │     embedding vectors                              │
  │                                                    │
  │     Query:     [ 0.198, -0.801, 0.461, ... ]       │
  │     Reference: [ 0.231, -0.814, 0.442, ... ]       │
  │                                                    │
  │     Score = 0.923  — vectors point in very         │
  │                       similar directions           │
  └────────────────────────────────────────────────────┘
          │
          ▼
  ┌────────────────────────────────────────────────────┐
  │  VERDICT                                           │
  │                                                    │
  │  6. Score 0.923 >= threshold 0.85                  │
  │     → Verdict = MATCH                              │
  │                                                    │
  │  7. Audit log written to match_logs table          │
  │     id=44, user_id=7, score=0.923, verdict=true    │
  └────────────────────────────────────────────────────┘
          │
          ▼
Response returned to staff:

  {
    "user_id": 7,
    "verdict": true,
    "verdict_label": "MATCH",
    "score": 0.923456,
    "confidence": "High",
    "threshold_used": 0.85,
    "best_match_signature_id": 12,
    "source": "image",
    "match_log_id": 44
  }

Staff sees MATCH → proceeds with the cheque.
```

---

### Phase 3 — What If Someone Forges Mr. X's Cheque?

Same flow, but the signature on the cheque is a forgery.

```
A forged cheque arrives at the bank
Staff scans the signature and uploads it

  POST /api/signatures/verify
  file    = forged_cheque_signature.png
  user_id = 7

System compares the forged signature embedding against
Mr. X's stored reference embedding.

The forged signature looks visually similar to a human eye
but the model detects differences in stroke patterns,
pen pressure distribution, loop proportions.

  Query (forgery): [ 0.412, -0.321, 0.088, ... ]
  Reference (genuine): [ 0.231, -0.814, 0.442, ... ]

  Score = 0.312  — vectors point in very different directions

  Score 0.312 < threshold 0.85 → VERDICT = NO MATCH

Response returned:
  {
    "verdict": false,
    "verdict_label": "NO MATCH",
    "score": 0.312,
    "confidence": "Very Low",
    "threshold_used": 0.85
  }

Staff sees NO MATCH → sends to officer review queue.
Transaction is held pending manual investigation.
```

---

### What If Mr. X Was Never Enrolled?

```
Staff tries to verify a signature but Mr. X was never enrolled

  POST /api/signatures/verify
  file    = some_signature.png
  user_id = 7

System looks up stored references for user_id = 7
Finds zero entries in the signatures table

Returns error:
  {
    "status_code": 404,
    "detail": "No reference signatures found for this user.
               Please register at least one reference signature first.",
    "error_type": "NoReferenceSignatureError"
  }

No verdict is possible.
Enrolment must always happen before verification.
```

---

### What If Mr. X's Signature Has Changed Over Time?

Signatures change gradually — age, injury, different pen.
If Mr. X's score starts dropping on genuine verifications,
add a new reference signature.

```
  POST /api/signatures/register
  file    = mr_x_updated_signature.png
  user_id = 7
  label   = "Updated 2026"

System now stores TWO references for Mr. X.
On the next verification it compares against BOTH
and uses the highest score.
Mr. X's old and new signing styles are both accepted.
```

---

### All Scenarios for Mr. X — Summary Table

| Situation | What Happens | Verdict? |
|-----------|-------------|----------|
| Mr. X first visit — enrolment | POST /register — reference stored | No verdict — enrolment only |
| Mr. X submits genuine cheque | POST /verify — score 0.85+ | ✅ MATCH |
| Someone submits forged cheque | POST /verify — score below 0.85 | ❌ NO MATCH |
| Mr. X never enrolled | POST /verify attempted | 404 error — no verdict possible |
| Mr. X enrols multiple signatures | POST /register multiple times | Best score across all references used |
| Mr. X's signature changed | POST /register new reference | New and old both accepted |
| Wrong user_id used | POST /verify with wrong id | Compared against wrong person's reference |

---

### The One Rule That Never Changes

```
No enrolment = No verdict.

Enrolment must always come first.
Without a stored reference there is nothing to compare against.
The system cannot verify a person it has never seen before.
```


---

## Summary — The Three Things You Must Do

```
1. COLLECT AND TRAIN  (once)
   Collect genuine + forged signature images
   Place in data/processed/genuine/ and data/processed/forged/
   Run: python -m ml.train --data_dir data/processed --batch_size 64
   Wait for training to complete (~1 hour GPU)

2. ENROL CUSTOMERS  (once per customer)
   Create user account via /api/users/register
   Upload their genuine signature via /api/signatures/register
   Customer is now in the system

3. VERIFY SIGNATURES  (every transaction)
   Upload signature from document via /api/signatures/verify
   Read the score and verdict
   MATCH = proceed, NO MATCH = send to officer review
```

That is the complete workflow. Training happens once.
Enrolment happens once per customer.
Verification happens every time a signed document arrives.
