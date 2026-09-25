# Age / Gender Speech API

Production FastAPI service for **age** and **gender** prediction from speech using:

`audeering/wav2vec2-large-robust-24-ft-age-gender`

Upload a WAV file → get predicted age (years), gender probabilities (female / male / child), or a speaker embedding.

---

## Requirements

- Python **3.10+** (3.11 or 3.13 recommended)
- ~2–4 GB RAM for CPU inference
- Internet on first run (downloads the Hugging Face model), or a pre-warmed HF cache
- Optional: Docker / Docker Compose

---

## Project layout

```
age-gender-api/
├── app/
│   ├── main.py                 # FastAPI app + model load on startup
│   ├── api/routes/             # health, ready, predict endpoints
│   ├── core/                   # settings + logging
│   ├── ml/                     # AgeGenderModel architecture
│   ├── schemas/                # response models
│   └── services/               # WAV loading + inference
├── scripts/run_dev.sh          # helper to start the server
├── tests/                      # basic unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example                # copy to .env and edit
└── README.md
```

---

## 1. Setup (local)

```bash
# Unzip and enter the project
unzip age-gender-api.zip
cd age-gender-api

# Create virtualenv
python3 -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

# Install dependencies (includes torch + transformers — can take a while)
pip install --upgrade pip
pip install -r requirements.txt

# Config
cp .env.example .env
```

### Optional: Hugging Face cache

To store the model in a custom folder (or reuse an existing download):

```bash
# Linux / macOS
export HF_HOME=/path/to/huggingface-cache

# Windows (PowerShell)
# $env:HF_HOME="C:\path\to\huggingface-cache"
```

Or set `HF_HOME=...` inside `.env`.

After the model is cached once, you can set in `.env`:

```env
MODEL_LOCAL_FILES_ONLY=true
```

---

## 2. Run

From the project root (with venv activated):

```bash
# Option A — helper script (Linux / macOS)
chmod +x scripts/run_dev.sh
./scripts/run_dev.sh

# Option B — uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Hot-reload for development:

```bash
RELOAD=true ./scripts/run_dev.sh
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

First startup downloads/loads the model (can take 1–3 minutes). Subsequent starts are faster if `HF_HOME` is reused.

---

## 3. Usage

### Health checks

```bash
curl http://127.0.0.1:8000/api/v1/health
curl http://127.0.0.1:8000/api/v1/ready
```

### Predict age + gender (WAV upload)

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/predict/age-gender" \
  -F "file=@/path/to/your-audio.wav" | python -m json.tool
```

Example response:

```json
{
  "age_years": 27.1,
  "age_normalized": 0.271,
  "gender": {
    "female": 0.015,
    "male": 0.983,
    "child": 0.002
  },
  "predicted_gender": "male",
  "duration_seconds": 6.574,
  "sampling_rate": 16000,
  "filename": "your-audio.wav"
}
```

### Extract embedding

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/predict/embeddings" \
  -F "file=@/path/to/your-audio.wav" | python -m json.tool
```

Returns a 1024-dimensional vector plus metadata.

### Python client example

```python
import requests

url = "http://127.0.0.1:8000/api/v1/predict/age-gender"
with open("sample.wav", "rb") as f:
    r = requests.post(url, files={"file": ("sample.wav", f, "audio/wav")})
print(r.json())
```

### Audio requirements

- Format: **WAV** (PCM)
- Automatically converted to mono 16 kHz
- Default max duration: **60 seconds** (configurable)
- Default max upload size: **25 MB**

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/api/v1/health` | Liveness |
| GET | `/api/v1/ready` | Model loaded? |
| POST | `/api/v1/predict/age-gender` | Age + gender from WAV |
| POST | `/api/v1/predict/embeddings` | Speaker embedding from WAV |

Interactive Swagger UI: `/docs` (disabled in production unless `DEBUG=true`).

---

## 4. Docker

```bash
cd age-gender-api
docker compose up --build -d

# Logs
docker compose logs -f

# Stop
docker compose down
```

API: [http://127.0.0.1:8000](http://127.0.0.1:8000)

The Compose file persists the Hugging Face cache in a Docker volume so the model is not re-downloaded every restart.

To use a host cache instead, uncomment the volume mount in `docker-compose.yml`:

```yaml
# - /path/to/huggingface:/cache/huggingface:ro
```

---

## Configuration (`.env`)

Copy from `.env.example`. Important keys:

| Variable | Default | Meaning |
|----------|---------|---------|
| `ENVIRONMENT` | `development` | `development` / `staging` / `production` |
| `DEBUG` | `true` | Enable docs in production if true |
| `MODEL_NAME` | `audeering/wav2vec2-large-robust-24-ft-age-gender` | HF model id |
| `MODEL_DEVICE` | `cpu` | `cpu`, `cuda`, or `mps` |
| `MODEL_LOCAL_FILES_ONLY` | `false` | Offline mode after cache is warm |
| `HF_HOME` | _(unset)_ | Hugging Face cache directory |
| `MAX_AUDIO_SECONDS` | `60` | Reject longer clips |
| `MAX_UPLOAD_BYTES` | `26214400` | Max upload size (~25 MB) |
| `API_PREFIX` | `/api/v1` | Route prefix |
| `CORS_ORIGINS` | `*` | Comma-separated origins or `*` |

GPU example:

```env
MODEL_DEVICE=cuda
```

(Requires a CUDA-enabled PyTorch install / CUDA Docker image.)

---

## Production tips

- Keep **`workers=1`** unless you have enough RAM for one model copy per worker.
- Set `ENVIRONMENT=production` and `DEBUG=false` to hide `/docs`.
- Set `MODEL_LOCAL_FILES_ONLY=true` after the first successful download.
- Put a reverse proxy (nginx / Caddy) in front for TLS and rate limits.
- Prefer Docker Compose or a process manager (systemd) for restarts.

---

## Tests

```bash
source .venv/bin/activate
pip install pytest
pytest -q
```

---

## License / model credit

API code in this repo is provided for deployment use.

Model: [audeering/wav2vec2-large-robust-24-ft-age-gender](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender) — follow Hugging Face / audeering terms when redistributing or deploying.
