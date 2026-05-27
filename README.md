# OMR FastAPI Service

FastAPI service for Optical Music Recognition. It accepts a score image, runs the
OMR pipeline, and returns generated MusicXML, MIDI, and preview file URLs.

## Project Layout

```text
app/                 FastAPI app, routes, settings, error handling
src/                 OMR pipeline, inference, extraction, MusicXML/MIDI build
checkpoints/         TensorFlow model files, not committed to Git
sklearn_models/      sklearn classifier files, not committed to Git
uploads/             runtime uploads, ignored by Git
outputs/             generated outputs, ignored by Git
API.md               API contract and integration notes
```

## Required Model Assets

The service needs these files at runtime:

```text
checkpoints/cvc_unet/arch.json
checkpoints/cvc_unet/cvc_unet.weights.h5
checkpoints/ds2_unet/arch.json
checkpoints/ds2_unet/ds2_unet.weights.h5
sklearn_models/accidental.model
sklearn_models/clef.model
sklearn_models/rests.model
sklearn_models/rests_above8.model
```

These files are intentionally ignored by Git. Put them in a Kaggle Dataset,
private storage, or copy them manually before running the service.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a local `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Set a private API key in `.env`:

```env
OMR_API_KEY=<long-random-secret>
```

Do not commit `.env`.

## Run Locally

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## Main Endpoint

```http
POST /api/v1/omr/process
X-API-Key: <OMR_API_KEY>
Content-Type: multipart/form-data
```

Form field:

```text
file=<score image>
```

Optional query parameters:

```text
include_xml_content=true|false
without_deskew=true|false
tempo=120
instrument=piano
```

Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/omr/process?tempo=120&instrument=piano" `
  -H "X-API-Key: <OMR_API_KEY>" `
  -F "file=@score.jpg"
```

See [API.md](API.md) for the full API contract.

For ngrok/Kaggle or slow images, use the asynchronous job API instead:

```http
POST /api/v1/omr/jobs
GET  /api/v1/omr/jobs/{job_id}
```

`POST /api/v1/omr/jobs` returns immediately with a `job_id`. Poll the status
endpoint until the job is `completed`, then use the returned MusicXML/MIDI URLs.

## GitHub Safety

This repo is prepared so the following are not pushed:

```text
.env
.venv/
uploads/
outputs/
checkpoints/
sklearn_models/
```

Keep secrets in environment variables, GitHub Actions secrets, Kaggle Secrets,
or your deployment platform's secret manager.

## Kaggle + ngrok

Recommended split:

```text
GitHub repo: app/, src/, requirements.txt, API.md, README.md
Kaggle Dataset: checkpoints/, sklearn_models/
Kaggle Secrets: OMR_API_KEY, NGROK_AUTH_TOKEN
```

Use [KAGGLE_NGROK.md](KAGGLE_NGROK.md) as a notebook cell guide.
