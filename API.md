# OMR Service API

This FastAPI app is intended to run as an internal OMR/model service. A web,
Java, Swift, or mobile backend should call this service, then store or proxy the
generated files for its own users.

## Base URL

Local development:

```text
http://127.0.0.1:8000
```

With ngrok/Kaggle:

```text
https://<your-ngrok-domain>
```

## Authentication

Authentication is controlled by `OMR_API_KEY`. When it is set in the
environment, callers must send:

```http
X-API-Key: <OMR_API_KEY>
```

Keep `OMR_API_KEY` set for public/ngrok deployments. If it is empty, the process
endpoint accepts requests without this header.

## Public Deployment Hardening

Recommended settings for ngrok/public demos:

```env
OMR_API_KEY=<long-random-secret>
ENABLE_DOCS=false
TRUSTED_HOSTS=<your-ngrok-host>,localhost,127.0.0.1
CORS_ORIGINS=<your-frontend-origin>
INCLUDE_LOCAL_PATHS=false
INCLUDE_XML_CONTENT_DEFAULT=false
KEEP_UPLOADS=false
```

Additional recommended controls outside this app:

```text
Use HTTPS through ngrok or a reverse proxy.
Keep the OMR service behind your own backend instead of exposing it to browsers.
Store API keys in backend/server secrets, not frontend code.
Rotate the API key if it is leaked.
Add a reverse-proxy rate limit for public deployments.
Limit ngrok sessions to demo/testing, not production.
```

## Health

```http
GET /health
```

Returns whether the service is running and whether required model assets exist.

## Capabilities

```http
GET /api/v1/omr/capabilities
```

Returns supported upload extensions, max upload size, output base URL, model
asset readiness, whether API key auth is required, and supported process
options.

## Process Score Image

```http
POST /api/v1/omr/process
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

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/omr/process?tempo=120&instrument=piano" \
  -H "X-API-Key: change-me" \
  -F "file=@score.jpg"
```

This endpoint is synchronous. For ngrok/Kaggle or slow images, prefer the job
API below so the caller does not hold one HTTP request open for several minutes.

## Queue Score Image Job

```http
POST /api/v1/omr/jobs
Content-Type: multipart/form-data
```

Form field:

```text
file=<score image>
```

Optional query parameters are the same as `/api/v1/omr/process`:

```text
include_xml_content=true|false
without_deskew=true|false
tempo=120
instrument=piano
```

Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/omr/jobs?tempo=120&instrument=piano" \
  -H "X-API-Key: change-me" \
  -F "file=@score.jpg"
```

Queued response:

```json
{
  "success": true,
  "job_id": "20260525_101154_0e0a01",
  "status": "queued",
  "status_url": "/api/v1/omr/jobs/20260525_101154_0e0a01",
  "message": "Job queued. Poll the status_url until status is completed or failed."
}
```

Poll status:

```http
GET /api/v1/omr/jobs/{job_id}
```

Possible statuses:

```text
queued
processing
completed
failed
```

Completed response includes the same `files`, `xml_content`, and `debug_paths`
fields as the synchronous endpoint.

Successful response:

```json
{
  "success": true,
  "job_id": "20260525_101154_0e0a01",
  "status": "completed",
  "processing_time_sec": 8.42,
  "files": {
    "musicxml": {
      "url": "/outputs/20260525_101154_0e0a01/score.xml",
      "filename": "score.xml",
      "content_type": "application/vnd.recordare.musicxml+xml"
    },
    "midi": {
      "url": "/outputs/20260525_101154_0e0a01/score.mid",
      "filename": "score.mid",
      "content_type": "audio/midi"
    }
  },
  "xml_content": null,
  "debug_paths": null
}
```

Errors use this shape:

```json
{
  "success": false,
  "error": {
    "code": "unsupported_file_type",
    "message": "File type '.txt' is not supported"
  }
}
```

## Runtime Settings

Common environment variables:

```env
OMR_API_KEY=<long-random-secret>
ENABLE_DOCS=true
TRUSTED_HOSTS=*
PUBLIC_BASE_URL=
MAX_UPLOAD_MB=15
INCLUDE_LOCAL_PATHS=false
INCLUDE_XML_CONTENT_DEFAULT=false
KEEP_UPLOADS=false
UPLOADS_DIR=
OUTPUTS_DIR=
CHECKPOINTS_DIR=
SKLEARN_MODELS_DIR=
```

`KEEP_UPLOADS=false` removes uploaded source images after processing. Generated
outputs remain available under `/outputs/<job_id>/...`.
