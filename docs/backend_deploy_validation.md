# Backend Deploy Validation (`ai-v1.1-stabilized-c1`)

## Validation objective

Verify whether backend startup and health checks are deploy-ready in a clean setup flow, including behavior without model weights.

## What was validated

### 1) Minimal env vars for startup

Observed from `backend/api.py` lifecycle:

- Required to boot app: `SERVER_PRIVATE_KEY`
- Optional for startup: `ALLOWED_ORIGINS`, `RPC_URL`, `CHAIN_ID`, `CONTRACT_ADDRESS`
- Optional dev fallback path:
  - `ALLOW_INSECURE_DEV_KEY=true`
  - `INSECURE_DEV_PRIVATE_KEY=<key>`

Evidence:

- Starting without key fails startup (expected): runtime error at startup gate.

### 2) Requirements completeness (clean install check)

Command run:

- `python -m pip install --dry-run -r backend/requirements.txt`

Result on current machine (`Python 3.14.3`):

- Failed on pinned dependency: `torch==2.2.2` not available for this Python.

Implication:

- Clean setup must use supported Python runtime (3.11/3.12 for this repo baseline), not Python 3.14.

### 3) Startup command correctness

Validated:

- From backend directory, startup works:
  - `cd backend`
  - `python -m uvicorn api:app --host 127.0.0.1 --port 8011`

Previously failing pattern:

- `python -m uvicorn backend.api:app ...` from repo root failed due local import path (`zkp_oracle`).

Fix applied:

- `backend/Dockerfile` updated to run with `WORKDIR /app/backend` and `uvicorn api:app`.

### 4) Behavior when model weights are absent

Test method:

- Temporarily hid local `ai_deepfake/models/`
- Started backend with required key
- Queried `/api/health`

Result:

- Backend booted successfully.
- Health response returned `detector: false`.
- `zkp_oracle: true` remained healthy.

### 5) `/api/health` behavior in clean startup

With key and models present:

- `/api/health` returned HTTP 200 and:
  - `status: "ok"`
  - `detector: true`
  - `zkp_oracle: true`

With key and models absent:

- `/api/health` returned HTTP 200 and:
  - `status: "ok"`
  - `detector: false`
  - `zkp_oracle: true`

## Deploy readiness decision

**Conditionally deploy-ready.**

Conditions:

1. Deploy runtime must be Python 3.11 or 3.12 (not 3.14 with current torch pin).
2. `SERVER_PRIVATE_KEY` must be configured; otherwise app cannot start.
3. If model weights are not provided at runtime, backend still boots but AI verify endpoints are degraded (`detector=false`).

## Deploy blockers found

### Blocker A (resolved in this pass)

- Container startup command context in `backend/Dockerfile` was incompatible with `api.py` local imports.
- Fixed by running uvicorn from `/app/backend` with `api:app`.

### Blocker B (active condition)

- `backend/requirements.txt` pin `torch==2.2.2` is incompatible with Python 3.14 on clean install.
- Resolution path: use Python 3.11/3.12 for deployment environment.

## Recommended clean deploy validation commands

```bash
# 1) Create clean env (Python 3.11 or 3.12 recommended)
python -m venv .venv
.venv\Scripts\activate

# 2) Install dependencies
pip install -r backend/requirements.txt

# 3) Minimal startup env
set SERVER_PRIVATE_KEY=0x<64_hex_chars>

# 4) Start backend
cd backend
python -m uvicorn api:app --host 127.0.0.1 --port 8000

# 5) Health check in another shell
python -c "import urllib.request, json; print(json.loads(urllib.request.urlopen('http://127.0.0.1:8000/api/health').read().decode()))"
```
