# Financial tracker

Personal finance dashboard with PDF import, salary-window metrics, and category mapping.

**Branches:** Work in **`main`** (full monorepo: API + `web/` + Alembic). For **separate deploys**, use **`frontend`** (SPA only — contents of `web/` at repo root) and **`backend`** (Python API only — no `web/`). Refresh those branches with [`scripts/sync-deploy-branches.ps1`](scripts/sync-deploy-branches.ps1) (or `.sh`) after merging to `main`. See [docs/DEPLOY.md](docs/DEPLOY.md#git-branches-for-separate-deploys).

## Architecture

- **SPA (recommended):** React + Vite in [`web/`](web/) — talks to the REST API with cookie-based auth.
- **API:** FastAPI in [`finance_tracker/api/`](finance_tracker/api/) — same domain logic as the legacy UI.
- **Legacy UI:** Streamlit [`app.py`](app.py) — optional; use for quick local experiments.

## Quick start (local)

### 1. Python API

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
set PYTHONPATH=.
set PASSWORD=yourdevpassword
python -m uvicorn finance_tracker.api.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Web app

```bash
cd web
npm install
npm run dev
```

Open **http://localhost:5173** and sign in with `PASSWORD` (or use `PASSWORD_HASH` on the API only).

### 3. Streamlit (optional)

```bash
set PYTHONPATH=.
streamlit run app.py
```

## Configuration

Secrets and DB URL are read from environment variables (and optionally Streamlit secrets when using `app.py`). See [`finance_tracker/config.py`](finance_tracker/config.py).

- `DB_URL` — database URL (defaults to in-memory SQLite if unset; not suitable for real data).
- `PASSWORD` / `PASSWORD_HASH` — dashboard login.
- `SESSION_SECRET` — API session signing (set in production).
- `CORS_ORIGINS` — browser origins allowed to call the API (production SPA URL).

## Production deployment

See **[docs/DEPLOY.md](docs/DEPLOY.md)** for Hostinger (frontend) + separate API hosting, CORS, and `VITE_API_BASE_URL`.

## Tests

```bash
set PYTHONPATH=.
pytest
```

## API entry

- `GET /health` — liveness and DB connectivity.
- `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`
- `GET /api/dashboard` — KPIs, Sankey data, calendar HTML, to-pay rows.
- `POST /api/imports/pdf` — upload bank statement PDF.
- `GET /api/imports/{id}/uncategorized`, `PATCH /api/transactions/{id}` — map uncategorized lines.

Run `python -m uvicorn finance_tracker.api.main:app` (or `run_api:app` if you use [`run_api.py`](run_api.py)).
