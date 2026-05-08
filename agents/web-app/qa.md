# Agent: Web App QA
**Domain:** Web Application — Quality Assurance

---

## Current Project State
*Last updated: 2026-05-08*

**Live DB state:** 10 clips total — 8 ANNOTATED, 1 PENDING, 1 ERROR. 0 REVIEW clips.
**Backend:** FastAPI on port 8000 (`python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000`)
**Frontend:** Vite on port 5173 (`npm run dev` in `web-app/frontend/`)
**Admin credentials:** `admin` / `admin123` (from `web-app/backend/.env`)

**AdminPanel current capabilities:**
- Clips table: paginated, filter tabs (ALL / PENDING / ANNOTATED / REVIEW / ERROR)
- APPROVE button on REVIEW clips → `POST /api/admin/clips/{id}/approve`
- DECLINE button on REVIEW clips → `DELETE /api/admin/clips/{id}`
- PREVIEW button on all clips → modal with video (ANNOTATED) or URL link (REVIEW)
- Training runs table with mAP50 + status
- Train buttons (BASELINE/FINETUNE per model)

**Known gaps (not yet implemented):**
- WebSocket training progress (3.14/3.14b)
- Celery worker E2E not verified end-to-end
- Hero background video (waiting for GENERAL annotated clips to accumulate)

---

## Identity & Role
You are the **Web App QA Agent** for the Ukraine Combat Footage project.
Your job is to validate that the FastAPI backend and Vue 3 frontend meet
correctness, security, UX, and performance requirements.

---

## Backend QA Checklist

### API Contract
- [ ] `GET /api/feed` returns paginated JSON with `clips[]`, `total`, `page`, `per_page`
- [ ] `GET /api/archive?q=&from=&to=&page=` supports all query params
- [ ] `POST /api/submit` validates URL format and returns `201` on success
- [ ] `GET /api/admin/datasets` returns only datasets with `status=LABELED`
- [ ] `POST /api/admin/train` accepts `{stage: "BASELINE"|"FINETUNE", dataset_ids: int[]}` and returns Celery task ID
- [ ] `WebSocket /ws/training/{run_id}` sends JSON `{epoch, loss, mAP50, status}` every 2s
- [ ] All `/api/admin/*` endpoints return `401` without valid JWT
- [ ] `POST /api/auth/login` returns `401` on wrong credentials (not 500)

### Error Handling
- [ ] 404 returned for unknown clip/dataset IDs (not 500)
- [ ] 422 returned for invalid request bodies (Pydantic validation errors)
- [ ] Database connection errors return 503, not 500 with stack trace
- [ ] No stack traces exposed in production error responses

### Security
- [ ] JWT tokens expire (recommended: 8 hours)
- [ ] JWT secret is loaded from environment variable, not hardcoded
- [ ] Admin password is hashed with bcrypt (not stored as plaintext)
- [ ] CORS is configured to only allow the frontend origin
- [ ] No sensitive data (password, JWT secret) in API responses or logs

---

## Frontend QA Checklist

### PublicFeed (`/`)
- [ ] Clips load on mount and display as a responsive grid
- [ ] Each `VideoCard` shows: thumbnail/video player, title, source, date
- [ ] Feed auto-refreshes every 60 seconds without full page reload
- [ ] Loading skeleton shown while fetching
- [ ] Empty state shown when no clips available ("No footage yet")
- [ ] Pagination or infinite scroll works correctly

### Archive (`/archive`)
- [ ] Search input is debounced (300ms) — not firing on every keystroke
- [ ] Date range filter works correctly
- [ ] Search results update without full page reload
- [ ] Empty search results show friendly message

### AdminPanel (`/admin`)
- [ ] Unauthenticated users are redirected to `/admin/login`
- [ ] Training runs table shows latest run per model with mAP50, status, timestamps
- [ ] BASELINE / FINETUNE train buttons dispatch to `POST /api/admin/train`; success shows amber msg, error shows red bordered panel for 10s
- [ ] Clips table paginated; filter tabs include REVIEW status
- [ ] APPROVE button visible on REVIEW clips; calls `POST /api/admin/clips/{id}/approve`
- [ ] Error: duplicate active run → red panel "ERROR: {model} already has a QUEUED run"
- [ ] Error: FINETUNE without baseline → red panel "No completed baseline run found"

### Cross-Cutting
- [ ] Dark mode renders correctly on all views (no white flash)
- [ ] All interactive elements are keyboard-accessible (Tab, Enter)
- [ ] Mobile responsive down to 375px width
- [ ] No console errors in browser devtools on any view
- [ ] JWT token is NOT stored in `localStorage` (use Pinia in-memory store)
