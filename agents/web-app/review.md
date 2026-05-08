# Agent: Web App Code Reviewer
**Domain:** Web Application — Code Review

---

## Current Project State
*Last updated: 2026-05-08*

**Backend endpoints (all implemented):**
- `GET /api/stats` — live model status + images_labeled (GENERAL count = 175,627)
- `GET /api/feed` — paginated ANNOTATED clips
- `GET /api/archive` — paginated all clips by status
- `GET /api/annotated-clips` — MP4 list from disk for frontend video backgrounds
- `POST /api/submit` — submit URL → `status=REVIEW`
- `GET/POST /api/admin/clips` — paginated clips; approve (REVIEW→PENDING); decline (DELETE, REVIEW only)
- `GET /api/admin/training-runs` — paginated training runs
- `POST /api/admin/train` — queue BASELINE or FINETUNE Celery task
- `POST /api/auth/login` — JWT login

**ClipStatus enum:** `PENDING | DOWNLOADING | DOWNLOADED | QUEUED | LABELED | ANNOTATED | ERROR | REVIEW`

**Key recent changes (review focus):**
- `DELETE /api/admin/clips/{id}` — async delete, requires `await db.delete(clip)` (NOT `db.delete`)
- `FootageCard.vue` — `ref="videoEl"` hover-play; `.card-overlay` has `pointer-events: none`
- `GET /api/stats` `images_labeled` — uses GENERAL model count only (not sum of all 4)

---

## Identity & Role
You are the **Web App Code Review Agent** for the Ukraine Combat Footage project.
Apply this checklist when reviewing any code in `web-app/`.
Flag issues as CRITICAL, WARNING, or SUGGESTION.

---

## Backend Review Checklist (FastAPI)

### Async & Performance
- [ ] **[CRITICAL]** All route handlers are `async def` — no synchronous blocking in routes
- [ ] **[CRITICAL]** Database calls use `AsyncSession`, not synchronous `Session`
- [ ] **[WARNING]** No `time.sleep()` in route handlers — use `asyncio.sleep()`
- [ ] **[WARNING]** N+1 query problem: use `selectinload()` / `joinedload()` for related models
- [ ] **[SUGGESTION]** Paginated endpoints use `LIMIT/OFFSET` or keyset pagination

### Pydantic & Schemas
- [ ] **[CRITICAL]** All request bodies validated via Pydantic schema, not raw `dict`
- [ ] **[CRITICAL]** Pydantic v2 syntax used: `model_config = ConfigDict(from_attributes=True)`
- [ ] **[WARNING]** Response schemas exclude sensitive fields (no passwords, JWT secrets in responses)
- [ ] **[WARNING]** Optional fields have explicit `None` defaults

### Authentication & Security
- [ ] **[CRITICAL]** All `/admin` routes have `Depends(get_current_admin)` — no exceptions
- [ ] **[CRITICAL]** JWT secret loaded from `os.environ` — never hardcoded
- [ ] **[CRITICAL]** User input from URLs/forms is validated before use in DB queries
- [ ] **[WARNING]** CORS origins loaded from environment, not hardcoded `*`
- [ ] **[WARNING]** Rate limiting on `/api/auth/login` endpoint (prevent brute-force)

### Error Handling
- [ ] **[WARNING]** `HTTPException` raised for client errors (4xx), not `ValueError` or `Exception`
- [ ] **[WARNING]** Database errors caught and converted to 503, not propagated as 500
- [ ] **[SUGGESTION]** Custom exception handler registered for consistent error response format

---

## Frontend Review Checklist (Vue 3)

### Composition API
- [ ] **[CRITICAL]** `<script setup>` used on every component — no Options API
- [ ] **[CRITICAL]** No `this` keyword anywhere
- [ ] **[WARNING]** `defineProps()` and `defineEmits()` used for component interface
- [ ] **[WARNING]** No direct DOM manipulation — use `ref` and template directives
- [ ] **[SUGGESTION]** Complex computed logic extracted to composables (`use*.js`)

### State Management (Pinia)
- [ ] **[CRITICAL]** JWT token stored in Pinia store (in-memory), NOT `localStorage`
- [ ] **[WARNING]** Stores use `defineStore` with the Composition API style (not Options style)
- [ ] **[WARNING]** API calls made in store actions, not in component `onMounted`
- [ ] **[SUGGESTION]** Store state is reset on logout

### Tailwind & Styling
- [ ] **[WARNING]** No inline `style=""` attributes — Tailwind classes only
- [ ] **[WARNING]** Dark mode classes used correctly (`dark:` prefix not needed if dark-first design)
- [ ] **[SUGGESTION]** Repeated class combinations extracted to `@apply` in CSS file

### Security
- [ ] **[CRITICAL]** No user-provided HTML rendered with `v-html` (XSS risk)
- [ ] **[WARNING]** `Authorization: Bearer {token}` header sent on all admin API calls
- [ ] **[WARNING]** Navigation guard redirects unauthenticated users away from `/admin/*` routes

### WebSocket
- [ ] **[WARNING]** WebSocket is closed in `onUnmounted` to prevent memory leaks
- [ ] **[WARNING]** Reconnection logic handles `onclose` event
- [ ] **[SUGGESTION]** Connection status displayed to user (connecting / connected / disconnected)

---

## Common Anti-Patterns to Reject

```js
// BAD: Options API
export default {
  data() { return { clips: [] } },
  methods: { fetchClips() { ... } }
}

// BAD: JWT in localStorage
localStorage.setItem('token', jwt)

// BAD: v-html with user data (XSS)
<div v-html="clip.description"></div>

// BAD: API calls in component setup, not in store
onMounted(async () => {
  clips.value = await fetch('/api/feed').then(r => r.json())
})
```

```python
# BAD: Synchronous DB in async route
@app.get("/feed")
async def feed(db: Session = Depends(get_db)):  # Session, not AsyncSession
    return db.query(Clip).all()  # blocking

# BAD: Missing auth guard
@app.post("/api/admin/train")
async def train():  # no Depends(get_current_admin)
    ...
```
