<template>
  <div class="panel-root">

    <!-- header -->
    <header class="panel-header">
      <a href="/" class="nav-logo" style="text-decoration:none;cursor:pointer">
        <div class="nav-logo-mark"></div>
        UKRARCHIVE
        <span class="mono" style="color:var(--fg-3);font-size:10px">ADMIN</span>
      </a>
      <div style="display:flex;align-items:center;gap:24px">
        <div class="nav-status">
          <div class="status-dot"></div>
          <span class="mono" style="font-size:11px">SYSTEM ONLINE</span>
        </div>
        <a href="/" class="btn-back mono">← HOME</a>
        <button class="btn-access" @click="logout">LOGOUT</button>
      </div>
    </header>

    <main class="panel-main">

      <!-- ── TRAINING ── -->
      <section class="panel-section">
        <div class="panel-section-title mono">TRAINING CONTROL</div>

        <div class="model-grid">
          <div v-for="m in MODELS" :key="m" class="model-card" :data-model="m">
            <div class="model-card-top">
              <div class="mono" style="font-size:10px;letter-spacing:0.2em;color:var(--cat-color)">{{ m }}</div>
              <div :class="['run-status', latestRun(m)?.status?.toLowerCase()]" class="mono">
                {{ latestRun(m)?.status ?? 'NO RUNS' }}
              </div>
            </div>
            <div v-if="latestRun(m)?.map50 != null" class="mono" style="font-size:16px;font-weight:300;color:var(--fg-0);letter-spacing:-0.02em;margin:2px 0">
              {{ latestRun(m).map50.toFixed(3) }}
              <span style="font-size:10px;color:var(--fg-3);letter-spacing:0.1em">mAP50</span>
            </div>
            <div v-else style="height:4px"></div>
            <div style="display:flex;gap:8px;margin-top:auto">
              <button class="train-btn mono" :disabled="launching === m" @click="queueTrain(m, 'BASELINE')">
                {{ launching === m ? '...' : 'BASELINE' }}
              </button>
              <button class="train-btn mono" :disabled="launching === m" @click="queueTrain(m, 'FINETUNE')">
                {{ launching === m ? '...' : 'FINETUNE' }}
              </button>
            </div>
          </div>
        </div>

        <div v-if="trainMsg" class="panel-msg mono">{{ trainMsg }}</div>
      </section>

      <!-- ── TRAINING RUNS ── -->
      <section class="panel-section">
        <div class="panel-section-title mono">TRAINING RUNS</div>
        <div v-if="runsLoading" class="panel-loading mono">LOADING...</div>
        <table v-else class="panel-table">
          <thead>
            <tr>
              <th>ID</th><th>MODEL</th><th>STAGE</th><th>STATUS</th><th>mAP50</th><th>STARTED</th><th>COMPLETED</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in runs" :key="r.id">
              <td class="mono dim">#{{ r.id }}</td>
              <td class="mono" :style="{ color: modelColor(r.model_type) }">{{ r.model_type }}</td>
              <td class="mono dim">{{ r.stage }}</td>
              <td><span :class="['run-status', r.status.toLowerCase()]" class="mono">{{ r.status }}</span></td>
              <td class="mono">{{ r.map50 != null ? r.map50.toFixed(3) : '—' }}</td>
              <td class="mono dim">{{ fmtDate(r.started_at) }}</td>
              <td class="mono dim">{{ fmtDate(r.completed_at) }}</td>
            </tr>
            <tr v-if="!runs.length">
              <td colspan="7" class="mono dim" style="text-align:center;padding:10px">NO TRAINING RUNS YET</td>
            </tr>
          </tbody>
        </table>
        <div class="panel-pagination">
          <button class="page-btn mono" :disabled="runsPage <= 1" @click="runsPage--; loadRuns()">PREV</button>
          <span class="mono dim" style="font-size:11px">PAGE {{ runsPage }}</span>
          <button class="page-btn mono" :disabled="runs.length < 20" @click="runsPage++; loadRuns()">NEXT</button>
        </div>
      </section>

      <!-- ── CLIPS ── -->
      <section class="panel-section">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
          <div class="panel-section-title mono" style="margin-bottom:0">CLIP ARCHIVE</div>
          <div style="display:flex;gap:8px">
            <button
              v-for="s in CLIP_STATUSES"
              :key="s"
              :class="['status-filter-btn mono', { active: clipStatus === s }]"
              @click="clipStatus = s; clipsPage = 1; loadClips()"
            >{{ s || 'ALL' }}</button>
          </div>
        </div>
        <div v-if="clipsLoading" class="panel-loading mono">LOADING...</div>
        <table v-else class="panel-table">
          <thead>
            <tr><th>ID</th><th>TITLE</th><th>SOURCE</th><th>STATUS</th><th>DURATION</th><th>ADDED</th></tr>
          </thead>
          <tbody>
            <tr v-for="c in clips" :key="c.id">
              <td class="mono dim">#{{ c.id }}</td>
              <td style="max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{{ c.title ?? c.url }}</td>
              <td class="mono dim">{{ c.source }}</td>
              <td><span :class="['clip-status', c.status.toLowerCase()]" class="mono">{{ c.status }}</span></td>
              <td class="mono dim">{{ c.duration_seconds ? fmtDur(c.duration_seconds) : '—' }}</td>
              <td class="mono dim">{{ fmtDate(c.created_at) }}</td>
            </tr>
            <tr v-if="!clips.length">
              <td colspan="6" class="mono dim" style="text-align:center;padding:10px">NO CLIPS FOUND</td>
            </tr>
          </tbody>
        </table>
        <div class="panel-pagination">
          <button class="page-btn mono" :disabled="clipsPage <= 1" @click="clipsPage--; loadClips()">PREV</button>
          <span class="mono dim" style="font-size:11px">PAGE {{ clipsPage }} · {{ clipsTotal }} TOTAL</span>
          <button class="page-btn mono" :disabled="clips.length < 20" @click="clipsPage++; loadClips()">NEXT</button>
        </div>
      </section>

    </main>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

const MODELS       = ['AIRCRAFT', 'VEHICLE', 'PERSONNEL', 'GENERAL']
const CLIP_STATUSES = ['', 'PENDING', 'DOWNLOADING', 'DOWNLOADED', 'ANNOTATED', 'ERROR']

const MODEL_COLORS = {
  AIRCRAFT:  'oklch(0.62 0.16 220deg)',
  VEHICLE:   'oklch(0.60 0.20 25deg)',
  PERSONNEL: 'oklch(0.60 0.18 145deg)',
  GENERAL:   'oklch(0.65 0.18 55deg)',
}

const runs        = ref([])
const runsLoading = ref(false)
const runsPage    = ref(1)

const clips        = ref([])
const clipsLoading = ref(false)
const clipsPage    = ref(1)
const clipsTotal   = ref(0)
const clipStatus   = ref('')

const launching = ref(null)
const trainMsg  = ref('')

function token() { return localStorage.getItem('token') }

function authHeaders() {
  return { 'Authorization': `Bearer ${token()}`, 'Content-Type': 'application/json' }
}

async function apiFetch(url, opts = {}) {
  const res = await fetch(url, { ...opts, headers: authHeaders() })
  if (res.status === 401) { router.push('/admin/login'); throw new Error('401') }
  return res
}

async function loadRuns() {
  runsLoading.value = true
  try {
    const res = await apiFetch(`/api/admin/training-runs?page=${runsPage.value}&per_page=20`)
    const data = await res.json()
    runs.value = data.items
  } finally {
    runsLoading.value = false
  }
}

async function loadClips() {
  clipsLoading.value = true
  try {
    const statusQ = clipStatus.value ? `&status=${clipStatus.value}` : ''
    const res = await apiFetch(`/api/admin/clips?page=${clipsPage.value}&per_page=20${statusQ}`)
    const data = await res.json()
    clips.value = data.items
    clipsTotal.value = data.total
  } finally {
    clipsLoading.value = false
  }
}

async function queueTrain(modelType, stage) {
  launching.value = modelType
  trainMsg.value = ''
  try {
    const res = await apiFetch('/api/admin/train', {
      method: 'POST',
      body: JSON.stringify({ model_type: modelType, stage }),
    })
    if (res.ok) {
      trainMsg.value = `QUEUED: ${modelType} ${stage}`
      await loadRuns()
    } else {
      const d = await res.json()
      trainMsg.value = `ERROR: ${d.detail ?? res.status}`
    }
  } finally {
    launching.value = null
    setTimeout(() => { trainMsg.value = '' }, 4000)
  }
}

function latestRun(modelType) {
  return runs.value.find(r => r.model_type === modelType)
}

function modelColor(m) { return MODEL_COLORS[m] ?? 'var(--fg-1)' }

function logout() {
  localStorage.removeItem('token')
  router.push('/admin/login')
}

function fmtDate(iso) {
  if (!iso) return '—'
  return new Date(iso).toISOString().slice(0, 16).replace('T', ' ')
}

function fmtDur(s) {
  const m = Math.floor(s / 60), sec = s % 60
  return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`
}

onMounted(() => { loadRuns(); loadClips() })
</script>

<style scoped>
.panel-root {
  height: 100vh;
  overflow: hidden;
  background: var(--bg-0);
  color: var(--fg-0);
  display: flex;
  flex-direction: column;
}

.panel-header {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 32px;
  height: 48px;
  background: rgba(8,10,11,0.95);
  border-bottom: 1px solid var(--fg-3);
}

.panel-main {
  flex: 1;
  overflow: hidden;
  padding: 16px 32px 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.panel-section {
  padding: clamp(30px, 4vw, 50px) 0;
}

.panel-section-title {
  font-size: 10px;
  letter-spacing: 0.25em;
  color: var(--amber);
  margin-bottom: 8px;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1px;
  background: var(--fg-3);
  border: 1px solid var(--fg-3);
}

.model-card {
  background: var(--bg-1);
  padding: 10px 14px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.model-card[data-model="AIRCRAFT"] { --cat-color: oklch(0.62 0.16 220deg); }
.model-card[data-model="VEHICLE"]  { --cat-color: oklch(0.60 0.20 25deg); }
.model-card[data-model="PERSONNEL"]{ --cat-color: oklch(0.60 0.18 145deg); }
.model-card[data-model="GENERAL"]  { --cat-color: oklch(0.65 0.18 55deg); }

.model-card-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 2px;
}

.train-btn {
  flex: 1;
  padding: 6px 10px;
  font-size: 10px;
  letter-spacing: 0.12em;
  background: transparent;
  border: 1px solid var(--fg-3);
  color: var(--fg-1);
  cursor: pointer;
  transition: all 0.2s;
}
.train-btn:hover:not(:disabled) {
  border-color: var(--amber-border);
  color: var(--amber);
}
.train-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.run-status {
  font-size: 9px;
  letter-spacing: 0.15em;
  padding: 2px 6px;
  border: 1px solid currentColor;
}
.run-status.done    { color: var(--green); }
.run-status.running { color: var(--amber); }
.run-status.queued  { color: var(--fg-2); }
.run-status.error   { color: var(--red); }
.run-status.no\ runs { color: var(--fg-3); }

.panel-msg {
  margin-top: 16px;
  font-size: 11px;
  letter-spacing: 0.15em;
  color: var(--amber);
}

.panel-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.panel-table th {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.2em;
  color: var(--fg-3);
  text-align: left;
  padding: 4px 10px;
  border-bottom: 1px solid var(--fg-3);
}
.panel-table td {
  padding: 5px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  font-size: 12px;
  color: var(--fg-1);
}
.panel-table tr:hover td { background: var(--bg-1); }

.clip-status {
  font-size: 9px;
  letter-spacing: 0.12em;
  padding: 2px 6px;
  border: 1px solid currentColor;
}
.clip-status.annotated  { color: var(--green); }
.clip-status.downloaded { color: oklch(0.62 0.16 220deg); }
.clip-status.downloading { color: var(--amber); }
.clip-status.pending    { color: var(--fg-2); }
.clip-status.error      { color: var(--red); }

.panel-pagination {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 6px;
}
.page-btn {
  font-size: 10px;
  letter-spacing: 0.15em;
  padding: 6px 14px;
  background: transparent;
  border: 1px solid var(--fg-3);
  color: var(--fg-2);
  cursor: pointer;
  transition: all 0.2s;
}
.page-btn:hover:not(:disabled) { border-color: var(--fg-1); color: var(--fg-0); }
.page-btn:disabled { opacity: 0.3; cursor: not-allowed; }

.panel-loading { font-size: 11px; letter-spacing: 0.2em; color: var(--fg-3); padding: 24px 0; }

.status-filter-btn {
  font-size: 9px;
  letter-spacing: 0.15em;
  padding: 4px 10px;
  background: transparent;
  border: 1px solid var(--fg-3);
  color: var(--fg-2);
  cursor: pointer;
  transition: all 0.2s;
}
.status-filter-btn:hover { color: var(--fg-0); border-color: var(--fg-1); }
.status-filter-btn.active { border-color: var(--amber-border); color: var(--amber); }

/* reuse global logo mark */
.nav-logo-mark {
  width: 28px; height: 28px;
  border: 1.5px solid var(--amber);
  display: flex; align-items: center; justify-content: center;
  position: relative;
  flex-shrink: 0;
}
.nav-logo-mark::before {
  content: '';
  position: absolute;
  width: 8px; height: 8px;
  background: var(--amber);
  clip-path: polygon(50% 0%, 100% 100%, 0% 100%);
}
</style>
