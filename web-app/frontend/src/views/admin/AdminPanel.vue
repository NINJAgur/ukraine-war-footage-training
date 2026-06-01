<template>
  <div class="panel-root">

    <!-- header -->
    <header class="panel-header">
      <a href="/" class="nav-logo" style="text-decoration:none;cursor:pointer">
        <img src="/favicon.svg" alt="" style="width:28px;height:28px;flex-shrink:0" />
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
              <div :class="['run-status', liveStatus(m) ?? latestRun(m)?.status?.toLowerCase()]" class="mono">
                {{ liveStatus(m)?.toUpperCase() ?? latestRun(m)?.status ?? 'NO RUNS' }}
              </div>
            </div>
            <div v-if="liveProgress(m)" class="train-progress">
              <div class="mono" style="font-size:10px;color:var(--amber);margin-bottom:4px">
                <span v-if="liveProgress(m).epochs != null">EPOCH {{ liveProgress(m).epoch - 1 }}/{{ liveProgress(m).epochs }}</span>
                <span v-else>INITIALIZING...</span>
                <span v-if="liveProgress(m).map50" style="color:var(--fg-1);margin-left:8px">mAP50 {{ liveProgress(m).map50.toFixed(3) }}</span>
              </div>
              <div class="progress-bar">
                <div class="progress-fill" :style="{ width: liveProgress(m).epochs ? ((liveProgress(m).epoch - 1) / liveProgress(m).epochs * 100) + '%' : '5%' }"></div>
              </div>
              <div class="mono" style="font-size:9px;color:var(--fg-3);margin-top:3px">
                <span v-if="liveProgress(m).box_loss">box {{ liveProgress(m).box_loss }}</span>
                <span v-if="liveProgress(m).cls_loss" style="margin-left:8px">cls {{ liveProgress(m).cls_loss }}</span>
              </div>
            </div>
            <div v-else-if="latestRun(m)?.map50 != null" class="mono" style="font-size:16px;font-weight:300;color:var(--fg-0);letter-spacing:-0.02em;margin:2px 0">
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

        <div v-if="trainMsg" :class="['panel-msg mono', trainMsg.startsWith('ERROR') ? 'panel-msg-error' : '']">{{ trainMsg }}</div>
      </section>

      <!-- ── TRAINING RUNS ── -->
      <section class="panel-section">
        <div class="panel-section-title mono">TRAINING RUNS</div>
        <div v-if="runsLoading" class="panel-loading mono">LOADING...</div>
        <div v-else class="table-scroll">
          <table class="panel-table">
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
        </div>
        <div class="panel-pagination">
          <button class="page-btn mono" :disabled="runsPage <= 1" @click="runsPage--; loadRuns()">PREV</button>
          <span class="mono dim" style="font-size:11px">PAGE {{ runsPage }}</span>
          <button class="page-btn mono" :disabled="runs.length < 20" @click="runsPage++; loadRuns()">NEXT</button>
        </div>
      </section>

      <!-- ── CLIPS ── -->
      <section class="panel-section">
        <div class="clip-section-header">
          <div class="panel-section-title mono" style="margin-bottom:0">CLIP ARCHIVE</div>
          <div class="clip-filter-row">
            <button
              v-for="s in CLIP_STATUSES"
              :key="s"
              :class="['status-filter-btn mono', { active: clipStatus === s }]"
              @click="clipStatus = s; clipsPage = 1; loadClips()"
            >{{ s || 'ALL' }}</button>
          </div>
        </div>
        <div v-if="clipsLoading" class="panel-loading mono">LOADING...</div>
        <div v-else class="table-scroll">
          <table class="panel-table">
            <thead>
              <tr><th>ID</th><th>TITLE</th><th>SOURCE</th><th>STATUS</th><th>DURATION</th><th>ADDED</th><th></th></tr>
            </thead>
            <tbody>
              <tr v-for="c in clips" :key="c.id">
                <td class="mono dim">#{{ c.id }}</td>
                <td style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{{ c.title ?? c.url }}</td>
                <td class="mono dim">{{ c.source }}</td>
                <td><span :class="['clip-status', c.status.toLowerCase()]" class="mono">{{ c.status }}</span></td>
                <td class="mono dim">{{ c.duration_seconds ? fmtDur(c.duration_seconds) : '—' }}</td>
                <td class="mono dim">{{ fmtDate(c.created_at) }}</td>
                <td style="display:flex;gap:6px;align-items:center">
                  <button class="preview-btn mono" @click.stop="openPreview(c)" title="Preview">&#9654;</button>
                  <button v-if="c.status === 'REVIEW'" class="approve-btn mono" @click="approveClip(c.id)">APPROVE</button>
                  <button v-if="c.status === 'REVIEW'" class="decline-btn mono" @click="declineClip(c.id)">DECLINE</button>
                </td>
              </tr>
              <tr v-if="!clips.length">
                <td colspan="7" class="mono dim" style="text-align:center;padding:10px">NO CLIPS FOUND</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="panel-pagination">
          <button class="page-btn mono" :disabled="clipsPage <= 1" @click="clipsPage--; loadClips()">PREV</button>
          <span class="mono dim" style="font-size:11px">PAGE {{ clipsPage }} · {{ clipsTotal }} TOTAL</span>
          <button class="page-btn mono" :disabled="clips.length < 20" @click="clipsPage++; loadClips()">NEXT</button>
        </div>
      </section>

    </main>

    <!-- ── CLIP PREVIEW MODAL ── -->
    <Teleport to="body">
      <div v-if="previewClip" class="preview-backdrop" @click="previewClip = null">
        <div class="preview-panel" @click.stop>
          <div class="preview-header">
            <div style="min-width:0;flex:1;overflow:hidden;margin-right:16px">
              <div class="mono" style="font-size:9px;letter-spacing:0.15em;color:var(--amber);margin-bottom:3px">{{ previewClip.source?.toUpperCase() }} · {{ previewClip.status }}</div>
              <div style="font-size:14px;font-weight:600;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{{ previewClip.title ?? previewClip.url_hash }}</div>
            </div>
            <button class="preview-close mono" @click="previewClip = null">[ CLOSE ]</button>
          </div>
          <div class="preview-body">
            <video
              v-if="previewClip.video_url"
              :src="previewClip.video_url"
              class="preview-video"
              controls autoplay muted playsinline
            />
            <div v-else class="preview-no-video">
              <div class="mono" style="font-size:10px;letter-spacing:0.2em;color:var(--fg-3);margin-bottom:12px">VIDEO NOT YET DOWNLOADED</div>
              <a :href="previewClip.url" target="_blank" rel="noopener" class="preview-url-link mono">
                &#8599; OPEN ORIGINAL SOURCE
              </a>
              <div v-if="previewClip.description" class="preview-desc">{{ previewClip.description }}</div>
            </div>
          </div>
          <div class="preview-meta">
            <div class="preview-meta-cell"><span class="mono dim">URL</span><a :href="previewClip.url" target="_blank" rel="noopener" class="preview-url-mini mono">{{ previewClip.url }}</a></div>
            <div class="preview-meta-cell"><span class="mono dim">ADDED</span><span class="mono">{{ fmtDate(previewClip.created_at) }}</span></div>
            <div v-if="previewClip.det_class" class="preview-meta-cell"><span class="mono dim">CLASS</span><span class="mono">{{ previewClip.det_class }}</span></div>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

const MODELS       = ['AIRCRAFT', 'VEHICLE', 'PERSONNEL', 'GENERAL']
const CLIP_STATUSES = ['', 'REVIEW', 'PENDING', 'DOWNLOADING', 'DOWNLOADED', 'ANNOTATED', 'ERROR']

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

const launching   = ref(null)
const trainMsg    = ref('')
const previewClip = ref(null)

// model → { status, epoch_progress } from WebSocket
const wsData = ref({})
// model → WebSocket instance
const wsSockets = {}

function liveStatus(m) { return wsData.value[m]?.status?.toLowerCase() ?? null }
function liveProgress(m) {
  const d = wsData.value[m]
  if (d?.status?.toLowerCase() !== 'running') return null
  return d.metrics?.epoch_progress ?? { epoch: 0, epochs: null, map50: null, box_loss: null, cls_loss: null }
}

function openWs(modelType, runId) {
  if (wsSockets[modelType]) { wsSockets[modelType].close(); delete wsSockets[modelType] }
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/ws/training/${runId}`)
  wsSockets[modelType] = ws
  ws.onmessage = (e) => {
    const d = JSON.parse(e.data)
    wsData.value[modelType] = d
    if (d.status?.toLowerCase() === 'done' || d.status?.toLowerCase() === 'error') {
      ws.close()
      delete wsSockets[modelType]
      loadRuns()
    }
  }
  ws.onerror = () => { delete wsSockets[modelType] }
}

onUnmounted(() => { Object.values(wsSockets).forEach(ws => ws.close()) })

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
    for (const r of runs.value) {
      if (r.status === 'RUNNING' && !wsSockets[r.model_type]) {
        openWs(r.model_type, r.id)
      }
    }
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
      const d = await res.json()
      trainMsg.value = `QUEUED: ${modelType} ${stage}`
      openWs(modelType, d.training_run_id)
      await loadRuns()
      setTimeout(() => { trainMsg.value = '' }, 4000)
    } else {
      const d = await res.json()
      trainMsg.value = `ERROR: ${d.detail ?? res.status}`
      setTimeout(() => { trainMsg.value = '' }, 10000)
    }
  } finally {
    launching.value = null
  }
}

function openPreview(clip) {
  previewClip.value = clip
}

async function approveClip(clipId) {
  await apiFetch(`/api/admin/clips/${clipId}/approve`, { method: 'POST' })
  await loadClips()
}

async function declineClip(clipId) {
  const res = await apiFetch(`/api/admin/clips/${clipId}`, { method: 'DELETE' })
  if (!res.ok) {
    const d = await res.json().catch(() => ({}))
    trainMsg.value = `DECLINE ERROR ${res.status}: ${d.detail ?? 'unknown'}`
    setTimeout(() => { trainMsg.value = '' }, 8000)
    return
  }
  await loadClips()
}

function latestRun(modelType) {
  const done = runs.value.find(r => r.model_type === modelType && r.status === 'DONE')
  return done ?? runs.value.find(r => r.model_type === modelType)
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
  overflow-y: auto;
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

.train-progress { margin: 4px 0; }
.progress-bar {
  height: 2px;
  background: var(--fg-3);
  width: 100%;
}
.progress-fill {
  height: 100%;
  background: var(--amber);
  transition: width 0.5s ease;
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
  letter-spacing: 0.1em;
  color: var(--amber);
}
.panel-msg-error {
  color: var(--red);
  border: 1px solid rgba(210,40,40,0.4);
  background: rgba(210,40,40,0.06);
  padding: 10px 14px;
  border-radius: 2px;
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
.clip-status.review     { color: var(--amber); }

.approve-btn {
  background: none;
  border: 1px solid var(--green);
  color: var(--green);
  font-size: 9px;
  letter-spacing: 0.12em;
  padding: 3px 8px;
  cursor: pointer;
  transition: background 0.15s;
}
.approve-btn:hover { background: rgba(34,197,94,0.1); }

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


.decline-btn {
  background: none;
  border: 1px solid var(--red);
  color: var(--red);
  font-size: 9px;
  letter-spacing: 0.12em;
  padding: 3px 8px;
  cursor: pointer;
  transition: background 0.15s;
}
.decline-btn:hover { background: rgba(210,40,40,0.1); }

.preview-btn {
  background: none;
  border: 1px solid var(--fg-3);
  color: var(--fg-2);
  font-size: 10px;
  padding: 2px 7px;
  cursor: pointer;
  transition: all 0.15s;
}
.preview-btn:hover { border-color: var(--amber-border); color: var(--amber); }

.preview-backdrop {
  position: fixed; inset: 0; z-index: 1000;
  background: rgba(0,0,0,0.75);
  display: flex; align-items: center; justify-content: center;
}
.preview-panel {
  background: var(--bg-1);
  border: 1px solid var(--fg-3);
  width: min(860px, 95vw);
  max-height: 90vh;
  display: flex; flex-direction: column;
  overflow: hidden;
}
.preview-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 18px;
  border-bottom: 1px solid var(--fg-3);
  flex-shrink: 0;
}
.preview-close {
  background: none; border: 1px solid var(--fg-3); color: var(--fg-2);
  font-size: 10px; letter-spacing: 0.15em; padding: 4px 10px; cursor: pointer;
  flex-shrink: 0;
}
.preview-close:hover { border-color: var(--fg-1); color: var(--fg-0); }
.preview-body { flex: 1; overflow: hidden; min-height: 0; }
.preview-video { width: 100%; max-height: 55vh; display: block; background: #000; }
.preview-no-video {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  min-height: 200px; padding: 32px;
  background: var(--bg-0);
}
.preview-url-link {
  font-size: 11px; letter-spacing: 0.1em;
  color: var(--amber); border: 1px solid var(--amber-border);
  padding: 8px 16px; text-decoration: none;
}
.preview-url-link:hover { background: rgba(217,119,6,0.1); }
.preview-desc { margin-top: 16px; font-size: 12px; color: var(--fg-2); max-width: 500px; text-align: center; }
.preview-meta {
  display: flex; flex-wrap: wrap; gap: 0;
  border-top: 1px solid var(--fg-3);
  flex-shrink: 0;
}
.preview-meta-cell {
  display: flex; flex-direction: column; gap: 2px;
  padding: 10px 18px;
  border-right: 1px solid var(--fg-3);
  min-width: 0; flex: 1;
}
.preview-meta-cell .mono.dim { font-size: 9px; letter-spacing: 0.15em; color: var(--fg-3); }
.preview-meta-cell .mono:not(.dim) { font-size: 11px; color: var(--fg-0); }
.preview-url-mini {
  font-size: 10px; color: var(--amber); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; text-decoration: none; max-width: 200px; display: block;
}
.preview-url-mini:hover { text-decoration: underline; }

.table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }

.clip-section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.clip-filter-row { display: flex; gap: 8px; }

@media (max-width: 768px) {
  .panel-header { padding: 0 16px; }
  .panel-header > div { gap: 12px; }
  .panel-main { padding: 12px 16px; }
  .clip-section-header { flex-wrap: wrap; row-gap: 8px; }
  .clip-filter-row { flex-wrap: wrap; row-gap: 6px; }
  /* preview panel: full width on mobile */
  .preview-panel { width: 98vw; max-height: 95vh; }
  .preview-video { max-height: 45vh; }
}

@media (max-width: 480px) {
  .model-grid { grid-template-columns: 1fr; }
}
</style>
