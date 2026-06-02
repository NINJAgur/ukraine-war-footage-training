<template>
  <div class="models-root">
    <AppNav />
    <main class="models-main">
      <div class="models-header">
        <div class="section-tag">Model Hub</div>
        <h1 class="models-title">Detection Models</h1>
        <p class="models-sub mono">
          YOLOv8m models trained on real conflict footage — continuously improving.<br>
          Download weights or query via the <router-link to="/api-docs" class="models-link">public API</router-link>.
        </p>
      </div>

      <div v-if="loading" class="models-loading mono">Loading...</div>

      <template v-else>
        <div v-for="modelName in modelOrder" :key="modelName" class="model-group">
          <div class="model-group-header">
            <span class="model-badge" :class="`badge-${modelName.toLowerCase()}`">{{ modelName }}</span>
            <span class="model-group-desc mono">{{ classDefs[modelName] }}</span>
          </div>

          <table class="versions-table">
            <thead>
              <tr>
                <th class="mono">Run</th>
                <th class="mono">Stage</th>
                <th class="mono">mAP50</th>
                <th class="mono">Images</th>
                <th class="mono">Date</th>
                <th class="mono">Download</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in byModel[modelName]" :key="r.run_id" :class="{ 'row-best': r.is_best }">
                <td class="mono">#{{ r.run_id }} <span v-if="r.is_best" class="best-tag">BEST</span></td>
                <td class="mono">{{ r.stage }}</td>
                <td class="mono map-val">{{ r.map50 != null ? r.map50.toFixed(3) : '—' }}</td>
                <td class="mono">{{ r.images ? fmt(r.images) : '—' }}</td>
                <td class="mono">{{ r.completed_at ? r.completed_at.slice(0,10) : '—' }}</td>
                <td>
                  <a v-if="r.download_url" :href="r.download_url" class="dl-btn" download>
                    best.pt
                  </a>
                  <span v-else class="mono" style="color:var(--fg-3)">—</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </template>

      <div class="models-note mono">
        Load weights: <code class="models-code">from ultralytics import YOLO; model = YOLO('best.pt')</code>
        &nbsp;·&nbsp; Classes: 0=AIRCRAFT · 1=VEHICLE · 2=PERSONNEL
        &nbsp;·&nbsp; <router-link to="/api-docs" class="models-link">API docs →</router-link>
      </div>
    </main>
    <SiteFooter />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import AppNav from '../components/AppNav.vue'
import SiteFooter from '../components/SiteFooter.vue'

const allRuns = ref([])
const loading = ref(true)
const modelOrder = ['AIRCRAFT', 'VEHICLE', 'PERSONNEL', 'GENERAL']

const classDefs = {
  AIRCRAFT:  'Drones, helicopters, fixed-wing, missiles',
  VEHICLE:   'Tanks, APCs, artillery, radar, ground vehicles',
  PERSONNEL: 'Soldiers, fighters, RPG/ATGM operators',
  GENERAL:   'All three classes combined',
}

const byModel = computed(() => {
  const map = {}
  for (const name of modelOrder) {
    map[name] = allRuns.value.filter(r => r.model === name)
  }
  return map
})

function fmt(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  if (n >= 1e3) return Math.round(n / 1e3) + 'K'
  return String(n)
}

onMounted(async () => {
  try {
    const res = await fetch('/api/models')
    if (res.ok) allRuns.value = await res.json()
  } catch {}
  loading.value = false
})
</script>

<style scoped>
.models-root { min-height: 100vh; display: flex; flex-direction: column; background: var(--bg-0); }
.models-main { flex: 1; padding: clamp(60px, 8vw, 100px) clamp(20px, 5vw, 80px) 60px; max-width: 1100px; margin: 0 auto; width: 100%; }
.models-header { margin-bottom: 48px; }
.models-title { font-size: clamp(28px, 4vw, 48px); font-weight: 700; letter-spacing: -0.02em; margin: 12px 0 16px; }
.models-sub { font-size: 13px; color: var(--fg-2); line-height: 1.7; }
.models-link { color: var(--amber); text-decoration: none; border-bottom: 1px solid var(--amber-border); }
.models-loading { color: var(--fg-3); font-size: 13px; }

.model-group { margin-bottom: 48px; }
.model-group-header { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; }
.model-badge { font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.18em; padding: 4px 12px; border: 1px solid; flex-shrink: 0; }
.badge-aircraft  { color: var(--cat-color-aircraft);   border-color: color-mix(in oklch, var(--cat-color-aircraft)  40%, transparent); }
.badge-vehicle   { color: var(--cat-color-vehicles);   border-color: color-mix(in oklch, var(--cat-color-vehicles)  40%, transparent); }
.badge-personnel { color: var(--cat-color-personnel);  border-color: color-mix(in oklch, var(--cat-color-personnel) 40%, transparent); }
.badge-general   { color: var(--cat-color-generalist); border-color: color-mix(in oklch, var(--cat-color-generalist) 40%, transparent); }
.model-group-desc { font-size: 12px; color: var(--fg-3); letter-spacing: 0.04em; }

.versions-table { width: 100%; border-collapse: collapse; }
.versions-table th {
  text-align: left; padding: 10px 16px; font-size: 10px; letter-spacing: 0.15em;
  color: var(--fg-3); border-bottom: 1px solid var(--fg-3); text-transform: uppercase;
}
.versions-table td { padding: 14px 16px; font-size: 13px; color: var(--fg-2); border-bottom: 1px solid rgba(255,255,255,0.04); }
.row-best td { background: rgba(223, 105, 0, 0.04); }
.map-val { color: var(--fg-0); font-weight: 600; }
.best-tag { font-size: 9px; letter-spacing: 0.12em; color: var(--amber); border: 1px solid var(--amber-border); padding: 1px 6px; margin-left: 6px; }

.dl-btn {
  font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.1em;
  color: var(--amber); border: 1px solid var(--amber-border);
  padding: 4px 10px; text-decoration: none; transition: background 0.15s;
}
.dl-btn:hover { background: var(--amber-glow); }

.models-note { font-size: 12px; color: var(--fg-3); border-top: 1px solid var(--fg-3); padding-top: 24px; line-height: 1.8; }
.models-code { background: var(--bg-2); padding: 2px 8px; font-family: var(--font-mono); font-size: 11px; color: var(--fg-1); }

@media (max-width: 600px) {
  .versions-table th:nth-child(4), .versions-table td:nth-child(4) { display: none; }
  .versions-table th:nth-child(3), .versions-table td:nth-child(3) { display: none; }
}
</style>
