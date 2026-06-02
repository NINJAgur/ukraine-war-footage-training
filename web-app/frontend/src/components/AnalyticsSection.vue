<template>
  <section class="analytics" id="analytics">
    <div class="radar-bg">
      <RadarCanvas :opacity="0.2" color="194, 120, 40" />
    </div>
    <div class="section-header" style="position:relative;z-index:1">
      <div>
        <div class="section-tag">Detection Index</div>
        <h2 class="section-title">Pipeline analytics</h2>
      </div>
      <div class="analytics-toggle mono">
        <button :class="{ active: days === 7 }"  @click="setDays(7)">7D</button>
        <button :class="{ active: days === 30 }" @click="setDays(30)">30D</button>
        <button :class="{ active: days === 90 }" @click="setDays(90)">90D</button>
      </div>
    </div>

    <!-- Detection frequency index -->
    <div class="det-index" style="position:relative;z-index:1">
      <div v-for="item in detectionIndex" :key="item.label" class="det-index-item">
        <div class="det-index-num" :style="{ color: item.color }">{{ item.count }}</div>
        <div class="det-index-label mono">{{ item.label }}</div>
      </div>
      <div class="det-index-note mono">detections in open-source footage · YOLOv8m conf≥0.25 · last {{ days }} days</div>
    </div>

    <div v-if="loading" class="analytics-loading mono" style="position:relative;z-index:1">Loading...</div>

    <div v-else class="analytics-grid" style="position:relative;z-index:1">
      <!-- Clips over time -->
      <div class="chart-card chart-wide">
        <div class="chart-label mono">Clips annotated · last {{ days }} days</div>
        <canvas ref="clipsChart"></canvas>
      </div>

      <!-- Class breakdown -->
      <div class="chart-card">
        <div class="chart-label mono">By detection class</div>
        <canvas ref="breakdownChart"></canvas>
      </div>

      <!-- Model evolution — always full history -->
      <div class="chart-card chart-wide">
        <div class="chart-label mono">Model mAP50 — self-improving flywheel</div>
        <canvas ref="mapChart" height="90"></canvas>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { Chart, registerables } from 'chart.js'
import RadarCanvas from './RadarCanvas.vue'

Chart.register(...registerables)

const days  = ref(30)
const loading = ref(true)
const data  = ref(null)

const clipsChart    = ref(null)
const breakdownChart = ref(null)
const mapChart      = ref(null)
let charts = {}

// Colour palette matching site
const C = {
  amber:      '#df6900',
  amberFaint: 'rgba(223,105,0,0.15)',
  aircraft:   'oklch(0.62 0.16 220deg)',
  vehicle:    'oklch(0.60 0.20 25deg)',
  personnel:  'oklch(0.60 0.18 145deg)',
  general:    'oklch(0.65 0.18 55deg)',
  grid:       'rgba(255,255,255,0.06)',
  tick:       'rgba(255,255,255,0.35)',
  bg:         '#111416',
}

const MODEL_COLORS = { AIRCRAFT: C.aircraft, VEHICLE: C.vehicle, PERSONNEL: C.personnel, GENERAL: C.general }
const CLASS_COLORS = { AIRCRAFT: C.aircraft, VEHICLE: C.vehicle, PERSONNEL: C.personnel, GENERAL: C.general }

const detectionIndex = computed(() => {
  if (!data.value) return [
    { label: 'AIRCRAFT', count: '—', color: C.aircraft },
    { label: 'VEHICLE',  count: '—', color: C.vehicle  },
    { label: 'PERSONNEL', count: '—', color: C.personnel },
  ]
  const bd = data.value.detection_breakdown
  const get = cls => (bd.find(r => r.class === cls)?.count ?? 0).toLocaleString()
  return [
    { label: 'AIRCRAFT',  count: get('AIRCRAFT'),  color: C.aircraft  },
    { label: 'VEHICLE',   count: get('VEHICLE'),   color: C.vehicle   },
    { label: 'PERSONNEL', count: get('PERSONNEL'), color: C.personnel },
  ]
})

const BASE = {
  responsive: true,
  maintainAspectRatio: true,
  aspectRatio: 3,
  plugins: { legend: { display: false } },
  scales: {
    x: { grid: { color: C.grid }, ticks: { color: C.tick, font: { family: 'IBM Plex Mono', size: 10 }, maxRotation: 0 } },
    y: { grid: { color: C.grid }, ticks: { color: C.tick, font: { family: 'IBM Plex Mono', size: 10 } }, beginAtZero: true },
  },
}

function destroyAll() { Object.values(charts).forEach(c => c?.destroy()); charts = {} }

function buildCharts() {
  destroyAll()
  const d = data.value
  if (!d) return

  if (clipsChart.value) {
    charts.clips = new Chart(clipsChart.value, {
      type: 'bar',
      data: {
        labels: d.clips_per_day.map(r => r.date.slice(5)),
        datasets: [{ data: d.clips_per_day.map(r => r.count), backgroundColor: C.amberFaint, borderColor: C.amber, borderWidth: 1.5 }],
      },
      options: { ...BASE },
    })
  }

  if (breakdownChart.value && d.detection_breakdown.length) {
    charts.breakdown = new Chart(breakdownChart.value, {
      type: 'bar',
      data: {
        labels: d.detection_breakdown.map(r => r.class),
        datasets: [{ data: d.detection_breakdown.map(r => r.count), backgroundColor: d.detection_breakdown.map(r => CLASS_COLORS[r.class] || C.amber), borderWidth: 0 }],
      },
      options: { ...BASE, aspectRatio: 1.8 },
    })
  }

  if (mapChart.value && d.map50_timeline.length) {
    const models = [...new Set(d.map50_timeline.map(r => r.model))]
    charts.map = new Chart(mapChart.value, {
      type: 'line',
      data: {
        datasets: models.map(m => ({
          label: m,
          data: d.map50_timeline.filter(r => r.model === m).map(r => ({ x: r.date.slice(0,10), y: r.map50 })),
          borderColor: MODEL_COLORS[m] || C.amber,
          backgroundColor: 'transparent',
          pointRadius: 5, pointHoverRadius: 7, pointBackgroundColor: MODEL_COLORS[m] || C.amber,
          borderWidth: 2, tension: 0.2,
        })),
      },
      options: {
        ...BASE,
        scales: {
          x: { ...BASE.scales.x, type: 'category' },
          y: { ...BASE.scales.y, min: 0.3, max: 1.0 },
        },
        plugins: {
          legend: { display: true, position: 'top', labels: { color: C.tick, font: { family: 'IBM Plex Mono', size: 10 }, boxWidth: 12, padding: 16 } },
        },
      },
    })
  }
}

async function fetch_() {
  loading.value = true
  try {
    const res = await fetch(`/api/stats/charts?days=${days.value}`)
    if (res.ok) data.value = await res.json()
  } catch {}
  loading.value = false
  await nextTick()
  buildCharts()
}

function setDays(d) { days.value = d }
watch(days, fetch_)
onMounted(fetch_)
</script>

<style scoped>
.analytics { background: var(--bg-1); border-top: 1px solid var(--fg-3); position: relative; overflow: hidden; }
.section-header { display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 16px; }
.analytics-toggle { display: flex; gap: 0; }
.analytics-toggle button {
  font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.12em;
  padding: 6px 16px; border: 1px solid var(--fg-3); background: none;
  color: var(--fg-3); cursor: pointer; transition: all 0.15s; margin-left: -1px;
}
.analytics-toggle button.active { border-color: var(--amber); color: var(--amber); z-index: 1; position: relative; }
.analytics-toggle button:hover:not(.active) { color: var(--fg-0); border-color: var(--fg-1); }

/* Detection index */
.det-index {
  display: flex; align-items: flex-end; gap: 40px; flex-wrap: wrap;
  padding: 32px clamp(20px, 5vw, 80px);
  border-bottom: 1px solid var(--fg-3);
}
.det-index-item { display: flex; flex-direction: column; gap: 4px; }
.det-index-num { font-family: var(--font-mono); font-size: clamp(32px, 5vw, 56px); font-weight: 700; line-height: 1; }
.det-index-label { font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.2em; color: var(--fg-3); }
.det-index-note { font-family: var(--font-mono); font-size: 10px; color: var(--fg-3); letter-spacing: 0.06em; line-height: 1.6; margin-left: auto; align-self: flex-end; max-width: 260px; text-align: right; }

.analytics-loading { color: var(--fg-3); font-size: 13px; padding: 40px clamp(20px,5vw,80px); }

.analytics-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 2px;
  padding: 2px;
}
.chart-card { background: var(--bg-2); padding: 24px; position: relative; }
.chart-wide { grid-column: 1; }
.chart-label { font-size: 10px; color: var(--fg-3); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 16px; }

@media (max-width: 768px) {
  .analytics-grid { grid-template-columns: 1fr; }
  .det-index { gap: 24px; }
  .det-index-note { margin-left: 0; text-align: left; }
}
</style>
