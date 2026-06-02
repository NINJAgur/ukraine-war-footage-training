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
      <div class="det-index-note mono">det_class annotations · YOLOv8m conf≥0.25 iou=0.45 · {{ days }}d window</div>
    </div>

    <div v-if="loading" class="analytics-loading mono" style="position:relative;z-index:1">Loading...</div>

    <div v-else class="analytics-grid" style="position:relative;z-index:1">

      <!-- RADAR: Model performance profile -->
      <div class="chart-card">
        <div class="chart-label mono">Model performance radar — mAP50 · images · finetune cycles</div>
        <canvas ref="radarChart"></canvas>
      </div>

      <!-- SCATTER: Training data efficiency -->
      <div class="chart-card">
        <div class="chart-label mono">Training efficiency scatter — images vs mAP50 · all runs</div>
        <canvas ref="scatterChart"></canvas>
      </div>

      <!-- BAR: Inference throughput -->
      <div class="chart-card chart-wide">
        <div class="chart-label mono">Inference throughput — annotated clips / day · conf≥0.25</div>
        <canvas ref="clipsChart"></canvas>
      </div>

      <!-- STACKED BAR: Bbox volume per day -->
      <div class="chart-card chart-wide">
        <div class="chart-label mono">Bounding box volume / day · stacked by class · iou=0.45</div>
        <canvas ref="bboxChart"></canvas>
      </div>

      <!-- LINE: mAP50 flywheel -->
      <div class="chart-card chart-wide">
        <div class="chart-label mono">mAP50@0.5 per training run — baseline → scraped finetune · {{ days }}d</div>
        <canvas ref="mapChart"></canvas>
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

const radarChart   = ref(null)
const scatterChart = ref(null)
const clipsChart   = ref(null)
const bboxChart    = ref(null)
const mapChart     = ref(null)
let charts = {}

const C = {
  amber:      '#df6900',
  amberFaint: 'rgba(223,105,0,0.15)',
  aircraft:   'oklch(0.62 0.16 220deg)',
  vehicle:    'oklch(0.60 0.20 25deg)',
  personnel:  'oklch(0.60 0.18 145deg)',
  general:    'oklch(0.65 0.18 55deg)',
  grid:       'rgba(255,255,255,0.06)',
  tick:       'rgba(255,255,255,0.35)',
  radarGrid:  'rgba(255,255,255,0.1)',
}
const MODEL_COLORS = { AIRCRAFT: C.aircraft, VEHICLE: C.vehicle, PERSONNEL: C.personnel, GENERAL: C.general }
const CLASS_COLORS = { AIRCRAFT: C.aircraft, VEHICLE: C.vehicle, PERSONNEL: C.personnel, GENERAL: C.general }

const detectionIndex = computed(() => {
  if (!data.value) return [
    { label: 'AIRCRAFT',  count: '—', color: C.aircraft  },
    { label: 'VEHICLE',   count: '—', color: C.vehicle   },
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

const BASE_SCALE = {
  grid: { color: C.grid },
  ticks: { color: C.tick, font: { family: 'IBM Plex Mono', size: 10 } },
}

function destroyAll() { Object.values(charts).forEach(c => c?.destroy()); charts = {} }

function buildCharts() {
  destroyAll()
  const d = data.value
  if (!d) return

  // ── RADAR: model performance profile ──────────────────────────────
  if (radarChart.value) {
    // Build best-run stats per model
    const models = ['AIRCRAFT', 'VEHICLE', 'PERSONNEL', 'GENERAL']
    const best = {}
    const finetuneCounts = {}
    const MAX_IMAGES = 144466 // GENERAL baseline
    for (const r of d.training_scatter) {
      const m = r.model
      if (!best[m] || r.map50 > best[m].map50) best[m] = r
      if (r.stage === 'FINETUNE') finetuneCounts[m] = (finetuneCounts[m] || 0) + 1
    }
    const radarDatasets = models.filter(m => best[m]).map(m => {
      const b = best[m]
      return {
        label: m,
        data: [
          Math.round(b.map50 * 100),                        // mAP50 (%)
          Math.round((b.images / MAX_IMAGES) * 100),        // training data %
          Math.round((finetuneCounts[m] || 0) / 3 * 100),  // finetune cycles (max=3)
          b.precision != null ? Math.round(b.precision * 100) : 0,  // precision
          b.recall    != null ? Math.round(b.recall    * 100) : 0,  // recall
        ],
        borderColor: MODEL_COLORS[m],
        backgroundColor: MODEL_COLORS[m].replace('oklch', 'oklch').includes('oklch')
          ? `color-mix(in oklch, ${MODEL_COLORS[m]} 20%, transparent)`
          : 'rgba(255,255,255,0.05)',
        borderWidth: 2, pointRadius: 4,
      }
    })
    charts.radar = new Chart(radarChart.value, {
      type: 'radar',
      data: {
        labels: ['mAP50', 'Data size', 'Finetunes', 'Precision', 'Recall'],
        datasets: radarDatasets,
      },
      options: {
        responsive: true, maintainAspectRatio: true, aspectRatio: 1.2,
        scales: {
          r: {
            min: 0, max: 100,
            grid: { color: C.radarGrid },
            pointLabels: { color: C.tick, font: { family: 'IBM Plex Mono', size: 10 } },
            ticks: { color: 'transparent', backdropColor: 'transparent', stepSize: 25 },
          },
        },
        plugins: {
          legend: { display: true, position: 'bottom', labels: { color: C.tick, font: { family: 'IBM Plex Mono', size: 9 }, boxWidth: 10, padding: 12 } },
        },
      },
    })
  }

  // ── SCATTER: training data efficiency ─────────────────────────────
  if (scatterChart.value && d.training_scatter.length) {
    const models = ['AIRCRAFT', 'VEHICLE', 'PERSONNEL', 'GENERAL']
    charts.scatter = new Chart(scatterChart.value, {
      type: 'scatter',
      data: {
        datasets: models.map(m => ({
          label: m,
          data: d.training_scatter.filter(r => r.model === m && r.images > 0)
            .map(r => ({ x: r.images, y: r.map50, stage: r.stage, run: r.run_id })),
          backgroundColor: MODEL_COLORS[m],
          pointRadius: r => r.raw?.stage === 'FINETUNE' ? 7 : 5,
          pointStyle: r => r.raw?.stage === 'FINETUNE' ? 'triangle' : 'circle',
        })),
      },
      options: {
        responsive: true, maintainAspectRatio: true, aspectRatio: 1.2,
        scales: {
          x: { ...BASE_SCALE, title: { display: true, text: 'training images', color: C.tick, font: { family: 'IBM Plex Mono', size: 9 } } },
          y: { ...BASE_SCALE, min: 0.3, max: 1.0, title: { display: true, text: 'mAP50', color: C.tick, font: { family: 'IBM Plex Mono', size: 9 } } },
        },
        plugins: {
          legend: { display: true, position: 'bottom', labels: { color: C.tick, font: { family: 'IBM Plex Mono', size: 9 }, boxWidth: 10, padding: 12 } },
          tooltip: { callbacks: { label: ctx => `${ctx.dataset.label} run#${ctx.raw.run} (${ctx.raw.stage}) — ${ctx.raw.y} mAP50 · ${ctx.raw.x.toLocaleString()} imgs` } },
        },
      },
    })
  }

  // ── BAR: clips per day ────────────────────────────────────────────
  if (clipsChart.value) {
    charts.clips = new Chart(clipsChart.value, {
      type: 'bar',
      data: {
        labels: d.clips_per_day.map(r => r.date.slice(5)),
        datasets: [{ data: d.clips_per_day.map(r => r.count), backgroundColor: C.amberFaint, borderColor: C.amber, borderWidth: 1.5 }],
      },
      options: {
        responsive: true, maintainAspectRatio: true, aspectRatio: 3,
        plugins: { legend: { display: false } },
        scales: { x: BASE_SCALE, y: { ...BASE_SCALE, beginAtZero: true } },
      },
    })
  }

  // ── STACKED BAR: bbox volume ──────────────────────────────────────
  if (bboxChart.value) {
    const boxes = d.detection_boxes_per_day || []
    charts.bbox = new Chart(bboxChart.value, {
      type: 'bar',
      data: {
        labels: boxes.map(r => r.date.slice(5)),
        datasets: [
          { label: 'AIRCRAFT',  data: boxes.map(r => r.aircraft),  backgroundColor: C.aircraft,  borderWidth: 0 },
          { label: 'VEHICLE',   data: boxes.map(r => r.vehicle),   backgroundColor: C.vehicle,   borderWidth: 0 },
          { label: 'PERSONNEL', data: boxes.map(r => r.personnel), backgroundColor: C.personnel, borderWidth: 0 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: true, aspectRatio: 3,
        scales: {
          x: { ...BASE_SCALE, stacked: true },
          y: { ...BASE_SCALE, stacked: true, beginAtZero: true },
        },
        plugins: {
          legend: { display: true, position: 'top', labels: { color: C.tick, font: { family: 'IBM Plex Mono', size: 9 }, boxWidth: 10, padding: 12 } },
        },
      },
    })
  }

  // ── LINE: mAP50 flywheel ──────────────────────────────────────────
  if (mapChart.value && d.map50_timeline.length) {
    const cutoff = new Date(); cutoff.setDate(cutoff.getDate() - days.value)
    const timeline = d.map50_timeline.filter(r => new Date(r.date) >= cutoff)
    const src = timeline.length ? timeline : d.map50_timeline
    const models = [...new Set(src.map(r => r.model))]
    charts.map = new Chart(mapChart.value, {
      type: 'line',
      data: {
        datasets: models.map(m => ({
          label: m,
          data: src.filter(r => r.model === m).map(r => ({ x: r.date.slice(0,10), y: r.map50 })),
          borderColor: MODEL_COLORS[m] || C.amber,
          backgroundColor: 'transparent',
          pointRadius: r => r.raw?.stage === 'FINETUNE' ? 6 : 4,
          pointStyle: r => r.raw?.stage === 'FINETUNE' ? 'triangle' : 'circle',
          pointBackgroundColor: MODEL_COLORS[m] || C.amber,
          borderWidth: 2, tension: 0.2,
        })),
      },
      options: {
        responsive: true, maintainAspectRatio: true, aspectRatio: 4,
        scales: {
          x: { ...BASE_SCALE, type: 'category' },
          y: { ...BASE_SCALE, min: 0.3, max: 1.0 },
        },
        plugins: {
          legend: { display: true, position: 'top', labels: { color: C.tick, font: { family: 'IBM Plex Mono', size: 9 }, boxWidth: 10, padding: 16 } },
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

.det-index { display: flex; align-items: flex-end; gap: 40px; flex-wrap: wrap; padding: 32px clamp(20px, 5vw, 80px); border-bottom: 1px solid var(--fg-3); }
.det-index-item { display: flex; flex-direction: column; gap: 4px; }
.det-index-num { font-family: var(--font-mono); font-size: clamp(32px, 5vw, 56px); font-weight: 700; line-height: 1; }
.det-index-label { font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.2em; color: var(--fg-3); }
.det-index-note { font-family: var(--font-mono); font-size: 10px; color: var(--fg-3); letter-spacing: 0.06em; line-height: 1.6; margin-left: auto; align-self: flex-end; max-width: 280px; text-align: right; }

.analytics-loading { color: var(--fg-3); font-size: 13px; padding: 40px clamp(20px,5vw,80px); }

.analytics-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2px;
  padding: 2px;
}
.chart-card { background: var(--bg-2); padding: 24px; }
.chart-wide { grid-column: span 2; }
.chart-label { font-size: 10px; color: var(--fg-3); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 16px; }

@media (max-width: 768px) {
  .analytics-grid { grid-template-columns: 1fr; }
  .chart-wide { grid-column: span 1; }
  .det-index { gap: 24px; }
  .det-index-note { margin-left: 0; text-align: left; }
}
</style>
