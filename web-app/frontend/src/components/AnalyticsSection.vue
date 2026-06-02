<template>
  <section class="analytics" id="analytics">
    <div class="radar-bg"><RadarCanvas :opacity="0.2" color="194, 120, 40" /></div>

    <!-- Header + toggle -->
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

    <!-- 2×2 general charts -->
    <div v-else class="analytics-grid" style="position:relative;z-index:1">
      <div class="chart-card">
        <div class="chart-label mono">Model performance radar</div>
        <canvas ref="radarChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-label mono">Training efficiency — images vs mAP50</div>
        <canvas ref="scatterChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-label mono">Inference throughput — clips / day</div>
        <canvas ref="clipsChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-label mono">Bounding box volume / day · stacked by class</div>
        <canvas ref="bboxChart"></canvas>
      </div>
    </div>

    <!-- Model drill-down selector -->
    <div class="model-drill-header" style="position:relative;z-index:1">
      <div class="model-drill-label mono">Model drill-down</div>
      <div class="model-drill-tabs">
        <button
          v-for="m in availableModels" :key="m"
          :class="['model-drill-tab mono', `tab-${m.toLowerCase()}`, { active: selectedModel === m }]"
          @click="selectModel(m)"
        >{{ m }}</button>
      </div>
    </div>

    <!-- Expanded per-model charts -->
    <transition name="drill-expand">
      <div v-if="selectedModel" class="model-drill-grid" style="position:relative;z-index:1">
        <div class="chart-card">
          <div class="chart-label mono">mAP50 per epoch — {{ selectedModel }}</div>
          <canvas :ref="el => drillCanvases.map50 = el"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-label mono">Precision / Recall per epoch — {{ selectedModel }}</div>
          <canvas :ref="el => drillCanvases.pr = el"></canvas>
        </div>
        <div class="chart-card chart-wide-drill">
          <div class="chart-label mono">Validation loss curves — {{ selectedModel }} · box · cls · dfl</div>
          <canvas :ref="el => drillCanvases.loss = el"></canvas>
        </div>
      </div>
    </transition>
  </section>
</template>

<script setup>
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { Chart, registerables } from 'chart.js'
import RadarCanvas from './RadarCanvas.vue'

Chart.register(...registerables)

const days         = ref(30)
const loading      = ref(true)
const data         = ref(null)
const epochData    = ref([])
const selectedModel = ref(null)

const radarChart   = ref(null)
const scatterChart = ref(null)
const clipsChart   = ref(null)
const bboxChart    = ref(null)
const drillCanvases = ref({ map50: null, pr: null, loss: null })

let charts = {}
let drillCharts = {}

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

const availableModels = computed(() => [...new Set(epochData.value.map(r => r.model))].filter(Boolean).sort())

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
function destroyDrill() { Object.values(drillCharts).forEach(c => c?.destroy()); drillCharts = {} }

function buildGeneralCharts() {
  destroyAll()
  const d = data.value
  if (!d) return

  // RADAR
  if (radarChart.value) {
    const models = ['AIRCRAFT', 'VEHICLE', 'PERSONNEL', 'GENERAL']
    const best = {}; const fc = {}
    const MAX_IMAGES = 144466
    for (const r of d.training_scatter) {
      if (!best[r.model] || r.map50 > best[r.model].map50) best[r.model] = r
      if (r.stage === 'FINETUNE') fc[r.model] = (fc[r.model] || 0) + 1
    }
    charts.radar = new Chart(radarChart.value, {
      type: 'radar',
      data: {
        labels: ['mAP50', 'Data size', 'Finetunes', 'Precision', 'Recall'],
        datasets: models.filter(m => best[m]).map(m => {
          const b = best[m]
          return {
            label: m,
            data: [
              Math.round(b.map50 * 100),
              Math.round((b.images / MAX_IMAGES) * 100),
              Math.round((fc[m] || 0) / 3 * 100),
              b.precision != null ? Math.round(b.precision * 100) : 0,
              b.recall != null ? Math.round(b.recall * 100) : 0,
            ],
            borderColor: MODEL_COLORS[m], borderWidth: 2, pointRadius: 3,
            backgroundColor: `color-mix(in srgb, ${MODEL_COLORS[m]} 15%, transparent)`,
          }
        }),
      },
      options: {
        responsive: true, maintainAspectRatio: true, aspectRatio: 1.1,
        scales: { r: { min: 0, max: 100, grid: { color: C.radarGrid }, pointLabels: { color: C.tick, font: { family: 'IBM Plex Mono', size: 9 } }, ticks: { color: 'transparent', backdropColor: 'transparent', stepSize: 25 } } },
        plugins: { legend: { display: true, position: 'bottom', labels: { color: C.tick, font: { family: 'IBM Plex Mono', size: 9 }, boxWidth: 8, padding: 10 } } },
      },
    })
  }

  // SCATTER
  if (scatterChart.value && d.training_scatter.length) {
    charts.scatter = new Chart(scatterChart.value, {
      type: 'scatter',
      data: {
        datasets: ['AIRCRAFT','VEHICLE','PERSONNEL','GENERAL'].map(m => ({
          label: m,
          data: d.training_scatter.filter(r => r.model === m && r.images > 0).map(r => ({ x: r.images, y: r.map50 })),
          backgroundColor: MODEL_COLORS[m], pointRadius: 6,
        })),
      },
      options: {
        responsive: true, maintainAspectRatio: true, aspectRatio: 1.1,
        scales: {
          x: { ...BASE_SCALE, title: { display: true, text: 'images', color: C.tick, font: { family: 'IBM Plex Mono', size: 9 } } },
          y: { ...BASE_SCALE, min: 0.3, max: 1.0, title: { display: true, text: 'mAP50', color: C.tick, font: { family: 'IBM Plex Mono', size: 9 } } },
        },
        plugins: { legend: { display: true, position: 'bottom', labels: { color: C.tick, font: { family: 'IBM Plex Mono', size: 9 }, boxWidth: 8, padding: 10 } } },
      },
    })
  }

  // CLIPS/DAY
  if (clipsChart.value) {
    charts.clips = new Chart(clipsChart.value, {
      type: 'bar',
      data: { labels: d.clips_per_day.map(r => r.date.slice(5)), datasets: [{ data: d.clips_per_day.map(r => r.count), backgroundColor: C.amberFaint, borderColor: C.amber, borderWidth: 1.5 }] },
      options: { responsive: true, maintainAspectRatio: true, aspectRatio: 1.5, plugins: { legend: { display: false } }, scales: { x: BASE_SCALE, y: { ...BASE_SCALE, beginAtZero: true } } },
    })
  }

  // BBOX VOLUME
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
        responsive: true, maintainAspectRatio: true, aspectRatio: 1.5,
        scales: { x: { ...BASE_SCALE, stacked: true }, y: { ...BASE_SCALE, stacked: true, beginAtZero: true } },
        plugins: { legend: { display: true, position: 'bottom', labels: { color: C.tick, font: { family: 'IBM Plex Mono', size: 9 }, boxWidth: 8, padding: 8 } } },
      },
    })
  }
}

async function buildDrillCharts(model) {
  destroyDrill()
  if (!model) return
  await nextTick()

  const runs = epochData.value.filter(r => r.model === model).sort((a, b) => a.run_id - b.run_id)
  if (!runs.length) return

  const styleFor = r => ({
    borderColor: MODEL_COLORS[model] || C.amber,
    borderDash: r.stage === 'BASELINE' ? [4,3] : [],
    borderWidth: 1.5, pointRadius: 2, tension: 0.3, backgroundColor: 'transparent',
  })

  // mAP50 per epoch
  if (drillCanvases.value.map50) {
    drillCharts.map50 = new Chart(drillCanvases.value.map50, {
      type: 'line',
      data: {
        datasets: runs.map(r => ({
          label: `run #${r.run_id} ${r.stage?.toLowerCase()}`,
          data: r.epochs.map((e, i) => ({ x: i + 1, y: e['metrics/mAP50(B)'] })),
          ...styleFor(r),
        })),
      },
      options: {
        responsive: true, maintainAspectRatio: true, aspectRatio: 2.2,
        scales: { x: { ...BASE_SCALE }, y: { ...BASE_SCALE, min: 0, max: 1 } },
        plugins: { legend: { display: true, position: 'top', labels: { color: C.tick, font: { family: 'IBM Plex Mono', size: 9 }, boxWidth: 8, padding: 10 } } },
      },
    })
  }

  // P/R per epoch
  if (drillCanvases.value.pr) {
    const ds = runs.flatMap(r => [
      { label: `P #${r.run_id}`, data: r.epochs.map((e,i) => ({ x: i+1, y: e['metrics/precision(B)'] })), borderColor: MODEL_COLORS[model], borderDash: [], borderWidth: 1.5, pointRadius: 2, tension: 0.3, backgroundColor: 'transparent' },
      { label: `R #${r.run_id}`, data: r.epochs.map((e,i) => ({ x: i+1, y: e['metrics/recall(B)'] })),    borderColor: MODEL_COLORS[model], borderDash: [4,3],  borderWidth: 1.5, pointRadius: 2, tension: 0.3, backgroundColor: 'transparent' },
    ])
    drillCharts.pr = new Chart(drillCanvases.value.pr, {
      type: 'line',
      data: { datasets: ds },
      options: {
        responsive: true, maintainAspectRatio: true, aspectRatio: 2.2,
        scales: { x: { ...BASE_SCALE }, y: { ...BASE_SCALE, min: 0, max: 1 } },
        plugins: { legend: { display: true, position: 'top', labels: { color: C.tick, font: { family: 'IBM Plex Mono', size: 9 }, boxWidth: 8, padding: 8 } } },
      },
    })
  }

  // Val loss curves
  if (drillCanvases.value.loss) {
    const lossKeys = [['val/box_loss','box'], ['val/cls_loss','cls'], ['val/dfl_loss','dfl']]
    const lossColors = ['#df6900','oklch(0.62 0.16 220deg)','oklch(0.60 0.18 145deg)']
    const bestRun = runs.at(-1)
    if (bestRun) {
      drillCharts.loss = new Chart(drillCanvases.value.loss, {
        type: 'line',
        data: {
          datasets: lossKeys.map(([key, name], i) => ({
            label: name + '_loss',
            data: bestRun.epochs.map((e,j) => ({ x: j+1, y: e[key] })),
            borderColor: lossColors[i], borderWidth: 2, pointRadius: 2, tension: 0.3, backgroundColor: 'transparent',
          })),
        },
        options: {
          responsive: true, maintainAspectRatio: true, aspectRatio: 3.5,
          scales: { x: { ...BASE_SCALE, title: { display: true, text: 'epoch', color: C.tick, font: { family: 'IBM Plex Mono', size: 9 } } }, y: { ...BASE_SCALE } },
          plugins: { legend: { display: true, position: 'top', labels: { color: C.tick, font: { family: 'IBM Plex Mono', size: 9 }, boxWidth: 8, padding: 10 } } },
        },
      })
    }
  }
}

async function selectModel(m) {
  selectedModel.value = selectedModel.value === m ? null : m
  if (selectedModel.value) await buildDrillCharts(selectedModel.value)
  else destroyDrill()
}

function setDays(d) { days.value = d }

async function fetch_() {
  loading.value = true
  try {
    const [chartRes, epochRes] = await Promise.all([
      fetch(`/api/stats/charts?days=${days.value}`),
      fetch('/api/training/epoch-data'),
    ])
    if (chartRes.ok) data.value = await chartRes.json()
    if (epochRes.ok) epochData.value = await epochRes.json()
  } catch {}
  loading.value = false
  await nextTick()
  buildGeneralCharts()
}

watch(days, fetch_)
onMounted(fetch_)
</script>

<style scoped>
.analytics { background: var(--bg-1); border-top: 1px solid var(--fg-3); position: relative; overflow: hidden; }
.section-header { display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 16px; }
.analytics-toggle { display: flex; }
.analytics-toggle button { font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.12em; padding: 6px 16px; border: 1px solid var(--fg-3); background: none; color: var(--fg-3); cursor: pointer; transition: all 0.15s; margin-left: -1px; }
.analytics-toggle button.active { border-color: var(--amber); color: var(--amber); z-index:1; position:relative; }
.analytics-toggle button:hover:not(.active) { color: var(--fg-0); border-color: var(--fg-1); }

.det-index { display: flex; align-items: flex-end; gap: 40px; flex-wrap: wrap; padding: 32px clamp(20px, 5vw, 80px); border-bottom: 1px solid var(--fg-3); }
.det-index-item { display: flex; flex-direction: column; gap: 4px; }
.det-index-num { font-family: var(--font-mono); font-size: clamp(32px, 5vw, 56px); font-weight: 700; line-height: 1; }
.det-index-label { font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.2em; color: var(--fg-3); }
.det-index-note { font-family: var(--font-mono); font-size: 10px; color: var(--fg-3); letter-spacing: 0.06em; line-height: 1.6; margin-left: auto; align-self: flex-end; max-width: 280px; text-align: right; }

.analytics-loading { color: var(--fg-3); font-size: 13px; padding: 40px clamp(20px,5vw,80px); }

.analytics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px; padding: 2px; }
.chart-card { background: var(--bg-2); padding: 20px; }
.chart-label { font-size: 10px; color: var(--fg-3); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* Model drill-down */
.model-drill-header { display: flex; align-items: center; gap: 20px; padding: 20px clamp(20px,5vw,80px); border-top: 1px solid var(--fg-3); flex-wrap: wrap; }
.model-drill-label { font-size: 10px; color: var(--fg-3); letter-spacing: 0.18em; flex-shrink: 0; }
.model-drill-tabs { display: flex; gap: 2px; flex-wrap: wrap; }
.model-drill-tab { font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.15em; padding: 5px 16px; border: 1px solid var(--fg-3); background: none; color: var(--fg-3); cursor: pointer; transition: all 0.15s; }
.model-drill-tab:hover { color: var(--fg-0); border-color: var(--fg-1); }
.tab-aircraft.active  { color: var(--cat-color-aircraft);   border-color: var(--cat-color-aircraft); }
.tab-vehicle.active   { color: var(--cat-color-vehicles);   border-color: var(--cat-color-vehicles); }
.tab-personnel.active { color: var(--cat-color-personnel);  border-color: var(--cat-color-personnel); }
.tab-general.active   { color: var(--cat-color-generalist); border-color: var(--cat-color-generalist); }

.model-drill-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2px; padding: 2px; }
.chart-wide-drill { grid-column: span 2; }

/* Expand animation */
.drill-expand-enter-active, .drill-expand-leave-active { transition: max-height 0.4s ease, opacity 0.3s ease; overflow: hidden; }
.drill-expand-enter-from, .drill-expand-leave-to { max-height: 0; opacity: 0; }
.drill-expand-enter-to, .drill-expand-leave-from { max-height: 1200px; opacity: 1; }

@media (max-width: 900px) {
  .analytics-grid { grid-template-columns: 1fr 1fr; }
  .model-drill-grid { grid-template-columns: 1fr; }
  .chart-wide-drill { grid-column: span 1; }
}
@media (max-width: 480px) {
  .analytics-grid { grid-template-columns: 1fr; }
  .det-index { gap: 24px; }
  .det-index-note { margin-left: 0; text-align: left; }
}
</style>
