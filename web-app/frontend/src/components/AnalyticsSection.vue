<template>
  <section class="analytics" id="analytics">
    <div class="section-header">
      <div>
        <div class="section-tag">Data</div>
        <h2 class="section-title">Pipeline analytics</h2>
      </div>
      <div class="analytics-toggle mono">
        <button :class="{ active: days === 7 }"  @click="setDays(7)">7D</button>
        <button :class="{ active: days === 30 }" @click="setDays(30)">30D</button>
        <button :class="{ active: days === 90 }" @click="setDays(90)">90D</button>
      </div>
    </div>

    <div v-if="loading" class="analytics-loading mono">Loading...</div>

    <div v-else class="analytics-grid">

      <!-- Clips over time -->
      <div class="chart-card chart-wide">
        <div class="chart-title mono">Clips annotated per day</div>
        <canvas ref="clipsChart"></canvas>
      </div>

      <!-- Detection breakdown -->
      <div class="chart-card">
        <div class="chart-title mono">Detection breakdown</div>
        <canvas ref="breakdownChart"></canvas>
      </div>

      <!-- mAP50 timeline -->
      <div class="chart-card chart-wide">
        <div class="chart-title mono">Model performance timeline (mAP50)</div>
        <canvas ref="mapChart"></canvas>
      </div>

      <!-- By source -->
      <div class="chart-card">
        <div class="chart-title mono">Clips by source</div>
        <canvas ref="sourceChart"></canvas>
      </div>

    </div>
  </section>
</template>

<script setup>
import { ref, onMounted, watch, nextTick } from 'vue'
import { Chart, registerables } from 'chart.js'

Chart.register(...registerables)

const days = ref(30)
const loading = ref(true)
const data = ref(null)

const clipsChart    = ref(null)
const breakdownChart = ref(null)
const mapChart      = ref(null)
const sourceChart   = ref(null)

let charts = {}

const AMBER     = 'rgba(223, 105, 0, 0.85)'
const AMBER_BG  = 'rgba(223, 105, 0, 0.15)'
const AIRCRAFT  = 'rgba(110, 165, 255, 0.85)'
const VEHICLE   = 'rgba(223, 105, 0, 0.85)'
const PERSONNEL = 'rgba(74, 222, 128, 0.85)'
const GENERAL   = 'rgba(160, 160, 160, 0.85)'
const GRID      = 'rgba(255,255,255,0.06)'
const TICK      = 'rgba(255,255,255,0.35)'

const BASE_OPTS = {
  responsive: true,
  maintainAspectRatio: true,
  plugins: { legend: { display: false } },
  scales: {
    x: { grid: { color: GRID }, ticks: { color: TICK, font: { family: 'IBM Plex Mono', size: 10 } } },
    y: { grid: { color: GRID }, ticks: { color: TICK, font: { family: 'IBM Plex Mono', size: 10 } }, beginAtZero: true },
  },
}

function destroyAll() {
  Object.values(charts).forEach(c => c?.destroy())
  charts = {}
}

function buildCharts() {
  destroyAll()
  const d = data.value
  if (!d) return

  // Clips per day
  if (clipsChart.value) {
    charts.clips = new Chart(clipsChart.value, {
      type: 'bar',
      data: {
        labels: d.clips_per_day.map(r => r.date.slice(5)),
        datasets: [{ data: d.clips_per_day.map(r => r.count), backgroundColor: AMBER_BG, borderColor: AMBER, borderWidth: 1 }],
      },
      options: { ...BASE_OPTS, aspectRatio: 3 },
    })
  }

  // Detection breakdown (doughnut)
  const CLASS_COLORS = { AIRCRAFT: AIRCRAFT, VEHICLE: VEHICLE, PERSONNEL: PERSONNEL, GENERAL: GENERAL }
  if (breakdownChart.value && d.detection_breakdown.length) {
    charts.breakdown = new Chart(breakdownChart.value, {
      type: 'doughnut',
      data: {
        labels: d.detection_breakdown.map(r => r.class),
        datasets: [{
          data: d.detection_breakdown.map(r => r.count),
          backgroundColor: d.detection_breakdown.map(r => CLASS_COLORS[r.class] || GENERAL),
          borderWidth: 0,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: true,
        plugins: {
          legend: { display: true, position: 'bottom', labels: { color: TICK, font: { family: 'IBM Plex Mono', size: 10 }, padding: 12 } },
        },
        cutout: '60%',
      },
    })
  }

  // mAP50 timeline (multi-line per model)
  if (mapChart.value && d.map50_timeline.length) {
    const models = [...new Set(d.map50_timeline.map(r => r.model))]
    const modelColors = { AIRCRAFT: AIRCRAFT, VEHICLE: VEHICLE, PERSONNEL: PERSONNEL, GENERAL: GENERAL }
    charts.map = new Chart(mapChart.value, {
      type: 'line',
      data: {
        datasets: models.map(m => ({
          label: m,
          data: d.map50_timeline.filter(r => r.model === m).map(r => ({ x: r.date.slice(0,10), y: r.map50 })),
          borderColor: modelColors[m] || AMBER,
          backgroundColor: 'transparent',
          pointRadius: 5, pointHoverRadius: 7,
          borderWidth: 2, tension: 0.2,
        })),
      },
      options: {
        ...BASE_OPTS,
        aspectRatio: 3,
        scales: {
          ...BASE_OPTS.scales,
          x: { ...BASE_OPTS.scales.x, type: 'category' },
          y: { ...BASE_OPTS.scales.y, min: 0.3, max: 1.0 },
        },
        plugins: {
          legend: { display: true, position: 'top', labels: { color: TICK, font: { family: 'IBM Plex Mono', size: 10 }, padding: 16 } },
        },
      },
    })
  }

  // By source (bar)
  if (sourceChart.value && d.by_source.length) {
    charts.source = new Chart(sourceChart.value, {
      type: 'bar',
      data: {
        labels: d.by_source.map(r => r.source.toUpperCase()),
        datasets: [{ data: d.by_source.map(r => r.count), backgroundColor: AMBER_BG, borderColor: AMBER, borderWidth: 1 }],
      },
      options: { ...BASE_OPTS, indexAxis: 'y', aspectRatio: 2 },
    })
  }
}

async function fetchAndRender() {
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
watch(days, fetchAndRender)
onMounted(fetchAndRender)
</script>

<style scoped>
.analytics { background: var(--bg-0); border-top: 1px solid var(--fg-3); position: relative; }
.section-header { display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 16px; }
.analytics-toggle { display: flex; gap: 2px; }
.analytics-toggle button {
  font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.12em;
  padding: 6px 14px; border: 1px solid var(--fg-3); background: none;
  color: var(--fg-2); cursor: pointer; transition: all 0.15s;
}
.analytics-toggle button.active { border-color: var(--amber-border); color: var(--amber); }
.analytics-toggle button:hover:not(.active) { border-color: var(--fg-1); color: var(--fg-0); }
.analytics-loading { color: var(--fg-3); font-size: 13px; padding: 40px 0; }
.analytics-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 2px;
  margin-top: 2px;
}
.chart-card { background: var(--bg-1); border: 1px solid var(--fg-3); padding: 24px; }
.chart-wide { grid-column: span 1; }
.chart-title { font-size: 10px; color: var(--fg-3); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 20px; }
@media (max-width: 768px) {
  .analytics-grid { grid-template-columns: 1fr; }
}
</style>
