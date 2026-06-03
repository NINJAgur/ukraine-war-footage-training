<template>
  <section class="analytics" id="analytics">
    <div class="radar-bg"><RadarCanvas :opacity="0.2" color="194, 120, 40" /></div>

    <div class="section-header" style="position:relative;z-index:1">
      <div>
        <div class="section-tag">Detection Index</div>
        <h2 class="section-title">Pipeline analytics</h2>
      </div>
      <div class="analytics-toggle mono">
        <button :class="{ active: days===7  }" @click="setDays(7)">7D</button>
        <button :class="{ active: days===30 }" @click="setDays(30)">30D</button>
        <button :class="{ active: days===90 }" @click="setDays(90)">90D</button>
      </div>
    </div>

    <!-- Detection index headline -->
    <div class="det-index" style="position:relative;z-index:1">
      <div v-for="item in detectionIndex" :key="item.label" class="det-index-item">
        <div class="det-index-num" :style="{ color: item.color }">{{ item.count }}</div>
        <div class="det-index-label mono">{{ item.label }}</div>
      </div>
      <div class="det-index-note mono">det_class annotations · YOLOv8m conf≥0.25 iou=0.45 · {{ days }}d window</div>
    </div>

    <div v-if="loading" class="analytics-loading mono" style="position:relative;z-index:1">Loading...</div>

    <!-- 4 general charts in one row -->
    <div v-else class="analytics-grid" style="position:relative;z-index:1">
      <div class="chart-card">
        <div class="chart-label mono">Annotated clips / day</div>
        <canvas ref="clipsChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-label mono">Detection class split</div>
        <canvas ref="breakdownChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-label mono">mAP50 per training run</div>
        <canvas ref="mapChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-label mono">Model performance radar</div>
        <canvas ref="radarChart"></canvas>
      </div>
    </div>

    <!-- Per-run drill-down -->
    <div class="drill-header" style="position:relative;z-index:1">
      <span class="drill-label mono">Run drill-down</span>
      <select class="drill-select mono" @change="e => selectRun(+e.target.value || null)">
        <option value="">— Select a run —</option>
        <option v-for="r in availableRuns" :key="r.run_id" :value="r.run_id">
          #{{ r.run_id }} {{ r.model }} · {{ r.stage.toLowerCase() }} · mAP50={{ (r.map50||0).toFixed(3) }}
        </option>
      </select>
    </div>

    <transition name="drill-expand">
      <div v-if="selectedRun" class="drill-grid" style="position:relative;z-index:1">
        <!-- Row 1 -->
        <div class="chart-card">
          <div class="chart-label mono">mAP50 per epoch</div>
          <canvas :ref="el => dc.map50 = el"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-label mono">Precision &amp; Recall per epoch</div>
          <canvas :ref="el => dc.pr = el"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-label mono">Train loss per epoch</div>
          <canvas :ref="el => dc.trainLoss = el"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-label mono">Val loss per epoch</div>
          <canvas :ref="el => dc.valLoss = el"></canvas>
        </div>
        <!-- Row 2 -->
        <div class="chart-card">
          <div class="chart-label mono">Confusion matrix (val set)</div>
          <div class="cm-wrap" v-if="selectedRun?.confusion_matrix">
            <div class="cm-grid" :style="{ gridTemplateColumns: `repeat(${selectedRun.confusion_matrix.length}, 1fr)` }">
              <template v-for="(row, r) in selectedRun.confusion_matrix" :key="r">
                <div v-for="(val, c) in row" :key="c" class="cm-cell"
                  :style="{ background: `rgba(96,165,250,${Math.pow(val / (cmMax||1), 0.4)})` }"
                  :title="`${cmLabels[r]}→${cmLabels[c]}: ${val.toFixed(0)}`">
                  <span class="cm-val mono">{{ val > 0 ? val.toFixed(0) : '' }}</span>
                </div>
              </template>
            </div>
            <div class="cm-labels mono">
              <span v-for="l in cmLabels" :key="l" class="cm-label">{{ l }}</span>
            </div>
          </div>
          <div v-else class="chart-empty mono">No data yet</div>
        </div>
        <div class="chart-card">
          <div class="chart-label mono">BoxPR curve (P vs R)</div>
          <canvas :ref="el => dc.boxPR = el"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-label mono">Precision vs confidence</div>
          <canvas :ref="el => dc.boxP = el"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-label mono">Recall vs confidence</div>
          <canvas :ref="el => dc.boxR = el"></canvas>
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

const days = ref(30)
const loading = ref(true)
const data = ref(null)
const epochData = ref([])
const selectedRunId = ref(null)

const clipsChart    = ref(null)
const breakdownChart = ref(null)
const mapChart      = ref(null)
const radarChart    = ref(null)
const dc = ref({ map50:null, pr:null, trainLoss:null, valLoss:null, cm:null, boxPR:null, boxP:null, boxR:null })

let charts = {}
let drillCharts = {}

const C = {
  amber: '#df6900', amberFaint: 'rgba(223,105,0,0.15)',
  aircraft: 'oklch(0.62 0.16 220deg)', vehicle: 'oklch(0.60 0.20 25deg)',
  personnel: 'oklch(0.60 0.18 145deg)', general: 'oklch(0.65 0.18 55deg)',
  grid: 'rgba(255,255,255,0.06)', tick: 'rgba(255,255,255,0.35)',
}
const MC = { AIRCRAFT: C.aircraft, VEHICLE: C.vehicle, PERSONNEL: C.personnel, GENERAL: C.general }
const BS = { grid: { color: C.grid }, ticks: { color: C.tick, font: { family: 'IBM Plex Mono', size: 10 } } }

const detectionIndex = computed(() => {
  if (!data.value) return [
    { label: 'AIRCRAFT', count: '—', color: C.aircraft },
    { label: 'VEHICLE', count: '—', color: C.vehicle },
    { label: 'PERSONNEL', count: '—', color: C.personnel },
  ]
  const bd = data.value.detection_breakdown
  const get = cls => (bd.find(r => r.class === cls)?.count ?? 0).toLocaleString()
  return [
    { label: 'AIRCRAFT', count: get('AIRCRAFT'), color: C.aircraft },
    { label: 'VEHICLE', count: get('VEHICLE'), color: C.vehicle },
    { label: 'PERSONNEL', count: get('PERSONNEL'), color: C.personnel },
  ]
})

const cmMax = computed(() => {
  const cm = selectedRun.value?.confusion_matrix
  if (!cm) return 1
  return Math.max(...cm.flat())
})
const cmLabels = computed(() => {
  const nc = selectedRun.value?.confusion_matrix_nc || selectedRun.value?.confusion_matrix?.length || 0
  if (nc === 2) return ['aircraft', 'bg']
  if (nc === 3) return ['aircraft', 'vehicle', 'personnel']
  return Array.from({length: nc}, (_, i) => `c${i}`)
})

const availableRuns = computed(() =>
  epochData.value.filter(r => r.epochs?.length).map(r => ({
    ...r,
    map50: r.epochs ? Math.max(...r.epochs.map(e => e['metrics/mAP50(B)'] || 0)) : null,
  })).sort((a,b) => a.run_id - b.run_id)
)
const selectedRun = computed(() => epochData.value.find(r => r.run_id === selectedRunId.value))

function destroyAll() { Object.values(charts).forEach(c => c?.destroy()); charts = {} }
function destroyDrill() { Object.values(drillCharts).forEach(c => c?.destroy()); drillCharts = {} }

function buildGeneral() {
  destroyAll()
  const d = data.value
  if (!d) return

  // Clips/day
  if (clipsChart.value) {
    charts.clips = new Chart(clipsChart.value, {
      type: 'bar',
      data: { labels: d.clips_per_day.map(r => r.date.slice(5)), datasets: [{ data: d.clips_per_day.map(r => r.count), backgroundColor: C.amberFaint, borderColor: C.amber, borderWidth: 1.5 }] },
      options: { responsive:true, maintainAspectRatio:true, aspectRatio:1.4, plugins:{legend:{display:false}}, scales:{x:BS,y:{...BS,beginAtZero:true}} },
    })
  }

  // Class breakdown (doughnut)
  if (breakdownChart.value && d.detection_breakdown.length) {
    charts.breakdown = new Chart(breakdownChart.value, {
      type: 'doughnut',
      data: { labels: d.detection_breakdown.map(r => r.class), datasets: [{ data: d.detection_breakdown.map(r => r.count), backgroundColor: d.detection_breakdown.map(r => MC[r.class]||C.amber), borderWidth:0 }] },
      options: { responsive:true, maintainAspectRatio:true, aspectRatio:1.4, cutout:'55%', plugins:{legend:{display:true,position:'bottom',labels:{color:C.tick,font:{family:'IBM Plex Mono',size:9},boxWidth:8,padding:8}}} },
    })
  }

  // mAP50 timeline
  if (mapChart.value && d.map50_timeline.length) {
    const models = [...new Set(d.map50_timeline.map(r => r.model))]
    charts.map = new Chart(mapChart.value, {
      type: 'line',
      data: {
        datasets: models.map(m => ({
          label: m,
          data: d.map50_timeline.filter(r => r.model === m).map(r => ({ x: r.date.slice(0,10), y: r.map50 })),
          borderColor: MC[m]||C.amber, backgroundColor:'transparent',
          pointRadius:4, borderWidth:2, tension:0.2,
        })),
      },
      options: { responsive:true, maintainAspectRatio:true, aspectRatio:1.4, scales:{x:{...BS,type:'category'},y:{...BS,min:0.3,max:1.0}}, plugins:{legend:{display:true,position:'bottom',labels:{color:C.tick,font:{family:'IBM Plex Mono',size:9},boxWidth:8,padding:8}}} },
    })
  }

  // Radar
  if (radarChart.value && d.training_scatter.length) {
    const best = {}; const fc = {}; const MAX = 144466
    for (const r of d.training_scatter) {
      if (!best[r.model] || r.map50 > best[r.model].map50) best[r.model] = r
      if (r.stage === 'FINETUNE') fc[r.model] = (fc[r.model]||0) + 1
    }
    charts.radar = new Chart(radarChart.value, {
      type: 'radar',
      data: {
        labels: ['mAP50', 'Data size', 'Finetunes', 'Precision', 'Recall'],
        datasets: Object.entries(best).map(([m,b]) => ({
          label: m,
          data: [Math.round(b.map50*100), Math.round((b.images/MAX)*100), Math.round((fc[m]||0)/3*100), b.precision?Math.round(b.precision*100):0, b.recall?Math.round(b.recall*100):0],
          borderColor: MC[m], borderWidth:2, pointRadius:3,
          backgroundColor: `color-mix(in srgb, ${MC[m]} 15%, transparent)`,
        })),
      },
      options: { responsive:true, maintainAspectRatio:true, aspectRatio:1.4, scales:{r:{min:0,max:100,grid:{color:'rgba(255,255,255,0.1)'},pointLabels:{color:C.tick,font:{family:'IBM Plex Mono',size:9}},ticks:{color:'transparent',backdropColor:'transparent',stepSize:25}}}, plugins:{legend:{display:true,position:'bottom',labels:{color:C.tick,font:{family:'IBM Plex Mono',size:9},boxWidth:8,padding:8}}} },
    })
  }
}

async function buildDrill(runId) {
  destroyDrill()
  await nextTick()
  const run = epochData.value.find(r => r.run_id === runId)
  if (!run?.epochs?.length) return
  const epochs = run.epochs
  const xs = epochs.map((_,i) => i+1)
  const color = MC[run.model] || C.amber

  const lineOpts = (yLabel, min, max) => ({
    responsive:true, maintainAspectRatio:true, aspectRatio:1.3,
    scales:{ x:{...BS,title:{display:true,text:'epoch',color:C.tick,font:{family:'IBM Plex Mono',size:9}}}, y:{...BS,...(min!=null?{min,max}:{}),title:{display:true,text:yLabel,color:C.tick,font:{family:'IBM Plex Mono',size:9}}} },
    plugins:{legend:{display:true,position:'top',labels:{color:C.tick,font:{family:'IBM Plex Mono',size:9},boxWidth:8,padding:8}}},
  })

  // mAP50/epoch
  if (dc.value.map50) drillCharts.map50 = new Chart(dc.value.map50, {
    type:'line', data:{ labels:xs, datasets:[{ label:'mAP50', data:epochs.map(e=>e['metrics/mAP50(B)']), borderColor:color, backgroundColor:'transparent', pointRadius:3, borderWidth:2, tension:0.3 }] },
    options: lineOpts('mAP50',0,1),
  })

  // Precision + Recall
  if (dc.value.pr) drillCharts.pr = new Chart(dc.value.pr, {
    type:'line', data:{ labels:xs, datasets:[
      { label:'Precision', data:epochs.map(e=>e['metrics/precision(B)']), borderColor:color, backgroundColor:'transparent', pointRadius:2, borderWidth:2, tension:0.3 },
      { label:'Recall',    data:epochs.map(e=>e['metrics/recall(B)']),    borderColor:color, backgroundColor:'transparent', borderDash:[4,3], pointRadius:2, borderWidth:1.5, tension:0.3 },
    ]},
    options: lineOpts('P / R',0,1),
  })

  // Train losses
  if (dc.value.trainLoss) drillCharts.trainLoss = new Chart(dc.value.trainLoss, {
    type:'line', data:{ labels:xs, datasets:[
      { label:'box', data:epochs.map(e=>e['train/box_loss']), borderColor:color, backgroundColor:'transparent', pointRadius:2, borderWidth:1.5, tension:0.3 },
      { label:'cls', data:epochs.map(e=>e['train/cls_loss']), borderColor:'oklch(0.62 0.16 220deg)', backgroundColor:'transparent', pointRadius:2, borderWidth:1.5, tension:0.3 },
      { label:'dfl', data:epochs.map(e=>e['train/dfl_loss']), borderColor:'oklch(0.60 0.18 145deg)', backgroundColor:'transparent', pointRadius:2, borderWidth:1.5, tension:0.3 },
    ]},
    options: lineOpts('loss',null,null),
  })

  // Val losses
  if (dc.value.valLoss) drillCharts.valLoss = new Chart(dc.value.valLoss, {
    type:'line', data:{ labels:xs, datasets:[
      { label:'box', data:epochs.map(e=>e['val/box_loss']), borderColor:color, backgroundColor:'transparent', pointRadius:2, borderWidth:1.5, tension:0.3 },
      { label:'cls', data:epochs.map(e=>e['val/cls_loss']), borderColor:'oklch(0.62 0.16 220deg)', backgroundColor:'transparent', pointRadius:2, borderWidth:1.5, tension:0.3 },
      { label:'dfl', data:epochs.map(e=>e['val/dfl_loss']), borderColor:'oklch(0.60 0.18 145deg)', backgroundColor:'transparent', pointRadius:2, borderWidth:1.5, tension:0.3 },
    ]},
    options: lineOpts('val loss',null,null),
  })

  // Confusion matrix rendered as HTML (not Chart.js) — see template

  // BoxPR (P vs R scatter from curve data)
  const curveKeys = Object.keys(run).filter(k => k.startsWith('curve_'))
  const prKey = curveKeys.find(k => k.includes('pr') || k.includes('PR'))
  const pKey  = curveKeys.find(k => k.toLowerCase().endsWith('p_curve') || k.toLowerCase().endsWith('prec'))
  const rKey  = curveKeys.find(k => k.toLowerCase().endsWith('r_curve') || k.toLowerCase().endsWith('rec'))

  if (dc.value.boxPR && prKey && run[prKey]?.length) {
    const vals = run[prKey]
    drillCharts.boxPR = new Chart(dc.value.boxPR, {
      type:'line', data:{ labels:vals.map((_,i)=>+(i/vals.length).toFixed(2)), datasets:[{ label:'BoxPR', data:vals, borderColor:color, backgroundColor:'transparent', pointRadius:0, borderWidth:2 }] },
      options:{ responsive:true, maintainAspectRatio:true, aspectRatio:1.3, plugins:{legend:{display:false}}, scales:{x:{...BS,title:{display:true,text:'recall',color:C.tick,font:{family:'IBM Plex Mono',size:9}}},y:{...BS,min:0,max:1,title:{display:true,text:'precision',color:C.tick,font:{family:'IBM Plex Mono',size:9}}}} },
    })
  }

  if (dc.value.boxP && pKey && run[pKey]?.length) {
    const vals = run[pKey]
    drillCharts.boxP = new Chart(dc.value.boxP, {
      type:'line', data:{ labels:vals.map((_,i)=>+(i/vals.length).toFixed(2)), datasets:[{ label:'Precision', data:vals, borderColor:color, backgroundColor:'transparent', pointRadius:0, borderWidth:2 }] },
      options:{ responsive:true, maintainAspectRatio:true, aspectRatio:1.3, plugins:{legend:{display:false}}, scales:{x:{...BS,title:{display:true,text:'conf',color:C.tick,font:{family:'IBM Plex Mono',size:9}}},y:{...BS,min:0,max:1}} },
    })
  }

  if (dc.value.boxR && rKey && run[rKey]?.length) {
    const vals = run[rKey]
    drillCharts.boxR = new Chart(dc.value.boxR, {
      type:'line', data:{ labels:vals.map((_,i)=>+(i/vals.length).toFixed(2)), datasets:[{ label:'Recall', data:vals, borderColor:color, backgroundColor:'transparent', pointRadius:0, borderWidth:2 }] },
      options:{ responsive:true, maintainAspectRatio:true, aspectRatio:1.3, plugins:{legend:{display:false}}, scales:{x:{...BS,title:{display:true,text:'conf',color:C.tick,font:{family:'IBM Plex Mono',size:9}}},y:{...BS,min:0,max:1}} },
    })
  }
}

async function selectRun(id) {
  selectedRunId.value = selectedRunId.value === id ? null : id
  if (selectedRunId.value) await buildDrill(selectedRunId.value)
  else destroyDrill()
}

function setDays(d) { days.value = d }

async function fetch_(initial = false) {
  if (initial) loading.value = true
  try {
    const [chartRes, epochRes] = await Promise.all([
      fetch(`/api/stats/charts?days=${days.value}`),
      fetch('/api/training/epoch-data'),
    ])
    if (chartRes.ok) data.value = await chartRes.json()
    if (epochRes.ok) epochData.value = await epochRes.json()
  } catch {}
  if (initial) loading.value = false
  await nextTick()
  buildGeneral()
}

watch(days, () => fetch_(false))
onMounted(() => fetch_(true))
</script>

<style scoped>
.analytics { background: var(--bg-1); border-top: 1px solid var(--fg-3); position: relative; overflow: hidden; }
.section-header { display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 16px; }
.analytics-toggle { display: flex; }
.analytics-toggle button { font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.12em; padding: 6px 16px; border: 1px solid var(--fg-3); background: none; color: var(--fg-3); cursor: pointer; transition: all 0.15s; margin-left: -1px; }
.analytics-toggle button.active { border-color: var(--amber); color: var(--amber); z-index:1; position:relative; }
.analytics-toggle button:hover:not(.active) { color: var(--fg-0); border-color: var(--fg-1); }

.det-index { display: flex; align-items: flex-end; gap: 40px; flex-wrap: wrap; padding: 32px clamp(20px,5vw,80px); border-bottom: 1px solid var(--fg-3); }
.det-index-item { display: flex; flex-direction: column; gap: 4px; }
.det-index-num { font-family: var(--font-mono); font-size: clamp(32px,5vw,56px); font-weight: 700; line-height: 1; }
.det-index-label { font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.2em; color: var(--fg-3); }
.det-index-note { font-family: var(--font-mono); font-size: 10px; color: var(--fg-3); letter-spacing: 0.06em; line-height: 1.6; margin-left: auto; align-self: flex-end; max-width: 280px; text-align: right; }

.analytics-loading { color: var(--fg-3); font-size: 13px; padding: 40px clamp(20px,5vw,80px); }

.analytics-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 2px; padding: 2px; }
.chart-card { background: var(--bg-2); padding: 20px; }
.chart-label { font-size: 10px; color: var(--fg-3); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* Drill-down */
.drill-header { display: flex; align-items: center; gap: 20px; padding: 20px clamp(20px,5vw,80px); border-top: 1px solid var(--fg-3); flex-wrap: wrap; }
.drill-label { font-size: 10px; color: var(--fg-3); letter-spacing: 0.18em; flex-shrink: 0; }
.drill-select {
  font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.08em;
  background: var(--bg-2); color: var(--fg-1); border: 1px solid var(--fg-3);
  padding: 7px 14px; cursor: pointer; outline: none; min-width: 280px;
  transition: border-color 0.15s;
}
.drill-select:focus, .drill-select:hover { border-color: var(--amber); }
.drill-select option { background: var(--bg-2); color: var(--fg-1); }

.drill-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 2px; padding: 2px; }

.cm-wrap { display: flex; flex-direction: column; gap: 8px; }
.cm-grid { display: grid; gap: 2px; }
.cm-cell { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; border-radius: 2px; }
.cm-val { font-family: var(--font-mono); font-size: 10px; color: rgba(255,255,255,0.9); }
.cm-labels { display: flex; gap: 2px; }
.cm-label { flex: 1; font-family: var(--font-mono); font-size: 9px; color: var(--fg-3); text-align: center; letter-spacing: 0.05em; overflow: hidden; text-overflow: ellipsis; }
.chart-empty { font-family: var(--font-mono); font-size: 11px; color: var(--fg-3); padding: 40px 0; text-align: center; }

.drill-expand-enter-active, .drill-expand-leave-active { transition: max-height 0.4s ease, opacity 0.3s ease; overflow: hidden; }
.drill-expand-enter-from, .drill-expand-leave-to { max-height: 0; opacity: 0; }
.drill-expand-enter-to, .drill-expand-leave-from { max-height: 2000px; opacity: 1; }

@media (max-width: 900px) {
  .analytics-grid, .drill-grid { grid-template-columns: repeat(2,1fr); }
}
@media (max-width: 480px) {
  .analytics-grid, .drill-grid { grid-template-columns: 1fr; }
  .det-index { gap: 24px; }
  .det-index-note { margin-left:0; text-align:left; }
}
</style>
