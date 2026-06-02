<template>
  <section class="capabilities" id="capabilities">
    <div class="radar-bg">
      <RadarCanvas :opacity="0.35" color="194, 120, 40" />
    </div>
    <div class="section-header">
      <div>
        <div class="section-tag">How It Works</div>
        <h2 class="section-title">Automated pipeline</h2>
      </div>
    </div>
    <!-- Mobile pipeline: vertical list (hidden on desktop) -->
    <div class="pipeline-mobile">
      <div v-for="(node, i) in pipeline" :key="node.id" class="pipeline-mobile-row">
        <div class="pipeline-mobile-node">
          <span class="pipeline-mobile-label">{{ node.label }}</span>
          <span class="pipeline-mobile-stat">{{ node.stat }}</span>
          <span class="pipeline-mobile-sub">{{ node.sub }}</span>
        </div>
        <div v-if="i < pipeline.length - 1" class="pipeline-mobile-arrow">↓</div>
      </div>
    </div>

    <!-- Desktop pipeline SVG (hidden on mobile) -->
    <div class="pipeline-wrap">
      <svg class="pipeline-svg" viewBox="0 0 1140 170" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <filter id="amber-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        <!-- chevron nodes: each is a pentagon arrow shape, overlapping slightly -->
        <g v-for="(node, i) in pipeline" :key="node.id">
          <!--
            Chevron shape:
            - First node: flat left  → pointed right
            - Middle:     notched left → pointed right
            - Last:       notched left → flat right
            spacing=190, body half-width=85, notch/point depth=22
          -->
          <polygon
            :points="chevronPoints(i)"
            fill="#0d1012"
            stroke="#df6900"
            stroke-width="1"
            :stroke-opacity="0.3"
          >
            <animate attributeName="stroke-opacity"
              values="0.2;0.55;0.2"
              :dur="`${3 + i * 0.25}s`"
              :begin="`${i * 0.35}s`"
              repeatCount="indefinite" />
          </polygon>
          <!-- amber left-edge accent line (visible on first node only as top bar, others as notch highlight) -->
          <line
            :x1="190*i + (i===0 ? 15 : 0)" y1="20"
            :x2="190*i + 170" y2="20"
            stroke="#df6900" stroke-width="1.5" stroke-opacity="0.55"
          />
          <!-- label — center in body rect: first node body 15→170 (cx=92), others 0→170 (cx=85), but shift right +10 for notch visual balance -->
          <text :x="i===0 ? 190*i+92 : 190*i+96" y="55"
            text-anchor="middle"
            font-family="'IBM Plex Mono',monospace"
            font-size="10" fill="#df6900"
            letter-spacing="2.5" font-weight="600">{{ node.label }}</text>
          <text :x="i===0 ? 190*i+92 : 190*i+96" y="100"
            text-anchor="middle"
            font-family="'IBM Plex Mono',monospace"
            font-size="28" fill="#eef0f2"
            font-weight="700">{{ node.stat }}</text>
          <text :x="i===0 ? 190*i+92 : 190*i+96" y="133"
            text-anchor="middle"
            font-family="'IBM Plex Mono',monospace"
            font-size="9" fill="#3d4650"
            letter-spacing="1.8">{{ node.sub }}</text>
        </g>

        <!-- grey animated dashes at each chevron junction (tip of node i-1 → notch of node i) -->
        <line v-for="i in 5" :key="`l${i}`"
          :x1="i*190 - 28" y1="85"
          :x2="i*190 + 52" y2="85"
          stroke="rgba(255,255,255,0.2)" stroke-width="1.5"
          stroke-dasharray="4 7"
        >
          <animate attributeName="stroke-dashoffset"
            from="22" to="0"
            :dur="`${0.85 + i*0.04}s`"
            repeatCount="indefinite" />
        </line>

        <!-- animated dots traveling over the chevron tips -->
        <circle v-for="i in 5" :key="`d${i}`" r="4" cy="85"
          fill="#df6900" filter="url(#amber-glow)">
          <animate attributeName="cx"
            :from="`${190*i - 22}`"
            :to="`${190*i + 22}`"
            :dur="`${1.2}s`"
            :begin="`${i * 0.22}s`"
            repeatCount="indefinite" />
          <animate attributeName="opacity"
            values="0;1;1;0" keyTimes="0;0.15;0.85;1"
            :dur="`${1.2}s`"
            :begin="`${i * 0.22}s`"
            repeatCount="indefinite" />
        </circle>
      </svg>
    </div>

    <div class="cap-grid">
      <div v-for="cap in capabilities" :key="cap.num" class="cap-card">
        <div class="cap-num">{{ cap.num }}</div>
        <div class="cap-icon" style="color:var(--amber)">
          <component :is="icons[cap.icon]" />
        </div>
        <div class="cap-title">{{ cap.title }}</div>
        <div class="cap-desc">{{ cap.desc }}</div>
        <div class="cap-metric">
          <div>
            <div class="cap-metric-val">{{ cap.m1 }}</div>
            <div class="cap-metric-label">{{ cap.m1l }}</div>
          </div>
          <div>
            <div class="cap-metric-val">{{ cap.m2 }}</div>
            <div class="cap-metric-label">{{ cap.m2l }}</div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { h, ref, computed, onMounted } from 'vue'
import RadarCanvas from './RadarCanvas.vue'

const stats = ref(null)

onMounted(async () => {
  try {
    const res = await fetch('/api/stats')
    if (res.ok) stats.value = await res.json()
  } catch {}
})

function fmt(n) {
  if (n == null) return '—'
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  if (n >= 1e3) return Math.round(n / 1e3) + 'K'
  return String(n)
}

// Chevron shape: spacing=190, body left=0, right=170, point tip at 192, notch at -22
// First node: flat left (x=15). Last node: no right point (x=170 flat).
function chevronPoints(i) {
  const x = i * 190      // left edge of this chevron body
  const W = 170          // body width
  const P = 22           // point / notch depth
  const top = 20, bot = 150, mid = 85
  const isFirst = i === 0
  const isLast  = i === 5
  if (isFirst) {
    // flat left, pointed right
    return `${x+15},${top} ${x+W},${top} ${x+W+P},${mid} ${x+W},${bot} ${x+15},${bot}`
  } else if (isLast) {
    // notched left, flat right
    return `${x},${top} ${x+W},${top} ${x+W},${bot} ${x},${bot} ${x+P},${mid}`
  } else {
    // notched left, pointed right
    return `${x},${top} ${x+W},${top} ${x+W+P},${mid} ${x+W},${bot} ${x},${bot} ${x+P},${mid}`
  }
}

const pipeline = computed(() => {
  const s = stats.value
  const models = s?.models ?? {}
  const bestMap = Object.values(models).reduce((best, m) => m.map50 != null && m.map50 > best ? m.map50 : best, 0)
  return [
    { id: 'scrape',   label: 'SCRAPE',   stat: s ? fmt(s.clips_total)      : '—', sub: 'clips ingested'    },
    { id: 'gdino',    label: 'GDINO',    stat: 'Zero-shot',                        sub: 'auto-labeling'     },
    { id: 'dataset',  label: 'DATASET',  stat: s ? fmt(s.images_labeled)    : '—', sub: 'training images'  },
    { id: 'train',    label: 'TRAIN',    stat: bestMap ? bestMap.toFixed(3)  : '—', sub: 'best mAP50'       },
    { id: 'annotate', label: 'ANNOTATE', stat: s ? String(s.clips_annotated) : '—', sub: 'annotated clips'  },
    { id: 'feed',     label: 'FEED',     stat: 'Live',                              sub: 'public archive'   },
  ]
})

const capabilities = computed(() => [
  {
    num: '01',
    title: 'Continuous Scraping',
    desc: 'Celery beat task monitors Funker530 and GeoConfirmed REST APIs on an hourly schedule. New clips are deduplicated by SHA-256 URL hash and downloaded with yt-dlp.',
    m1: stats.value ? String(stats.value.clips_total) : '—', m1l: 'Clips archived',
    m2: stats.value ? `${stats.value.raw_gb}GB`         : '—', m2l: 'Media stored',
    icon: 'geo',
  },
  {
    num: '02',
    title: 'Specialist YOLO Models',
    desc: 'Three nc=1 YOLO models run sequentially on every clip — Aircraft, Vehicle, Personnel. Each is trained exclusively on its target class to eliminate cross-class noise.',
    m1: '3',   m1l: 'Specialist models',
    m2: '0.4', m2l: 'Conf. threshold',
    icon: 'time',
  },
  {
    num: '03',
    title: 'GDINO Auto-Labeling',
    desc: 'GroundingDINO zero-shot model labels incoming footage before specialist weights are ready. 15-term prompt covers all military asset classes. No manual annotation.',
    m1: 'Zero-shot',                                                           m1l: 'Label method',
    m2: stats.value ? fmt(stats.value.images_labeled) : '—',                  m2l: 'Training images',
    icon: 'chain',
  },
  {
    num: '04',
    title: 'Annotated Archive',
    desc: 'Every processed clip is re-encoded as H.264 MP4 with per-model colour-coded bounding boxes. Browse by detection class or source. Admin panel triggers retraining.',
    m1: stats.value ? String(stats.value.clips_annotated) : '—', m1l: 'Annotated clips',
    m2: '3',                                                      m2l: 'Detection classes',
    icon: 'api',
  },
])

const GeoIcon  = () => h('svg', { width: 40, height: 40, viewBox: '0 0 40 40', fill: 'none' }, [
  h('rect',   { x: 1, y: 1, width: 38, height: 38, stroke: 'currentColor', 'stroke-opacity': 0.2, 'stroke-width': 1 }),
  h('circle', { cx: 20, cy: 20, r: 8, stroke: 'currentColor', 'stroke-opacity': 0.5, 'stroke-width': 1 }),
  h('circle', { cx: 20, cy: 20, r: 2, fill: 'currentColor' }),
  h('line',   { x1: 20, y1: 1, x2: 20, y2: 39, stroke: 'currentColor', 'stroke-opacity': 0.2, 'stroke-width': 1 }),
  h('line',   { x1: 1, y1: 20, x2: 39, y2: 20, stroke: 'currentColor', 'stroke-opacity': 0.2, 'stroke-width': 1 }),
  h('circle', { cx: 20, cy: 12, r: 1.5, fill: 'currentColor', 'fill-opacity': 0.7 }),
])
const TimeIcon = () => h('svg', { width: 40, height: 40, viewBox: '0 0 40 40', fill: 'none' }, [
  h('rect',   { x: 1, y: 1, width: 38, height: 38, stroke: 'currentColor', 'stroke-opacity': 0.2, 'stroke-width': 1 }),
  h('circle', { cx: 20, cy: 20, r: 10, stroke: 'currentColor', 'stroke-opacity': 0.5, 'stroke-width': 1 }),
  h('line',   { x1: 20, y1: 12, x2: 20, y2: 20, stroke: 'currentColor', 'stroke-width': 1.5 }),
  h('line',   { x1: 20, y1: 20, x2: 27, y2: 20, stroke: 'currentColor', 'stroke-width': 1.5, 'stroke-opacity': 0.6 }),
  h('circle', { cx: 20, cy: 20, r: 1.5, fill: 'currentColor' }),
])
const ChainIcon = () => h('svg', { width: 40, height: 40, viewBox: '0 0 40 40', fill: 'none' }, [
  h('rect', { x: 1, y: 1, width: 38, height: 38, stroke: 'currentColor', 'stroke-opacity': 0.2, 'stroke-width': 1 }),
  h('rect', { x: 6, y: 15, width: 12, height: 10, rx: 5, stroke: 'currentColor', 'stroke-opacity': 0.7, 'stroke-width': 1.5 }),
  h('rect', { x: 22, y: 15, width: 12, height: 10, rx: 5, stroke: 'currentColor', 'stroke-opacity': 0.7, 'stroke-width': 1.5 }),
  h('line', { x1: 18, y1: 20, x2: 22, y2: 20, stroke: 'currentColor', 'stroke-width': 1.5, 'stroke-opacity': 0.7 }),
])
const ApiIcon  = () => h('svg', { width: 40, height: 40, viewBox: '0 0 40 40', fill: 'none' }, [
  h('rect',     { x: 1, y: 1, width: 38, height: 38, stroke: 'currentColor', 'stroke-opacity': 0.2, 'stroke-width': 1 }),
  h('polyline', { points: '8,20 14,12 14,28', stroke: 'currentColor', 'stroke-opacity': 0.6, 'stroke-width': 1.5, fill: 'none' }),
  h('line',     { x1: 18, y1: 12, x2: 22, y2: 28, stroke: 'currentColor', 'stroke-opacity': 0.7, 'stroke-width': 1.5 }),
  h('polyline', { points: '26,12 32,20 26,28', stroke: 'currentColor', 'stroke-opacity': 0.6, 'stroke-width': 1.5, fill: 'none' }),
])
const icons = { geo: GeoIcon, time: TimeIcon, chain: ChainIcon, api: ApiIcon }
</script>
