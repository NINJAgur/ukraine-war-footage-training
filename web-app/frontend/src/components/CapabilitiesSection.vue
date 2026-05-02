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
import { h } from 'vue'
import RadarCanvas from './RadarCanvas.vue'

const capabilities = [
  {
    num: '01',
    title: 'Continuous Scraping',
    desc: 'Celery beat task monitors Funker530 and GeoConfirmed REST APIs on an hourly schedule. New clips are deduplicated by SHA-256 URL hash and downloaded with yt-dlp.',
    m1: '24/7', m1l: 'Monitoring',
    m2: '2',    m2l: 'Active sources',
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
    m1: 'Zero-shot', m1l: 'Label method',
    m2: '26K+',      m2l: 'Training images',
    icon: 'chain',
  },
  {
    num: '04',
    title: 'Annotated Archive',
    desc: 'Every processed clip is re-encoded as H.264 MP4 with per-model colour-coded bounding boxes. Browse by detection class or source. Admin panel triggers retraining.',
    m1: 'REST', m1l: 'API',
    m2: '3',    m2l: 'Detection classes',
    icon: 'api',
  },
]

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
