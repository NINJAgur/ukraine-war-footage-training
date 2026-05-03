<template>
  <div
    ref="cardEl"
    class="ml-card"
    :class="{ expanded }"
    :data-cat="cat.id"
  >
    <div class="ml-card-inner">
      <div class="ml-video-bg">
        <video
          v-if="cat.videoSrc"
          autoplay muted loop playsinline
          :src="cat.videoSrc"
          style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0.55"
        />
      </div>
      <canvas ref="canvasEl" class="ml-canvas" style="position:absolute;inset:0;width:100%;height:100%;display:block"></canvas>
      <div class="ml-scanline"></div>
      <div class="ml-hud-corner tl"></div>
      <div class="ml-hud-corner tr"></div>
      <div class="ml-hud-corner bl"></div>
      <div class="ml-hud-corner br"></div>
      <div class="ml-hud-top">
        <span>{{ cat.id.toUpperCase() }}</span>
        <span style="color:var(--fg-2)">{{ liveModelInfo }}</span>
      </div>
      <div class="ml-hud-bottom">
        <div v-if="cat.desc" style="text-align:right;max-width:100%;overflow:hidden">
          <div style="font-size:10px;color:var(--fg-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{{ cat.desc }}</div>
        </div>
      </div>
      <div class="ml-cat-label" :style="{ bottom: expanded ? '60px' : '24px' }">
        {{ cat.label.toUpperCase() }}
      </div>
      <div class="ml-cat-title" :style="{ bottom: expanded ? '80px' : '44px', fontSize: expanded ? 'clamp(32px, 4vw, 56px)' : 'clamp(22px, 3vw, 38px)' }">
        {{ cat.title }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useMLCanvas } from '../composables/useMLCanvas.js'

const props = defineProps({
  cat:      { type: Object, required: true },
  heroMode: { type: Boolean, default: false },
  stats:    { type: Object, default: null },
})

const liveModelInfo = computed(() => {
  const s = props.stats
  if (!s) return props.cat.modelInfo || 'YOLOv8m'
  const imgs = s.images >= 1000 ? Math.round(s.images / 1000) + 'K' : (s.images || '—')
  const map  = s.map50 != null ? `mAP50 ${s.map50.toFixed(3)}` : s.status ?? '—'
  return `YOLOv8m · ${imgs} imgs · ${map}`
})

const cardEl   = ref(null)
const canvasEl = ref(null)
const expanded = ref(false)

const detectionsRef = computed(() => props.cat.detections)
useMLCanvas(canvasEl, props.cat.id, detectionsRef, expanded)

let obs = null
onMounted(() => {
  if (props.heroMode) { expanded.value = true; return }
  obs = new IntersectionObserver(
    ([entry]) => { expanded.value = entry.isIntersecting },
    { threshold: 0.35 }
  )
  if (cardEl.value) obs.observe(cardEl.value)
})
onUnmounted(() => { if (obs) obs.disconnect() })
</script>
