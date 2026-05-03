<template>
  <section class="ml-section" id="detection">
    <div class="radar-bg">
      <RadarCanvas :opacity="0.35" color="194, 120, 40" />
    </div>
    <div class="ml-section-header">
      <div class="section-tag">ML Detection Layers</div>
      <h2 class="section-title">Object detection models</h2>
      <p style="margin-top:16px;font-size:14px;color:var(--fg-2);max-width:520px;line-height:1.7;font-family:var(--font-mono);letter-spacing:0.02em">
        All archived footage is processed through specialized neural networks.<br>
        Scroll to expand each detection category.
      </p>
    </div>
    <div class="ml-cards-track">
      <template v-for="(cat, i) in categories" :key="cat.id">
        <div class="ml-card-spacer">
          <div class="ml-card-spacer-inner">
            <div class="ml-spacer-index mono" style="color:var(--fg-3)">
              {{ String(i + 1).padStart(2, '0') }} / {{ String(categories.length).padStart(2, '0') }}
            </div>
            <div>
              <div class="ml-spacer-cat" :style="{ color: `var(--cat-color-${cat.id})` }">
                {{ cat.label.toUpperCase() }}
              </div>
              <h3 class="ml-spacer-title">{{ cat.title }}</h3>
              <p class="ml-spacer-desc">{{ cat.desc }}</p>
              <div class="ml-spacer-meta">
                <span class="ml-spacer-badge">MODEL v2.4</span>
                <span class="ml-spacer-badge">{{ cat.detections.length }} CLASSES</span>
                <span class="ml-spacer-badge">{{ cat.src }}</span>
              </div>
            </div>
            <div class="ml-spacer-right">
              <div>
                <div class="ml-spacer-stat-val">{{ modelStat(cat.id, 'images') }}</div>
                <div class="ml-spacer-stat-key">Training images</div>
              </div>
              <div>
                <div class="ml-spacer-stat-val">{{ modelStat(cat.id, 'map50') }}</div>
                <div class="ml-spacer-stat-key">mAP50</div>
              </div>
              <div>
                <div class="ml-spacer-stat-val" :style="{ color: modelStatusColor(cat.id) }">{{ modelStat(cat.id, 'status') }}</div>
                <div class="ml-spacer-stat-key">Status</div>
              </div>
            </div>
          </div>
        </div>
        <MLCard :cat="cat" :index="i" :stats="models[CAT_TO_MODEL[cat.id]]" />
      </template>
    </div>
  </section>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ML_CATEGORIES } from '../data/constants.js'
import MLCard from './MLCard.vue'
import RadarCanvas from './RadarCanvas.vue'

const models    = ref({})
const videoUrls = ref({})

onMounted(async () => {
  try {
    const [statsRes, clipsRes] = await Promise.all([
      fetch('/api/stats'),
      fetch('/api/annotated-clips'),
    ])
    if (statsRes.ok) models.value = (await statsRes.json()).models ?? {}
    if (clipsRes.ok) {
      const clips = await clipsRes.json()
      const CLASS_MAP = { aircraft: 'AIRCRAFT', personnel: 'PERSONNEL', vehicles: 'VEHICLE' }
      for (const [catId, detClass] of Object.entries(CLASS_MAP)) {
        const clip = clips.find(c => c.detClass === detClass)
        if (clip) videoUrls.value[catId] = clip.videoUrl
      }
    }
  } catch {}
})

const categories = computed(() =>
  ML_CATEGORIES.map(cat => ({
    ...cat,
    videoSrc: videoUrls.value[cat.id] ?? null,
  }))
)

const CAT_TO_MODEL = { aircraft: 'AIRCRAFT', personnel: 'PERSONNEL', vehicles: 'VEHICLE' }

function modelStat(catId, key) {
  const m = models.value[CAT_TO_MODEL[catId]]
  if (!m) return '—'
  if (key === 'images') return m.images >= 1000 ? Math.round(m.images / 1000) + 'K' : String(m.images || '—')
  if (key === 'map50')  return m.map50 != null ? m.map50.toFixed(3) : '—'
  if (key === 'status') return m.status ?? '—'
  return '—'
}

function modelStatusColor(catId) {
  const s = models.value[CAT_TO_MODEL[catId]]?.status
  if (s === 'DONE')     return 'var(--green)'
  if (s === 'TRAINING') return 'var(--amber)'
  return 'var(--fg-3)'
}
</script>
