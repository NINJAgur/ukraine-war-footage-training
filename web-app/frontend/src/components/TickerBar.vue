<template>
  <div class="ticker-bar">
    <div class="ticker-label">LIVE</div>
    <div style="overflow:hidden;flex:1">
      <div class="ticker-track">
        <span v-for="(item, i) in doubled" :key="i" class="ticker-item">
          <span class="ticker-sep">◆</span>{{ item }}
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const BASE_ITEMS = [
  'ARCHIVE ONLINE — ACTIVE COLLECTION',
  'FUNKER530 — MONITORING ACTIVE',
  'GEOCONFIRMED — MONITORING ACTIVE',
  'GDINO LABELING — 26K IMAGES COMPLETE',
  'LAST SCRAPE: AUTO',
  'COLLECTION PERIOD: FEB 2022 — PRESENT',
  '3 DETECTION CLASSES: AIRCRAFT · VEHICLE · PERSONNEL',
]

const models = ref({})
onMounted(async () => {
  try {
    const res = await fetch('/api/stats')
    if (res.ok) models.value = (await res.json()).models ?? {}
  } catch {}
})

function modelLine(key, label) {
  const m = models.value[key]
  if (!m) return `${label} — QUEUED`
  if (m.status === 'DONE' && m.map50 != null) return `${label} — mAP50 ${m.map50.toFixed(3)}`
  return `${label} — ${m.status ?? 'QUEUED'}`
}

const items = computed(() => [
  ...BASE_ITEMS,
  modelLine('AIRCRAFT',  'AIRCRAFT MODEL'),
  modelLine('VEHICLE',   'VEHICLE MODEL'),
  modelLine('PERSONNEL', 'PERSONNEL MODEL'),
])
const doubled = computed(() => [...items.value, ...items.value])
</script>
