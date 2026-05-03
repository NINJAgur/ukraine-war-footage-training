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
import { TICKER_ITEMS } from '../data/constants.js'

const BASE_ITEMS = TICKER_ITEMS.filter(t => !t.startsWith('AIRCRAFT MODEL') && !t.startsWith('VEHICLE MODEL') && !t.startsWith('PERSONNEL MODEL'))

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
