<template>
  <div class="data-strip">
    <div v-for="d in items" :key="d.label" class="data-strip-item">
      <div class="data-big">{{ d.big }}<span>{{ d.unit }}</span></div>
      <div class="data-label">{{ d.label }}</div>
      <div class="data-sublabel">{{ d.sub }}</div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const stats = ref(null)
onMounted(async () => {
  try {
    const res = await fetch('/api/stats')
    if (res.ok) stats.value = await res.json()
  } catch {}
})

function fmtK(n) {
  if (n == null) return '—'
  if (n >= 1e6) return { big: (n / 1e6).toFixed(1), unit: 'M' }
  if (n >= 1e3) return { big: Math.round(n / 1e3), unit: 'K' }
  return { big: n, unit: '' }
}

const items = computed(() => {
  const labeled = fmtK(stats.value?.images_labeled)
  return [
    {
      big:  stats.value ? String(stats.value.clips_total) : '—',
      unit: '',
      label: 'Total clips archived',
      sub: 'Raw footage downloaded',
    },
    {
      big:  stats.value ? String(stats.value.raw_gb) : '—',
      unit: 'GB',
      label: 'Raw footage stored',
      sub: 'From Russian, Ukrainian and Independent sources',
    },
    {
      big:  stats.value ? labeled.big : '—',
      unit: stats.value ? labeled.unit : '',
      label: 'Training images',
      sub: 'Labeled and annotated across 5 datasets',
    },
  ]
})
</script>
