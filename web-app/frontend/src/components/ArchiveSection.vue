<template>
  <section class="archive" id="archive">
    <div class="radar-bg">
      <RadarCanvas :opacity="0.35" color="194, 120, 40" />
    </div>
    <div class="section-header">
      <div>
        <div class="section-tag">Evidence Archive</div>
        <h2 class="section-title">Browse footage</h2>
        <div class="section-count mono">{{ filtered.length }} results</div>
      </div>
      <RouterLink to="/archive" class="view-all-link mono">VIEW ALL →</RouterLink>
    </div>

    <div class="archive-controls">
      <button
        v-for="c in DET_CLASSES" :key="c"
        class="filter-pill"
        :class="{ active: activeClass === c }"
        @click="activeClass = c"
      >{{ c }}</button>

      <button
        v-for="s in SOURCES" :key="s"
        class="filter-pill"
        :class="{ active: activeSource === s }"
        @click="activeSource = s"
      >{{ s }}</button>

      <input
        class="search-input"
        placeholder="Search title or source ID..."
        v-model="search"
      />
    </div>

    <div v-if="filtered.length === 0" style="padding:80px 0;text-align:center;font-family:var(--font-mono);font-size:13px;color:var(--fg-2);letter-spacing:0.1em">
      NO RESULTS — ADJUST FILTERS
    </div>
    <div v-else class="footage-grid">
      <FootageCard v-for="item in filtered" :key="item.id" :item="item" @open="openModal" />
    </div>
  </section>

  <Teleport to="body">
    <FootageModal v-if="modalItem" :item="modalItem" @close="modalItem = null" />
  </Teleport>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import FootageCard from './FootageCard.vue'
import FootageModal from './FootageModal.vue'
import RadarCanvas from './RadarCanvas.vue'

const DET_CLASSES = ['All', 'Aircraft', 'Vehicle', 'Personnel']
const SOURCES     = ['All sources', 'Funker530', 'GeoConfirmed']

const items        = ref([])
const activeClass  = ref('All')
const activeSource = ref('All sources')
const search       = ref('')
const modalItem    = ref(null)

onMounted(async () => {
  try {
    const res = await fetch('/api/annotated-clips')
    if (res.ok) items.value = await res.json()
  } catch {}
})

function openModal(item) { modalItem.value = item }

const filtered = computed(() => items.value.filter(item => {
  const cm = activeClass.value === 'All' || item.detClass.toLowerCase() === activeClass.value.toLowerCase()
  const sm = activeSource.value === 'All sources' || item.source === activeSource.value
  const qm = !search.value || item.title.toLowerCase().includes(search.value.toLowerCase()) || item.src.toLowerCase().includes(search.value.toLowerCase())
  return cm && sm && qm
}))
</script>
