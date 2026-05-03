<template>
  <div>
    <AppNav />

    <div class="arch-wrap">
      <!-- Sidebar -->
      <aside class="arch-sidebar">
        <a href="/" class="sidebar-back mono">← HOME</a>
        <div class="sidebar-header">EVIDENCE ARCHIVE</div>

        <div class="sidebar-group">
          <div class="sidebar-group-label">BY CLASS</div>
          <button class="sidebar-item" :class="{ active: !activeClass && !activeSource }" @click="clearAll">
            <span>ALL</span><span class="sidebar-badge">{{ items.length }}</span>
          </button>
          <template v-for="cls in classNodes" :key="cls.id">
            <button class="sidebar-item" :class="{ active: activeClass === cls.id && !activeSource }"
              @click="pickClass(cls.id)">
              <span>{{ cls.id }}</span><span class="sidebar-badge">{{ cls.total }}</span>
            </button>
            <button v-for="src in cls.sources" :key="src.id"
              class="sidebar-item sidebar-sub"
              :class="{ active: activeClass === cls.id && activeSource === src.id }"
              @click="pickClassSource(cls.id, src.id)">
              <span>↳ {{ src.id }}</span><span class="sidebar-badge">{{ src.count }}</span>
            </button>
          </template>
        </div>

        <div class="sidebar-divider" />

        <div class="sidebar-group">
          <div class="sidebar-group-label">BY SOURCE</div>
          <button v-for="src in sourceNodes" :key="src.id"
            class="sidebar-item"
            :class="{ active: !activeClass && activeSource === src.id }"
            @click="pickSource(src.id)">
            <span>{{ src.id }}</span><span class="sidebar-badge">{{ src.count }}</span>
          </button>
        </div>
      </aside>

      <!-- Main content -->
      <main class="arch-main">
        <div class="arch-controls">
          <input class="arch-search" v-model="search" placeholder="Search title or ID..." />
          <div class="arch-meta mono">{{ filtered.length }} RESULTS &nbsp;·&nbsp; PAGE {{ page }} / {{ totalPages }}</div>
        </div>

        <div v-if="filtered.length === 0" class="arch-empty">NO RESULTS — ADJUST FILTERS</div>
        <div v-else class="footage-grid">
          <FootageCard v-for="item in paginated" :key="item.id" :item="item" @open="openModal" />
        </div>

        <div class="arch-pagination" v-if="totalPages > 1">
          <button @click="page = Math.max(1, page - 1)" :disabled="page === 1">← PREV</button>
          <span class="mono">{{ page }} / {{ totalPages }}</span>
          <button @click="page = Math.min(totalPages, page + 1)" :disabled="page === totalPages">NEXT →</button>
        </div>
      </main>
    </div>

    <Teleport to="body">
      <FootageModal v-if="modalItem" :item="modalItem" @close="modalItem = null" />
    </Teleport>

    <SiteFooter />
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AppNav      from '../components/AppNav.vue'
import FootageCard from '../components/FootageCard.vue'
import FootageModal from '../components/FootageModal.vue'
import SiteFooter  from '../components/SiteFooter.vue'

const route  = useRoute()
const router = useRouter()

const items        = ref([])
const search       = ref(route.query.q      || '')
const activeClass  = ref(route.query.class  || null)
const activeSource = ref(route.query.source || null)
const page         = ref(Number(route.query.page) || 1)
const PER_PAGE     = 20
const modalItem    = ref(null)

onMounted(async () => {
  try {
    const res = await fetch('/api/annotated-clips')
    if (res.ok) items.value = await res.json()
  } catch {}
  await nextTick()
  window.scrollTo({ top: 0, behavior: 'instant' })
})

function openModal(item) { modalItem.value = item }

function syncUrl() {
  const q = {}
  if (activeClass.value)  q.class  = activeClass.value
  if (activeSource.value) q.source = activeSource.value
  if (search.value)       q.q      = search.value
  if (page.value > 1)     q.page   = page.value
  router.replace({ query: q })
}

function clearAll()                    { activeClass.value = null; activeSource.value = null; search.value = ''; page.value = 1; syncUrl() }
function pickClass(id)                 { activeClass.value = id;   activeSource.value = null; page.value = 1; syncUrl() }
function pickSource(id)                { activeClass.value = null; activeSource.value = id;   page.value = 1; syncUrl() }
function pickClassSource(cls, src)     { activeClass.value = cls;  activeSource.value = src;  page.value = 1; syncUrl() }

watch(search, () => { page.value = 1; syncUrl() })
watch(page,   syncUrl)

const filtered = computed(() => items.value.filter(item => {
  const cm = !activeClass.value  || item.detClass === activeClass.value
  const sm = !activeSource.value || item.source   === activeSource.value
  const qm = !search.value       || item.title.toLowerCase().includes(search.value.toLowerCase())
                                  || item.src.toLowerCase().includes(search.value.toLowerCase())
  return cm && sm && qm
}))

const totalPages = computed(() => Math.max(1, Math.ceil(filtered.value.length / PER_PAGE)))
const paginated  = computed(() => {
  const s = (page.value - 1) * PER_PAGE
  return filtered.value.slice(s, s + PER_PAGE)
})

const classNodes = computed(() => {
  const CLASSES  = ['AIRCRAFT', 'VEHICLE', 'PERSONNEL']
  const SOURCES  = ['Funker530', 'GeoConfirmed', 'Submitted']
  return CLASSES
    .map(cls => ({
      id:      cls,
      total:   items.value.filter(i => i.detClass === cls).length,
      sources: SOURCES
        .map(src => ({ id: src, count: items.value.filter(i => i.detClass === cls && i.source === src).length }))
        .filter(s => s.count > 0),
    }))
    .filter(c => c.total > 0)
})

const sourceNodes = computed(() =>
  ['Funker530', 'GeoConfirmed', 'Submitted']
    .map(src => ({ id: src, count: items.value.filter(i => i.source === src).length }))
    .filter(s => s.count > 0)
)
</script>

<style scoped>
.arch-wrap {
  display: flex;
  min-height: calc(100vh - 64px);
  padding-top: 64px;
}

.arch-sidebar {
  width: 220px;
  flex-shrink: 0;
  background: var(--bg-2);
  border-right: 1px solid rgba(255,255,255,0.06);
  padding: 24px 0;
  position: sticky;
  top: 64px;
  height: calc(100vh - 64px);
  overflow-y: auto;
}

.sidebar-back {
  display: block;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  color: var(--fg-3);
  padding: 12px 20px 8px;
  text-decoration: none;
  transition: color 0.15s;
}
.sidebar-back:hover { color: var(--amber); }

.sidebar-header {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.18em;
  color: var(--amber);
  text-transform: uppercase;
  padding: 0 20px 16px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  margin-bottom: 8px;
}

.sidebar-group { padding: 8px 0; }

.sidebar-group-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.15em;
  color: var(--fg-3);
  text-transform: uppercase;
  padding: 0 20px 6px;
}

.sidebar-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  padding: 6px 20px;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.06em;
  color: var(--fg-2);
  background: none;
  border: none;
  cursor: pointer;
  text-align: left;
  text-transform: uppercase;
  transition: color 0.15s, background 0.15s;
}
.sidebar-item:hover { color: var(--fg-1); background: rgba(255,255,255,0.03); }
.sidebar-item.active { color: var(--amber); background: rgba(194,120,40,0.08); }
.sidebar-sub { padding-left: 32px; font-size: 10px; }

.sidebar-badge {
  font-size: 9px;
  color: var(--fg-3);
  background: rgba(255,255,255,0.05);
  padding: 1px 5px;
  border-radius: 2px;
  flex-shrink: 0;
}
.sidebar-item.active .sidebar-badge { color: var(--amber); background: rgba(194,120,40,0.15); }

.sidebar-divider { height: 1px; background: rgba(255,255,255,0.06); margin: 8px 0; }

.arch-main {
  flex: 1;
  padding: 32px 40px 60px;
  min-width: 0;
}

.arch-controls {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 28px;
}

.arch-search {
  flex: 1;
  max-width: 400px;
  background: var(--bg-2);
  border: 1px solid rgba(255,255,255,0.08);
  color: var(--fg-1);
  font-family: var(--font-mono);
  font-size: 12px;
  padding: 8px 12px;
  outline: none;
  letter-spacing: 0.06em;
}
.arch-search:focus { border-color: var(--amber); }

.arch-meta { font-size: 11px; color: var(--fg-3); letter-spacing: 0.08em; white-space: nowrap; }

.arch-empty {
  padding: 80px 0;
  text-align: center;
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--fg-3);
  letter-spacing: 0.1em;
}

.arch-pagination {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 32px;
  font-family: var(--font-mono);
  font-size: 11px;
}
.arch-pagination button {
  background: var(--bg-2);
  border: 1px solid rgba(255,255,255,0.08);
  color: var(--fg-2);
  padding: 6px 14px;
  cursor: pointer;
  font-family: inherit;
  font-size: inherit;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  transition: border-color 0.15s, color 0.15s;
}
.arch-pagination button:hover:not(:disabled) { border-color: var(--amber); color: var(--amber); }
.arch-pagination button:disabled { opacity: 0.35; cursor: default; }
.arch-pagination span { color: var(--fg-3); }
</style>
