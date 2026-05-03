<template>
  <section class="hero">

    <video
      autoplay muted loop playsinline
      src="/hero.mp4"
      style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0.42;z-index:1"
    />

    <div style="position:absolute;inset:0;z-index:0;pointer-events:none">
      <div class="hero-ml-bg">
        <MLCard :cat="GENERALIST_CAT" :hero-mode="true" />
      </div>
    </div>

    <div class="hero-ml-veil"></div>

    <div class="hero-coords fade-up fade-up-1 mono">
      <div>48.3794° N</div>
      <div>31.1656° E</div>
      <div style="margin-top:8px;font-size:10px;color:var(--fg-3)">UA THEATER</div>
    </div>

    <div class="hero-content">
      <div class="hero-tag fade-up fade-up-1">Ukraine Combat Footage Archive</div>
      <h1 class="hero-headline fade-up fade-up-2">
        Every strike.<br>
        Every <em>asset.</em><br>
        Detected.
      </h1>
      <p class="hero-sub fade-up fade-up-3">
        Automated scraping, YOLO detection, and annotated archiving of combat footage
        from open-source channels. Aircraft · Vehicle · Personnel.
      </p>
      <div class="hero-actions">
        <router-link to="/archive" class="btn-primary">Browse Archive</router-link>
        <router-link to="/admin/login" class="btn-secondary">Admin Panel</router-link>
      </div>
    </div>

    <div class="hero-stats">
      <div v-for="s in stats" :key="s.label" class="hero-stat fade-up fade-up-3">
        <div class="hero-stat-num">{{ s.num }}</div>
        <div class="hero-stat-label">{{ s.label }}</div>
      </div>
    </div>

  </section>
</template>

<script setup>
import MLCard from './MLCard.vue'
import { GENERALIST_CAT } from '../data/constants.js'

import { ref, onMounted } from 'vue'

const stats = ref([
  { num: '3',     label: 'YOLO specialists' },
  { num: '102K+', label: 'Training images' },
  { num: '0.91',  label: 'AIRCRAFT mAP50' },
])

onMounted(async () => {
  try {
    const res = await fetch('/api/stats')
    if (!res.ok) return
    const d = await res.json()
    const k = d.images_labeled >= 1000 ? Math.round(d.images_labeled / 1000) + 'K+' : String(d.images_labeled)
    stats.value[1] = { num: k, label: 'Training images' }
    stats.value[0] = { num: String(d.clips_total), label: 'Clips archived' }
  } catch {}
})
</script>
