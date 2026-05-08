<template>
  <section class="hero">

    <video
      autoplay muted loop playsinline
      :src="heroVideoSrc"
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

const heroVideoSrc = ref('/hero.mp4')
const stats = ref([
  { num: '—', label: 'Clips archived' },
  { num: '—', label: 'Training images' },
  { num: '—', label: 'Best mAP50' },
])

onMounted(async () => {
  try {
    const [statsRes, clipsRes] = await Promise.all([
      fetch('/api/stats'),
      fetch('/api/annotated-clips'),
    ])
    if (statsRes.ok) {
      const d = await statsRes.json()
      const k = d.images_labeled >= 1000 ? Math.round(d.images_labeled / 1000) + 'K+' : String(d.images_labeled)
      stats.value[0] = { num: String(d.clips_total), label: 'Clips archived' }
      stats.value[1] = { num: k, label: 'Training images' }
      const maps = Object.values(d.models ?? {}).filter(m => m.map50 != null).map(m => m.map50)
      if (maps.length) stats.value[2] = { num: Math.max(...maps).toFixed(3), label: 'Best mAP50' }
    }
    if (clipsRes.ok) {
      const clips = await clipsRes.json()
      // Prefer most recent GENERAL clip, fall back to any clip
      const best = clips.find(c => c.det_class === 'GENERAL') ?? clips[0]
      if (best?.videoUrl) heroVideoSrc.value = best.videoUrl
    }
  } catch {}
})
</script>
