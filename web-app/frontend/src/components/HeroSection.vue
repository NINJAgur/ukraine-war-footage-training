<template>
  <section class="hero">

    <video
      v-if="heroVideoSrc"
      autoplay muted loop playsinline
      :src="heroVideoSrc"
      style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0.42;z-index:1"
    />

    <div style="position:absolute;inset:0;z-index:0;pointer-events:none">
      <div class="hero-ml-bg">
        <MLCard :cat="heroCat" :hero-mode="true" />
      </div>
    </div>

    <div class="hero-ml-veil"></div>

    <div class="hero-coords fade-up fade-up-1 mono">
      <div>48.3794° N</div>
      <div>31.1656° E</div>
      <div style="margin-top:8px;font-size:10px;color:var(--fg-3)">UA THEATER</div>
    </div>

    <div class="hero-content">
      <div class="hero-tag fade-up fade-up-1">Open Military Asset Detection Models</div>
      <h1 class="hero-headline fade-up fade-up-2">
        <span class="hl-line">
          <span class="hl-dark" aria-hidden="true">Every strike.</span>
          <span class="hl-orig">Every strike.</span>
        </span>
        <span class="hl-line">
          <span class="hl-dark" aria-hidden="true">Every <em>asset.</em></span>
          <span class="hl-orig">Every <em>asset.</em></span>
        </span>
        <span class="hl-line">
          <span class="hl-dark" aria-hidden="true">Detected.</span>
          <span class="hl-orig">Detected.</span>
        </span>
      </h1>
      <p class="hero-sub fade-up fade-up-3">
        Four YOLOv8 models trained on real conflict footage — aircraft, vehicles, personnel.
        Continuously self-improving. Free to download and use.
      </p>
      <div class="hero-actions">
        <a href="#detection" class="btn-primary">Explore Models</a>
        <router-link to="/archive" class="btn-secondary">Browse Archive</router-link>
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

import { ref, computed, onMounted } from 'vue'

const heroVideoSrc = ref(null)
const heroCat = computed(() => heroVideoSrc.value ? { ...GENERALIST_CAT, videoSrc: heroVideoSrc.value } : GENERALIST_CAT)
const stats = ref([
  { num: '—', label: 'AIRCRAFT mAP50' },
  { num: '—', label: 'VEHICLE mAP50' },
  { num: '—', label: 'PERSONNEL mAP50' },
])

onMounted(async () => {
  try {
    const [statsRes, clipsRes] = await Promise.all([
      fetch('/api/stats'),
      fetch('/api/annotated-clips'),
    ])
    if (statsRes.ok) {
      const d = await statsRes.json()
      const m = d.models ?? {}
      stats.value[0] = { num: m.AIRCRAFT?.map50?.toFixed(3) ?? '—', label: 'AIRCRAFT mAP50' }
      stats.value[1] = { num: m.VEHICLE?.map50?.toFixed(3)  ?? '—', label: 'VEHICLE mAP50'  }
      stats.value[2] = { num: m.PERSONNEL?.map50?.toFixed(3) ?? '—', label: 'PERSONNEL mAP50' }
    }
    if (clipsRes.ok) {
      const clips = await clipsRes.json()
      const clip = clips.find(c => c.detClass === 'GENERAL')
      if (clip?.videoUrl) heroVideoSrc.value = clip.videoUrl
    }
  } catch {}
})
</script>

<style scoped>
.hl-line {
  display: block;
  position: relative;
  width: fit-content;
  isolation: isolate;
}

.hl-dark {
  position: absolute;
  top: 0; left: 0;
  color: var(--bg-0);
  white-space: nowrap;
  clip-path: inset(0 100% 0 0);
  animation: hl-clip-dark 9s linear infinite;
  animation-fill-mode: backwards;
  pointer-events: none;
}
.hl-dark em { color: var(--bg-0); }

.hl-orig {
  display: block;
  white-space: nowrap;
  clip-path: inset(0 0 0 0%);
  animation: hl-clip-orig 9s linear infinite;
  animation-fill-mode: backwards;
}

.hl-line:nth-child(2) .hl-dark,
.hl-line:nth-child(2) .hl-orig,
.hl-line:nth-child(2)::before { animation-delay: 1.5s; }

.hl-line:nth-child(3) .hl-dark,
.hl-line:nth-child(3) .hl-orig,
.hl-line:nth-child(3)::before { animation-delay: 3s; }

.hl-line::before {
  content: '';
  position: absolute;
  inset: 0;
  background: var(--amber);
  z-index: -1;
  transform-origin: left;
  transform: scaleX(0);
  opacity: 0;
  animation: hl-sweep 9s linear infinite;
  animation-fill-mode: backwards;
}

/* 9s cycle: ~0.5s sweep, 0.5s hold, ~0.3s fade, 0.25s gap, ~4.75s pause */
@keyframes hl-sweep {
  0%    { transform: scaleX(0); opacity: 0; }
  1%    { transform: scaleX(0); opacity: 1; }
  5.5%  { transform: scaleX(1); opacity: 1; }
  11%   { transform: scaleX(1); opacity: 1; }
  14%   { transform: scaleX(1); opacity: 0; }
  14.5% { transform: scaleX(0); opacity: 0; }
  100%  { transform: scaleX(0); opacity: 0; }
}

@keyframes hl-clip-dark {
  0%    { clip-path: inset(0 100% 0 0); }
  1%    { clip-path: inset(0 100% 0 0); }
  5.5%  { clip-path: inset(0 0%   0 0); }
  11%   { clip-path: inset(0 0%   0 0); }
  11.1% { clip-path: inset(0 100% 0 0); }
  100%  { clip-path: inset(0 100% 0 0); }
}

@keyframes hl-clip-orig {
  0%    { clip-path: inset(0 0 0 0%);   }
  1%    { clip-path: inset(0 0 0 0%);   }
  5.5%  { clip-path: inset(0 0 0 100%); }
  11%   { clip-path: inset(0 0 0 100%); }
  11.1% { clip-path: inset(0 0 0 0%);   }
  100%  { clip-path: inset(0 0 0 0%);   }
}
</style>
