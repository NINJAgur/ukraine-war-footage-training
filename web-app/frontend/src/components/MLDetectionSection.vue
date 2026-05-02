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
      <template v-for="(cat, i) in ML_CATEGORIES" :key="cat.id">
        <div class="ml-card-spacer">
          <div class="ml-card-spacer-inner">
            <div class="ml-spacer-index mono" style="color:var(--fg-3)">
              {{ String(i + 1).padStart(2, '0') }} / {{ String(ML_CATEGORIES.length).padStart(2, '0') }}
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
                <div class="ml-spacer-stat-val">{{ cat.detections.length }}</div>
                <div class="ml-spacer-stat-key">Detection classes</div>
              </div>
              <div>
                <div class="ml-spacer-stat-val">{{ avgConf(cat) }}%</div>
                <div class="ml-spacer-stat-key">Avg. confidence</div>
              </div>
              <div>
                <div class="ml-spacer-stat-val">RT</div>
                <div class="ml-spacer-stat-key">Processing</div>
              </div>
            </div>
          </div>
        </div>
        <MLCard :cat="cat" :index="i" />
      </template>
    </div>
  </section>
</template>

<script setup>
import { ML_CATEGORIES } from '../data/constants.js'
import MLCard from './MLCard.vue'
import RadarCanvas from './RadarCanvas.vue'

function avgConf(cat) {
  return Math.round(cat.detections.reduce((a, d) => a + d.conf, 0) / cat.detections.length * 100)
}
</script>
