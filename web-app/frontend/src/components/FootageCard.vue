<template>
  <div class="footage-card" @click="$emit('open', item)">
    <div class="card-thumb">
      <video
        v-if="item.videoUrl"
        ref="videoEl"
        :src="item.videoUrl"
        class="card-video"
        muted playsinline
        preload="metadata"
        @mouseenter="videoEl.play()"
        @mouseleave="videoEl.pause(); videoEl.currentTime = 0"
        @click.stop="$emit('open', item)"
      />
      <div v-else class="card-thumb-placeholder">
        <div class="card-thumb-label">{{ item.detClass }}<br>{{ item.src }}</div>
      </div>
    </div>
    <div class="card-overlay"></div>
    <div v-if="!item.videoUrl" class="card-play"></div>
    <div :class="`card-tag tag-${item.tag}`">{{ item.tag }}</div>
    <div class="card-meta-bar">
      <div class="card-location">{{ item.source }}</div>
      <div class="card-title">{{ item.title }}</div>
      <div class="card-data-row">
        <span class="card-datum">{{ item.date }}</span>
        <span class="card-datum">{{ item.duration }}</span>
        <span class="card-datum" :style="classColor(item.detClass)">{{ item.detClass }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

defineProps({ item: { type: Object, required: true } })
defineEmits(['open'])

const videoEl = ref(null)

function classColor(cls) {
  const map = { AIRCRAFT: 'color:oklch(0.62 0.16 220deg)', VEHICLE: 'color:oklch(0.60 0.20 25deg)', PERSONNEL: 'color:oklch(0.60 0.18 145deg)' }
  return map[cls] || ''
}
</script>
