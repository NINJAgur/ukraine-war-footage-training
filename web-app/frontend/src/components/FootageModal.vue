<template>
  <div class="modal-backdrop" @click="$emit('close')">
    <div class="modal-panel" @click.stop>
      <div class="modal-header">
        <div style="min-width:0;flex:1;overflow:hidden;margin-right:20px">
          <div style="font-family:var(--font-mono);font-size:10px;letter-spacing:0.15em;color:var(--amber);text-transform:uppercase;margin-bottom:4px">
            {{ item.src }}
          </div>
          <div style="font-size:16px;font-weight:600;text-transform:uppercase;letter-spacing:-0.01em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
            {{ item.title }}
          </div>
        </div>
        <button class="modal-close" style="flex-shrink:0" @click="$emit('close')">[ CLOSE ]</button>
      </div>
      <div class="modal-body">
        <div class="modal-video-placeholder">
          <video
            v-if="item.videoUrl"
            ref="videoEl"
            :src="item.videoUrl"
            class="modal-video"
            controls
            muted
            playsinline
            preload="auto"
            @canplay="tryPlay"
          />
          <div v-else class="modal-video-label">
            ANNOTATED FOOTAGE<br>{{ item.detClass }} — {{ item.source }}
          </div>
        </div>
        <div class="modal-meta-grid">
          <div v-for="cell in metaCells" :key="cell.k" class="modal-meta-cell">
            <div class="modal-meta-key">{{ cell.k }}</div>
            <div class="modal-meta-val">{{ cell.v }}</div>
          </div>
        </div>
        <p class="modal-desc">
          {{ item.description || 'No description available.' }}
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
const props = defineProps({ item: { type: Object, required: true } })
defineEmits(['close'])

const videoEl = ref(null)
function tryPlay() {
  videoEl.value?.play().catch(() => {})
}

const metaCells = computed(() => [
  { k: 'Source',     v: props.item.source },
  { k: 'Date',       v: props.item.date },
  { k: 'Duration',   v: props.item.duration },
  { k: 'Detection',  v: props.item.detClass },
  { k: 'Source ID',  v: props.item.src },
  { k: 'Status',     v: props.item.tag.toUpperCase() },
])
</script>
