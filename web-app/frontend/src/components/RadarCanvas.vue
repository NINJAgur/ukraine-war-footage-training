<template>
  <canvas ref="canvasEl" style="position:absolute;inset:0;width:100%;height:100%;display:block"></canvas>
</template>

<script setup>
import { ref } from 'vue'
import { useRadarCanvas } from '../composables/useRadarCanvas.js'

const props = defineProps({
  opacity: { type: Number, default: 0.18 },
  color:   { type: String, default: '194, 120, 40' },
})

const canvasEl = ref(null)
useRadarCanvas(canvasEl, props.opacity, props.color)
</script>
