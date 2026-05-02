<template>
  <div class="ticker-bar">
    <div class="ticker-label">LIVE</div>
    <div style="overflow:hidden;flex:1">
      <div class="ticker-track">
        <span v-for="(item, i) in doubled" :key="i" class="ticker-item">
          <span class="ticker-sep">◆</span>{{ item }}
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { TICKER_ITEMS } from '../data/constants.js'
const doubled = [...TICKER_ITEMS, ...TICKER_ITEMS]
</script>
