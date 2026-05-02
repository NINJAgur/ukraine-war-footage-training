<template>
  <section class="mission">
    <div class="section-header">
      <div>
        <div class="section-tag">Mission</div>
        <h2 class="section-title">Why this archive exists</h2>
      </div>
    </div>
    <div class="mission-grid">
      <template v-for="(item, i) in cols" :key="i">
        <div v-if="item === null" class="mission-divider"></div>
        <div v-else class="mission-col">
          <div class="mission-col-num">[ {{ item.num }} ]</div>
          <h3>{{ item.title }}</h3>
          <p>{{ item.body }}</p>
        </div>
      </template>
    </div>
  </section>
</template>

<script setup>
const cols = [
  {
    num: 'I',
    title: 'Document',
    body: 'Footage scraped from open-source channels is automatically labeled and stored. Every clip captures equipment, personnel, and events that would otherwise be lost in the feed.',
  },
  null,
  {
    num: 'II',
    title: 'Detect',
    body: 'Specialist YOLO models identify Aircraft, Vehicles, and Personnel in each frame. Trained on 5 Kaggle datasets and continuously improved as new annotated footage accumulates.',
  },
  null,
  {
    num: 'III',
    title: 'Archive',
    body: 'Annotated MP4s are stored and browsable. Researchers and analysts can filter by detection class or source channel. The goal is a structured, searchable dataset — not a media feed.',
  },
]
</script>
