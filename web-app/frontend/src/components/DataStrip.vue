<template>
  <div class="data-strip">
    <div v-for="d in items" :key="d.label" class="data-strip-item">
      <div class="data-big">{{ d.big }}<span>{{ d.unit }}</span></div>
      <div class="data-label">{{ d.label }}</div>
      <div class="data-sublabel">{{ d.sub }}</div>
    </div>
  </div>
</template>

<script setup>
const items = [
  { big: '9',   unit: '',   label: 'Total clips archived',  sub: 'As of May 2026' },
  { big: '25',  unit: 'GB', label: 'Raw footage stored',    sub: 'From Russian, Ukranian and Independant sources' },
  { big: '178', unit: 'K',  label: 'Images Analyzed',       sub: 'Labelled and Annotated' },
]
</script>
