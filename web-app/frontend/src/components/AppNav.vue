<template>
  <nav :style="{ boxShadow: scrolled ? '0 1px 0 rgba(255,255,255,0.04)' : 'none' }">
    <router-link to="/" class="nav-logo" style="text-decoration:none;cursor:pointer">
      <div class="nav-logo-mark"></div>
      UKRARCHIVE
      <span class="mono" style="color: var(--fg-3); font-size: 10px">v1.0.0</span>
    </router-link>
    <ul class="nav-links">
      <li v-for="id in sections" :key="id">
        <a :href="`/#${id}`" :class="{ active: activeSection === id }">
          {{ id.charAt(0).toUpperCase() + id.slice(1) }}
        </a>
      </li>
    </ul>
    <div class="nav-right">
      <div class="nav-status">
        <div class="status-dot"></div>
        COLLECTION ACTIVE
      </div>
      <router-link to="/admin/login" class="btn-access">Admin Login</router-link>
    </div>
  </nav>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const sections = ['archive', 'detection', 'capabilities', 'about']
const scrolled = ref(false)
const activeSection = ref('archive')

function onScroll() {
  scrolled.value = window.scrollY > 20
  let current = 'archive'
  for (const id of sections) {
    const el = document.getElementById(id)
    if (el && el.getBoundingClientRect().top <= 80) current = id
  }
  activeSection.value = current
}

onMounted(() => window.addEventListener('scroll', onScroll, { passive: true }))
onUnmounted(() => window.removeEventListener('scroll', onScroll))
</script>
