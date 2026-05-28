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

    <!-- Mobile hamburger button -->
    <button class="nav-hamburger" :class="{ open: menuOpen }" @click="menuOpen = !menuOpen" aria-label="Menu">
      <span></span>
      <span></span>
      <span></span>
    </button>
  </nav>

  <!-- Mobile menu (outside nav so it overlays full page) -->
  <Teleport to="body">
    <div v-if="menuOpen" class="nav-mobile-backdrop" @click="menuOpen = false">
      <div class="nav-mobile-panel" @click.stop>
        <div class="nav-mobile-status">
          <div class="status-dot"></div>
          <span class="mono" style="font-size:10px;letter-spacing:0.15em">COLLECTION ACTIVE</span>
        </div>
        <ul class="nav-mobile-links">
          <li v-for="id in sections" :key="id">
            <a :href="`/#${id}`" :class="{ active: activeSection === id }" @click="menuOpen = false">
              {{ id.charAt(0).toUpperCase() + id.slice(1) }}
            </a>
          </li>
        </ul>
        <router-link to="/admin/login" class="nav-mobile-admin" @click="menuOpen = false">
          Admin Login
        </router-link>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const sections = ['archive', 'detection', 'capabilities', 'about']
const scrolled = ref(false)
const activeSection = ref('archive')
const menuOpen = ref(false)

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

<style scoped>
.nav-hamburger {
  display: none;
  flex-direction: column;
  justify-content: center;
  gap: 5px;
  width: 36px;
  height: 36px;
  background: none;
  border: 1px solid var(--fg-3);
  cursor: pointer;
  padding: 0 8px;
  flex-shrink: 0;
}
.nav-hamburger span {
  display: block;
  width: 100%;
  height: 1.5px;
  background: var(--amber);
  transition: transform 0.2s ease, opacity 0.2s ease;
  transform-origin: center;
}
.nav-hamburger.open span:nth-child(1) { transform: translateY(6.5px) rotate(45deg); }
.nav-hamburger.open span:nth-child(2) { opacity: 0; }
.nav-hamburger.open span:nth-child(3) { transform: translateY(-6.5px) rotate(-45deg); }

.nav-mobile-backdrop {
  position: fixed;
  inset: 0;
  top: 56px;
  z-index: 999;
  background: rgba(0,0,0,0.6);
  backdrop-filter: blur(4px);
}
.nav-mobile-panel {
  background: var(--bg-1);
  border-bottom: 1px solid var(--fg-3);
  padding: 24px clamp(20px, 5vw, 40px) 32px;
  display: flex;
  flex-direction: column;
  gap: 0;
}
.nav-mobile-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--fg-3);
  margin-bottom: 8px;
  color: var(--fg-2);
}
.nav-mobile-links {
  list-style: none;
  display: flex;
  flex-direction: column;
}
.nav-mobile-links li a {
  display: block;
  font-family: var(--font-mono);
  font-size: 13px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--fg-1);
  padding: 14px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  transition: color 0.15s;
  text-decoration: none;
}
.nav-mobile-links li a:hover,
.nav-mobile-links li a.active { color: var(--amber); }
.nav-mobile-admin {
  display: block;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--amber);
  border: 1px solid var(--amber-border);
  padding: 12px 20px;
  text-align: center;
  text-decoration: none;
  margin-top: 20px;
  transition: background 0.2s;
}
.nav-mobile-admin:hover { background: var(--amber-glow); }

@media (max-width: 768px) {
  .nav-hamburger { display: flex; }
}
</style>
