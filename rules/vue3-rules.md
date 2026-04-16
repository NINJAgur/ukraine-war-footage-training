# Vue 3 Rules — Ukraine Combat Footage Project

These rules are enforced on all code in `web-app/frontend/`.
Violations must be corrected before merging.

---

## MANDATORY RULES

### 1. Always Use `<script setup>` Syntax
```vue
<!-- CORRECT -->
<script setup>
import { ref, computed, onMounted } from 'vue'
const count = ref(0)
</script>

<!-- WRONG — Options API is banned -->
<script>
export default {
  data() { return { count: 0 } }
}
</script>
```

### 2. No `this` Keyword
The Composition API eliminates `this`. Any use of `this` signals incorrect code.

### 3. State Management: Pinia Only
- Use `defineStore` with Composition API style
- No Vuex, no `provide/inject` for cross-component global state
```js
// CORRECT
export const useFeedStore = defineStore('feed', () => {
  const clips = ref([])
  const fetchFeed = async () => {
    clips.value = await apiFeed()
  }
  return { clips, fetchFeed }
})

// WRONG — Options-style Pinia
export const useFeedStore = defineStore('feed', {
  state: () => ({ clips: [] })
})
```

### 4. JWT Token Storage: Pinia Only (Never localStorage)
```js
// CORRECT — token lives only in memory (Pinia store)
const authStore = useAuthStore()
authStore.token = response.data.access_token

// WRONG — persists across sessions, vulnerable to XSS
localStorage.setItem('token', token)
```

### 5. No `v-html` with User-Provided Content
```vue
<!-- CORRECT -->
<p>{{ clip.description }}</p>

<!-- WRONG — XSS vulnerability -->
<p v-html="clip.description"></p>
```

### 6. Component Props via `defineProps()`
```vue
<script setup>
const props = defineProps({
  clip: { type: Object, required: true },
  showBadge: { type: Boolean, default: false }
})
</script>
```

### 7. Events via `defineEmits()`
```vue
<script setup>
const emit = defineEmits(['select', 'dismiss'])
const handleSelect = () => emit('select', props.clip.id)
</script>
```

### 8. Tailwind Only — No Inline Styles
```vue
<!-- CORRECT -->
<div class="bg-zinc-900 border border-zinc-800 rounded-lg p-4">

<!-- WRONG -->
<div style="background: #18181b; border: 1px solid #27272a;">
```

### 9. API Calls in Stores, Not in Components
```js
// CORRECT — store action
const useFeedStore = defineStore('feed', () => {
  const fetchFeed = async () => { ... }
  return { fetchFeed }
})

// WRONG — direct fetch in component onMounted
onMounted(async () => {
  const res = await fetch('/api/feed')  // put this in a store action
})
```

### 10. Clean Up Side Effects in `onUnmounted`
```js
// CORRECT
let interval
onMounted(() => { interval = setInterval(refresh, 60000) })
onUnmounted(() => clearInterval(interval))

// Also for WebSockets:
onMounted(() => { ws = new WebSocket(url) })
onUnmounted(() => ws?.close())
```

---

## Dark Tactical Theme Conventions

| Element | Tailwind Classes |
|---------|-----------------|
| Page background | `bg-zinc-950` |
| Card | `bg-zinc-900 border border-zinc-800 rounded-lg` |
| Primary text | `text-zinc-100` |
| Secondary/muted text | `text-zinc-400` |
| Metadata (monospace) | `font-mono text-xs text-zinc-400` |
| Accent (green) | `text-green-500` / `bg-green-500` |
| Danger (red) | `text-red-500` / `bg-red-500` |
| Active/focus ring | `ring-2 ring-green-500` |
| Button (primary) | `bg-green-600 hover:bg-green-500 text-white px-4 py-2 rounded` |
| Button (danger) | `bg-red-600 hover:bg-red-500 text-white px-4 py-2 rounded` |
| Input | `bg-zinc-800 border border-zinc-700 text-zinc-100 rounded px-3 py-2 focus:ring-2 focus:ring-green-500` |
