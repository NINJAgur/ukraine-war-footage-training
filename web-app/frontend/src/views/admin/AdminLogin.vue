<template>
  <main class="login-root">
    <div class="login-card">
      <div class="login-header">
        <div class="nav-logo-mark"></div>
        <div class="login-title mono">UKRARCHIVE // ADMIN</div>
        <div class="login-sub mono">RESTRICTED ACCESS</div>
      </div>

      <form class="login-form" @submit.prevent="submit">
        <div class="login-field">
          <label class="login-label mono">USERNAME</label>
          <input
            v-model="username"
            type="text"
            class="login-input mono"
            autocomplete="username"
            :disabled="loading"
          />
        </div>
        <div class="login-field">
          <label class="login-label mono">PASSWORD</label>
          <input
            v-model="password"
            type="password"
            class="login-input mono"
            autocomplete="current-password"
            :disabled="loading"
          />
        </div>

        <div v-if="error" class="login-error mono">{{ error }}</div>

        <button type="submit" class="btn-primary login-btn" :disabled="loading">
          {{ loading ? 'AUTHENTICATING...' : 'AUTHENTICATE' }}
        </button>
      </form>

      <div class="login-footer mono">
        <router-link to="/">← RETURN TO ARCHIVE</router-link>
      </div>
    </div>
  </main>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const username = ref('')
const password = ref('')
const loading  = ref(false)
const error    = ref('')

async function submit() {
  error.value   = ''
  loading.value = true
  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: username.value, password: password.value }),
    })
    if (!res.ok) {
      error.value = res.status === 401 ? 'INVALID CREDENTIALS' : `ERROR ${res.status}`
      return
    }
    const { access_token } = await res.json()
    localStorage.setItem('token', access_token)
    router.push('/admin')
  } catch {
    error.value = 'CONNECTION FAILED'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-root {
  min-height: 100vh;
  background: var(--bg-0);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}

.login-card {
  width: 100%;
  max-width: 400px;
  border: 1px solid var(--fg-3);
  background: var(--bg-1);
  padding: 48px 40px;
  display: flex;
  flex-direction: column;
  gap: 40px;
}

.login-header {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
}

.nav-logo-mark {
  width: 36px;
  height: 36px;
  border: 1.5px solid var(--amber);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}
.nav-logo-mark::before {
  content: '';
  position: absolute;
  width: 10px;
  height: 10px;
  background: var(--amber);
  clip-path: polygon(50% 0%, 100% 100%, 0% 100%);
}

.login-title {
  font-size: 14px;
  letter-spacing: 0.2em;
  color: var(--fg-0);
}

.login-sub {
  font-size: 10px;
  letter-spacing: 0.25em;
  color: var(--amber);
}

.login-form {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.login-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.login-label {
  font-size: 10px;
  letter-spacing: 0.2em;
  color: var(--fg-2);
}

.login-input {
  background: var(--bg-0);
  border: 1px solid var(--fg-3);
  color: var(--fg-0);
  font-size: 13px;
  padding: 10px 14px;
  outline: none;
  transition: border-color 0.2s;
  width: 100%;
}
.login-input:focus {
  border-color: var(--amber-border);
}
.login-input:disabled {
  opacity: 0.5;
}

.login-error {
  font-size: 11px;
  letter-spacing: 0.15em;
  color: var(--red);
  padding: 8px 12px;
  border: 1px solid var(--red-dim);
  background: rgba(210, 40, 40, 0.05);
}

.login-btn {
  width: 100%;
  padding: 14px;
  margin-top: 4px;
}
.login-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.login-footer {
  text-align: center;
  font-size: 10px;
  letter-spacing: 0.15em;
  color: var(--fg-3);
}
.login-footer a:hover {
  color: var(--amber);
}
</style>
