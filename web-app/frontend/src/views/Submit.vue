<template>
  <div>
    <AppNav />

    <div class="submit-wrap">
      <div class="submit-box">
        <div class="submit-tag mono">SUBMIT FOOTAGE</div>
        <h1 class="submit-title">Contribute to the archive</h1>
        <p class="submit-desc">
          Submit a link to verified combat footage for review.
          Accepted sources: Telegram, YouTube, Twitter/X, direct video URLs.
        </p>

        <form class="submit-form" @submit.prevent="submit">
          <div class="field">
            <label class="field-label mono">VIDEO URL <span class="req">*</span></label>
            <input
              v-model="url"
              class="field-input"
              type="url"
              placeholder="https://t.me/... or https://youtube.com/..."
              required
              :disabled="loading || done"
            />
          </div>

          <div class="field">
            <label class="field-label mono">TITLE <span class="opt">optional</span></label>
            <input
              v-model="title"
              class="field-input"
              type="text"
              placeholder="Brief description of the clip"
              :disabled="loading || done"
            />
          </div>

          <div class="field">
            <label class="field-label mono">NOTES <span class="opt">optional</span></label>
            <textarea
              v-model="description"
              class="field-input field-textarea"
              placeholder="Location, date, unit IDs, geolocation confidence..."
              rows="4"
              :disabled="loading || done"
            />
          </div>

          <div v-if="error" class="submit-error mono">{{ error }}</div>

          <div v-if="done" class="submit-success mono">
            SUBMITTED — CLIP QUEUED FOR REVIEW
          </div>

          <div class="submit-actions">
            <button
              type="submit"
              class="btn-submit mono"
              :disabled="loading || done || !url"
            >
              {{ loading ? 'SUBMITTING...' : done ? 'SUBMITTED' : 'SUBMIT CLIP' }}
            </button>
            <a href="/archive" class="btn-back mono">← ARCHIVE</a>
          </div>
        </form>
      </div>
    </div>

    <SiteFooter />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import AppNav    from '../components/AppNav.vue'
import SiteFooter from '../components/SiteFooter.vue'

const url         = ref('')
const title       = ref('')
const description = ref('')
const loading     = ref(false)
const done        = ref(false)
const error       = ref('')

async function submit() {
  loading.value = true
  error.value   = ''
  try {
    const res = await fetch('/api/submit', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url:         url.value,
        title:       title.value || null,
        description: description.value || null,
      }),
    })
    if (res.status === 201) {
      done.value = true
    } else if (res.status === 409) {
      error.value = 'CLIP ALREADY IN ARCHIVE'
    } else {
      const d = await res.json().catch(() => ({}))
      error.value = d.detail ?? `ERROR ${res.status}`
    }
  } catch {
    error.value = 'NETWORK ERROR — PLEASE RETRY'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.submit-wrap {
  min-height: calc(100vh - 64px);
  padding-top: 64px;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding-bottom: 80px;
}

.submit-box {
  width: 100%;
  max-width: 600px;
  padding: 60px 32px 0;
}

.submit-tag {
  font-size: 10px;
  letter-spacing: 0.2em;
  color: var(--amber);
  text-transform: uppercase;
  margin-bottom: 12px;
}

.submit-title {
  font-size: clamp(24px, 4vw, 36px);
  font-weight: 300;
  letter-spacing: -0.02em;
  color: var(--fg-0);
  margin: 0 0 16px;
}

.submit-desc {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--fg-3);
  line-height: 1.7;
  letter-spacing: 0.04em;
  margin: 0 0 40px;
}

.submit-form { display: flex; flex-direction: column; gap: 24px; }

.field { display: flex; flex-direction: column; gap: 8px; }

.field-label {
  font-size: 10px;
  letter-spacing: 0.15em;
  color: var(--fg-3);
  text-transform: uppercase;
}

.req { color: var(--amber); }
.opt { color: var(--fg-3); font-size: 9px; letter-spacing: 0.08em; }

.field-input {
  background: var(--bg-2);
  border: 1px solid rgba(255,255,255,0.08);
  color: var(--fg-1);
  font-family: var(--font-mono);
  font-size: 12px;
  padding: 10px 14px;
  outline: none;
  letter-spacing: 0.04em;
  transition: border-color 0.15s;
  resize: none;
}
.field-input:focus { border-color: var(--amber); }
.field-input:disabled { opacity: 0.4; cursor: not-allowed; }
.field-textarea { line-height: 1.6; }

.submit-error {
  font-size: 11px;
  letter-spacing: 0.1em;
  color: #ef4444;
  padding: 10px 14px;
  border: 1px solid rgba(239,68,68,0.3);
  background: rgba(239,68,68,0.05);
}

.submit-success {
  font-size: 11px;
  letter-spacing: 0.1em;
  color: #22c55e;
  padding: 10px 14px;
  border: 1px solid rgba(34,197,94,0.3);
  background: rgba(34,197,94,0.05);
}

.submit-actions {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-top: 8px;
}

.btn-submit {
  background: var(--amber);
  color: #000;
  border: none;
  padding: 10px 28px;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  cursor: pointer;
  transition: opacity 0.15s;
}
.btn-submit:hover:not(:disabled) { opacity: 0.85; }
.btn-submit:disabled { opacity: 0.35; cursor: not-allowed; }

.btn-back {
  font-size: 10px;
  letter-spacing: 0.12em;
  color: var(--fg-3);
  text-decoration: none;
  text-transform: uppercase;
  transition: color 0.15s;
}
.btn-back:hover { color: var(--amber); }
</style>
