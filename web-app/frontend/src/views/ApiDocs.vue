<template>
  <div class="docs-root">
    <AppNav />
    <main class="docs-main">
      <div class="docs-header">
        <div class="section-tag">API</div>
        <h1 class="docs-title">Public API Reference</h1>
        <p class="docs-sub mono">
          Base URL: <code class="docs-code">https://ukrarchive.duckdns.org</code>
          &nbsp;·&nbsp; No authentication required for public endpoints.
        </p>
      </div>

      <div class="docs-section" v-for="ep in endpoints" :key="ep.method + ep.path">
        <div class="docs-endpoint">
          <span class="docs-method" :class="`method-${ep.method.toLowerCase()}`">{{ ep.method }}</span>
          <code class="docs-path">{{ ep.path }}</code>
        </div>
        <p class="docs-desc mono">{{ ep.desc }}</p>

        <div v-if="ep.params" class="docs-block">
          <div class="docs-block-label mono">Parameters</div>
          <table class="docs-table">
            <tr v-for="p in ep.params" :key="p.name">
              <td class="docs-param-name mono">{{ p.name }}</td>
              <td class="docs-param-type mono">{{ p.type }}</td>
              <td class="docs-param-desc">{{ p.desc }}</td>
            </tr>
          </table>
        </div>

        <div class="docs-block">
          <div class="docs-block-label mono">Example</div>
          <pre class="docs-pre">{{ ep.example }}</pre>
        </div>
      </div>

      <div class="docs-section">
        <div class="docs-endpoint">
          <span class="docs-method method-note">NOTE</span>
          <code class="docs-path">Rate limits &amp; usage</code>
        </div>
        <p class="docs-desc mono">
          The API is open and unauthenticated. Please be reasonable — scraping at high frequency
          will impact site performance for other users. Models are YOLOv8m format (~49 MB each).
          Detection classes: <strong>0 = AIRCRAFT</strong>, <strong>1 = VEHICLE</strong>, <strong>2 = PERSONNEL</strong>.
        </p>
      </div>
    </main>
    <SiteFooter />
  </div>
</template>

<script setup>
import AppNav from '../components/AppNav.vue'
import SiteFooter from '../components/SiteFooter.vue'

const endpoints = [
  {
    method: 'GET',
    path: '/api/models',
    desc: 'List all trained models with mAP50 score, training images, and weight download URL.',
    example: `curl https://ukrarchive.duckdns.org/api/models

[
  {
    "model": "AIRCRAFT",
    "run_id": 68,
    "stage": "FINETUNE",
    "map50": 0.968,
    "images": 65553,
    "download_url": "https://storage.googleapis.com/ukraine-footage-media/runs/...",
    "classes": { "id": 0, "covers": "Drones, helicopters, fixed-wing, missiles" },
    "completed_at": "2026-05-31T09:50:16"
  },
  ...
]`,
  },
  {
    method: 'GET',
    path: '/api/stats',
    desc: 'Live pipeline statistics: clip counts, training images, per-model mAP50 and status.',
    example: `curl https://ukrarchive.duckdns.org/api/stats

{
  "clips_total": 75,
  "clips_annotated": 69,
  "images_labeled": 176000,
  "models": {
    "AIRCRAFT": { "status": "DONE", "map50": 0.968, "images": 65553 },
    "VEHICLE":  { "status": "TRAINING", "map50": 0.904, "images": 56440 },
    ...
  }
}`,
  },
  {
    method: 'GET',
    path: '/api/annotated-clips',
    desc: 'Return all annotated clips with video URL, detection class, source, and description.',
    example: `curl https://ukrarchive.duckdns.org/api/annotated-clips

[
  {
    "id": "a3f2c1b0d9e8",
    "title": "RUSSIAN TANK DESTROYED BY DRONE",
    "description": "FPV drone strike on a T-72 near Avdiivka...",
    "date": "2026-06-01",
    "duration": "00:01:23",
    "detClass": "VEHICLE",
    "source": "Funker530",
    "videoUrl": "https://storage.googleapis.com/ukraine-footage-media/annotated/..."
  },
  ...
]`,
  },
  {
    method: 'GET',
    path: '/api/stats/charts',
    desc: 'Aggregated chart data: clips per day, detection breakdown, mAP50 timeline. (Coming in Phase 5.4)',
    example: `# Not yet available — coming soon`,
  },
]
</script>

<style scoped>
.docs-root { min-height: 100vh; display: flex; flex-direction: column; background: var(--bg-0); }
.docs-main { flex: 1; padding: clamp(60px, 8vw, 100px) clamp(20px, 5vw, 80px) 60px; max-width: 900px; margin: 0 auto; width: 100%; }
.docs-header { margin-bottom: 56px; }
.docs-title { font-size: clamp(28px, 4vw, 48px); font-weight: 700; letter-spacing: -0.02em; margin: 12px 0 16px; }
.docs-sub { font-size: 13px; color: var(--fg-2); line-height: 1.7; }
.docs-code { background: var(--bg-2); padding: 2px 8px; font-family: var(--font-mono); font-size: 12px; color: var(--fg-1); }

.docs-section { border-top: 1px solid var(--fg-3); padding: 36px 0; }
.docs-endpoint { display: flex; align-items: center; gap: 14px; margin-bottom: 12px; }
.docs-method { font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.15em; padding: 4px 10px; border: 1px solid; }
.method-get  { color: #4ade80; border-color: rgba(74,222,128,0.3); }
.method-post { color: var(--amber); border-color: var(--amber-border); }
.method-note { color: var(--fg-3); border-color: var(--fg-3); }
.docs-path { font-family: var(--font-mono); font-size: 15px; color: var(--fg-0); letter-spacing: 0.02em; }
.docs-desc { font-size: 13px; color: var(--fg-2); line-height: 1.7; margin-bottom: 20px; }

.docs-block { margin-bottom: 16px; }
.docs-block-label { font-size: 10px; color: var(--fg-3); letter-spacing: 0.15em; margin-bottom: 8px; text-transform: uppercase; }
.docs-pre {
  background: var(--bg-2); border: 1px solid var(--fg-3);
  padding: 20px; font-family: var(--font-mono); font-size: 12px;
  color: var(--fg-1); overflow-x: auto; line-height: 1.6;
  white-space: pre; tab-size: 2;
}
.docs-table { width: 100%; border-collapse: collapse; }
.docs-table tr { border-bottom: 1px solid var(--fg-3); }
.docs-table td { padding: 8px 12px; font-size: 12px; vertical-align: top; }
.docs-param-name { color: var(--amber); min-width: 120px; }
.docs-param-type { color: var(--fg-3); min-width: 80px; }
.docs-param-desc { color: var(--fg-2); }
</style>
