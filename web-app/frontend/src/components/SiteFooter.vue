<template>
  <footer id="access">
    <div class="footer-main">
      <div class="footer-brand-col">
        <a href="/" class="footer-logo" style="text-decoration:none">UKRARCHIVE</a>
        <div class="footer-tagline">Open military asset detection models.<br>Continuously trained on real conflict footage.</div>
        <router-link to="/submit" class="btn-footer-submit">Submit Footage →</router-link>
      </div>

      <div class="footer-about-col">
        <div class="section-tag" style="margin-bottom:20px">Project</div>
        <ul class="about-list">
          <li v-for="row in rows" :key="row.label">
            <span class="list-marker" style="min-width:120px;color:var(--fg-2);font-size:11px">{{ row.label }}</span>
            <span>{{ row.val }}</span>
          </li>
        </ul>
        <a href="https://github.com/NINJAgur/ukraine-war-footage-training" target="_blank" rel="noopener" class="footer-github">
          View on GitHub →
        </a>
      </div>
    </div>

    <div class="footer-bottom">
      <div class="footer-copy">© 2026 UKRARCHIVE — Open Evidence License. Non-partisan. Non-commercial.</div>
      <div class="footer-copy" style="color:var(--fg-3)">Built on <a href="https://github.com/computer-vision-with-marco/yolo-training-template" target="_blank" rel="noopener" style="color:var(--fg-3);text-decoration:underline">YOLO Training Template</a> by Marco Franzon</div>
    </div>
  </footer>
</template>

<script setup>
const rows = [
  { label: 'Source code',    val: 'github.com/NINJAgur/ukraine-war-footage-training' },
  { label: 'Hosted on',      val: 'Google Cloud Platform (FastAPI + Vue 3)' },
  { label: 'ML models',      val: 'YOLOv8m — AIRCRAFT · VEHICLE · PERSONNEL · GENERAL' },
  { label: 'License',        val: 'Open Evidence License' },
  { label: 'Started',        val: 'April 2026' },
  { label: 'Maintained by',  val: '1 person' },
]
</script>

<style scoped>
.footer-main {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 60px;
  margin-bottom: 48px;
}
.footer-brand-col {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.btn-footer-submit {
  display: inline-block;
  margin-top: 8px;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--amber);
  border: 1px solid var(--amber-border);
  padding: 10px 20px;
  text-decoration: none;
  width: fit-content;
  transition: background 0.2s;
}
.btn-footer-submit:hover { background: var(--amber-glow); }
.footer-github {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  margin-top: 24px;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--amber);
  border-bottom: 1px solid var(--amber-border);
  padding-bottom: 4px;
  text-decoration: none;
}
.footer-github:hover { border-color: var(--amber); }

@media (max-width: 768px) {
  .footer-main { grid-template-columns: 1fr; gap: 40px; }
}
</style>
