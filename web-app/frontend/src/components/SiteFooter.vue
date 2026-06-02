<template>
  <footer id="access">
    <div class="footer-top">
      <div class="footer-brand">
        <a href="/" class="footer-logo" style="text-decoration:none">UKRARCHIVE</a>
        <div class="footer-tagline">Open military asset detection models.<br>Continuously trained on real conflict footage.</div>
      </div>

      <div class="footer-col">
        <h4>Site</h4>
        <ul>
          <li><a href="/archive">Browse Archive</a></li>
          <li><a href="/archive?class=AIRCRAFT">Aircraft</a></li>
          <li><a href="/archive?class=VEHICLE">Vehicles</a></li>
          <li><a href="/archive?class=PERSONNEL">Personnel</a></li>
          <li><a href="/submit">Submit Footage</a></li>
          <li><a href="/admin">Admin Panel</a></li>
        </ul>
      </div>

      <div class="footer-col">
        <h4>About</h4>
        <ul>
          <li v-for="row in rows" :key="row.label" class="footer-stat-row">
            <span class="footer-stat-label">{{ row.label }}</span>
            <component :is="row.href ? 'a' : 'span'" style="font-size:13px;color:var(--fg-2);font-weight:400" v-bind="row.href ? { href: row.href, target: '_blank', rel: 'noopener' } : {}">{{ row.val }}</component>
          </li>
        </ul>
      </div>

      <div class="footer-col">
        <h4>Models</h4>
        <ul>
          <li><a href="#">AIRCRAFT — 0.968 mAP50</a></li>
          <li><a href="#">VEHICLE — 0.904 mAP50</a></li>
          <li><a href="#">PERSONNEL — 0.873 mAP50</a></li>
          <li><a href="#">GENERAL — 0.784 mAP50</a></li>
        </ul>
      </div>
    </div>

    <div class="footer-bottom">
      <div class="footer-copy">© 2026 UKRARCHIVE — Open Evidence License. Non-partisan. Non-commercial.</div>
      <div class="footer-copy">Built on <a href="https://github.com/computer-vision-with-marco/yolo-training-template" target="_blank" rel="noopener" style="color:var(--fg-3);text-decoration:underline">YOLO Training Template</a> by Marco Franzon</div>
    </div>
  </footer>
</template>

<script setup>
const rows = [
  { label: 'Source code',   val: 'github.com/NINJAgur',         href: 'https://github.com/NINJAgur/ukraine-war-footage-training' },
  { label: 'Hosted on',     val: 'Google Cloud Platform',        href: null },
  { label: 'Stack',         val: 'FastAPI · Vue 3 · YOLOv8',    href: null },
  { label: 'License',       val: 'Open Evidence License',        href: null },
  { label: 'Started',       val: 'April 2026',                   href: null },
]
</script>

<style scoped>
.footer-stat-row {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.footer-stat-label {
  font-size: 10px;
  color: var(--fg-3);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.footer-stat-val {
  font-size: 13px;
  color: var(--fg-2);
  font-weight: 400;
}
</style>
