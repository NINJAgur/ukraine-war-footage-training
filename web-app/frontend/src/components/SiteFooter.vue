<template>
  <footer id="access">
    <div class="footer-top">
      <div class="footer-brand">
        <a href="/" class="footer-logo" style="text-decoration:none">UKRARCHIVE</a>
        <div class="footer-tagline">Open military asset detection models. Continuously trained on real conflict footage.</div>
      </div>
      <div class="footer-col">
        <h4>Project</h4>
        <ul>
          <li><span class="footer-label">Source code</span><br><a href="https://github.com/NINJAgur/ukraine-war-footage-training" target="_blank" rel="noopener">github.com/NINJAgur</a></li>
          <li><span class="footer-label">Hosted on</span><br><span>Google Cloud Platform</span></li>
          <li><span class="footer-label">Stack</span><br><span>FastAPI · Vue 3 · YOLOv8</span></li>
          <li><span class="footer-label">License</span><br><span>Open Evidence License</span></li>
        </ul>
      </div>
      <div class="footer-col">
        <h4>Links</h4>
        <ul>
          <li><a href="/archive">Browse Archive</a></li>
          <li><a href="/submit">Submit Footage</a></li>
          <li><a href="https://github.com/NINJAgur/ukraine-war-footage-training" target="_blank" rel="noopener">Source Code</a></li>
          <li><a href="/admin">Admin Panel</a></li>
        </ul>
      </div>
    </div>
    <div class="footer-bottom">
      <div class="footer-copy">© 2026 UKRARCHIVE — Open Evidence License. Non-partisan. Non-commercial.</div>
      <div class="footer-copy">Built on <a href="https://github.com/computer-vision-with-marco/yolo-training-template" target="_blank" rel="noopener" style="color:var(--fg-3);text-decoration:underline">YOLO Training Template</a> by Marco Franzon</div>
    </div>
  </footer>
</template>

<style scoped>
.footer-label {
  font-size: 10px;
  color: var(--fg-3);
  letter-spacing: 0.08em;
}
</style>
