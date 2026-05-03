<template>
  <section class="about" id="about">
    <div>
      <div class="section-tag">About</div>
      <h2 class="section-title" style="margin-bottom:32px">One person.<br>One archive.</h2>
      <div class="about-body">
        <p>
          UKRARCHIVE is an independent, solo-maintained open-source project documenting
          the conflict in Ukraine through verified footage and geospatial data.
          Started in February 2022, it runs as a GitHub Pages site — no institution,
          no funding, no team.
        </p>
        <p>
          Every clip is manually reviewed. Verification follows open OSINT standards:
          geolocation, shadow/solar analysis, cross-referencing against satellite imagery.
          The goal is a durable, honest record — not a media platform.
        </p>
      </div>
    </div>
    <div class="about-right">
      <div class="section-tag" style="margin-bottom:20px">Project</div>
      <ul class="about-list">
        <li v-for="row in rows" :key="row.label">
          <span class="list-marker" style="min-width:120px;color:var(--fg-2);font-size:11px">{{ row.label }}</span>
          <span>{{ row.val }}</span>
        </li>
      </ul>
      <a href="https://github.com/NINJAgur/ukraine-war-footage-training" target="_blank" rel="noopener" style="display:inline-flex;align-items:center;gap:10px;margin-top:32px;font-family:var(--font-mono);font-size:11px;letter-spacing:0.12em;text-transform:uppercase;color:var(--amber);border-bottom:1px solid var(--amber-border);padding-bottom:4px">
        View on GitHub →
      </a>
    </div>
  </section>
</template>

<script setup>
const rows = [
  { label: 'Source code',  val: 'github.com/NINJAgur/ukraine-war-footage-training' },
  { label: 'Hosted on',    val: 'Self-hosted (FastAPI + Vue 3)' },
  { label: 'License',      val: 'Open Evidence License' },
  { label: 'Started',      val: 'February 24, 2022' },
  { label: 'Maintained by', val: '1 person' },
  { label: 'Submissions',  val: 'Open — secure upload' },
]
</script>
