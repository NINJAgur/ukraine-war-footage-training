import { onMounted, onUnmounted } from 'vue'

export function useRadarCanvas(canvasRef, opacity = 0.18, color = '194, 120, 40') {
  let animId = null
  let t = 0
  let points = []

  function initPoints(W, H) {
    points = Array.from({ length: 14 }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.4, vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 3 + 1.5, life: Math.random(),
    }))
  }

  function resize() {
    const canvas = canvasRef.value
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    if (rect.width > 0 && rect.height > 0) {
      canvas.width  = rect.width
      canvas.height = rect.height
      initPoints(rect.width, rect.height)
    }
  }

  function draw() {
    const canvas = canvasRef.value
    if (!canvas) { animId = requestAnimationFrame(draw); return }
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    const W = canvas.width, H = canvas.height
    if (!W || !H) { animId = requestAnimationFrame(draw); return }

    ctx.strokeStyle = `rgba(255,255,255,${opacity * 0.22})`
    ctx.lineWidth = 0.5
    const gs = 64
    for (let gx = 0; gx < W; gx += gs) { ctx.beginPath(); ctx.moveTo(gx,0); ctx.lineTo(gx,H); ctx.stroke() }
    for (let gy = 0; gy < H; gy += gs) { ctx.beginPath(); ctx.moveTo(0,gy); ctx.lineTo(W,gy); ctx.stroke() }

    const scanY = (t * 0.3) % H
    const scanGrad = ctx.createLinearGradient(0, scanY - 60, 0, scanY + 4)
    scanGrad.addColorStop(0, 'transparent')
    scanGrad.addColorStop(1, `rgba(${color}, ${opacity * 0.35})`)
    ctx.fillStyle = scanGrad; ctx.fillRect(0, scanY - 60, W, 64)
    ctx.strokeStyle = `rgba(${color}, ${opacity * 1.1})`; ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(0, scanY); ctx.lineTo(W, scanY); ctx.stroke()

    points.forEach(p => {
      p.x += p.vx; p.y += p.vy; p.life += 0.005
      if (p.x < 0 || p.x > W) p.vx *= -1
      if (p.y < 0 || p.y > H) p.vy *= -1
      const a = (0.25 + 0.15 * Math.sin(p.life * 3)) * opacity * 1.4
      ctx.strokeStyle = `rgba(${color}, ${a})`; ctx.lineWidth = 0.8
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r * 4, 0, Math.PI * 2); ctx.stroke()
      ctx.fillStyle = `rgba(${color}, ${a * 1.5})`
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r * 0.7, 0, Math.PI * 2); ctx.fill()
      ctx.strokeStyle = `rgba(${color}, ${a * 0.5})`; ctx.lineWidth = 0.6
      ctx.beginPath(); ctx.moveTo(p.x-8, p.y); ctx.lineTo(p.x+8, p.y); ctx.moveTo(p.x, p.y-8); ctx.lineTo(p.x, p.y+8); ctx.stroke()
    })

    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        const dx = points[i].x - points[j].x, dy = points[i].y - points[j].y
        const dist = Math.sqrt(dx*dx + dy*dy)
        if (dist < 200) {
          ctx.strokeStyle = `rgba(${color}, ${(1 - dist/200) * 0.07 * opacity * 6})`
          ctx.lineWidth = 0.5
          ctx.beginPath(); ctx.moveTo(points[i].x, points[i].y); ctx.lineTo(points[j].x, points[j].y); ctx.stroke()
        }
      }
    }

    t++
    animId = requestAnimationFrame(draw)
  }

  onMounted(() => {
    setTimeout(() => { resize(); draw() }, 60)
    window.addEventListener('resize', resize)
  })

  onUnmounted(() => {
    if (animId) cancelAnimationFrame(animId)
    window.removeEventListener('resize', resize)
  })
}
