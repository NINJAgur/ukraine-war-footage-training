import { onMounted, onUnmounted, watch } from 'vue'
import { CAT_RGB } from '../data/constants.js'

export function useMLCanvas(canvasRef, catId, detectionsRef, expandedRef) {
  let animId = null
  let frameCount = 0
  let ro = null

  function resize() {
    const canvas = canvasRef.value
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    if (rect.width > 0 && rect.height > 0) {
      canvas.width  = rect.width
      canvas.height = rect.height
    }
  }

  function draw() {
    frameCount++
    const f = frameCount
    const canvas = canvasRef.value
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    if (!W || !H) { animId = requestAnimationFrame(draw); return }
    ctx.clearRect(0, 0, W, H)

    const rgb = CAT_RGB[catId] || '194, 120, 40'
    const detections = detectionsRef.value
    const expanded = expandedRef.value

    // background grid
    ctx.strokeStyle = 'rgba(255,255,255,0.03)'
    ctx.lineWidth = 0.5
    const gs = 48
    for (let x = 0; x < W; x += gs) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke() }
    for (let y = 0; y < H; y += gs) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke() }

    if (catId === 'generalist') {
      drawGeneralist(ctx, detections, W, H, f, rgb, expanded)
    } else if (catId === 'aircraft') {
      drawAircraft(ctx, detections, W, H, f, rgb, expanded)
    } else if (catId === 'personnel') {
      drawPersonnel(ctx, detections, W, H, f, rgb)
    } else {
      drawVehicles(ctx, detections, W, H, f, rgb, expanded)
    }

    animId = requestAnimationFrame(draw)
  }

  onMounted(() => {
    setTimeout(() => {
      resize()
      draw()
    }, 60)
    ro = new ResizeObserver(resize)
    if (canvasRef.value?.parentElement) {
      ro.observe(canvasRef.value.parentElement)
    }
  })

  onUnmounted(() => {
    if (animId) cancelAnimationFrame(animId)
    if (ro) ro.disconnect()
  })
}

function drawGeneralist(ctx, dets, W, H, f, rgb, expanded) {
  dets.forEach((det, i) => {
    const appear = Math.max(0, Math.min(1, (f - i * 8) / 20))
    if (appear === 0) return
    const x = det.x * W, y = det.y * H, w = det.w * W, h = det.h * H
    const jx = expanded ? Math.sin(f * 0.03 + i) * 0.8 : 0
    const jy = expanded ? Math.cos(f * 0.04 + i) * 0.8 : 0
    const rx = x + jx, ry = y + jy
    const alpha = appear * (0.75 + 0.15 * Math.sin(f * 0.05 + i))
    ctx.fillStyle = `rgba(${rgb}, ${alpha * 0.07})`; ctx.fillRect(rx, ry, w, h)
    ctx.strokeStyle = `rgba(${rgb}, ${alpha * 0.9})`; ctx.lineWidth = 1.2; ctx.strokeRect(rx, ry, w, h)
    const tk = 8
    ctx.strokeStyle = `rgba(${rgb}, ${alpha})`; ctx.lineWidth = 2
    ctx.beginPath(); ctx.moveTo(rx,ry+tk); ctx.lineTo(rx,ry); ctx.lineTo(rx+tk,ry); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(rx+w-tk,ry); ctx.lineTo(rx+w,ry); ctx.lineTo(rx+w,ry+tk); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(rx,ry+h-tk); ctx.lineTo(rx,ry+h); ctx.lineTo(rx+tk,ry+h); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(rx+w-tk,ry+h); ctx.lineTo(rx+w,ry+h); ctx.lineTo(rx+w,ry+h-tk); ctx.stroke()
    const bh = h * det.conf
    ctx.fillStyle = `rgba(${rgb},${alpha*0.18})`; ctx.fillRect(rx-4, ry+h-bh, 3, bh)
    ctx.fillStyle = `rgba(${rgb},${alpha*0.6})`; ctx.fillRect(rx-4, ry+h-bh, 3, 2)
    ctx.font = '10px monospace'; ctx.fillStyle = `rgba(${rgb},${alpha*0.9})`
    ctx.fillText(`${det.cls} · ${Math.round(det.conf*100)}%`, rx+2, ry-4)
  })
}

function drawAircraft(ctx, dets, W, H, f, rgb) {
  dets.forEach((det, i) => {
    const appear = Math.max(0, Math.min(1, (f - i * 6) / 18))
    if (appear === 0) return
    const cx = (det.x + det.w/2) * W + Math.sin(f * 0.012 + i * 1.3) * W * 0.04
    const cy = (det.y + det.h/2) * H + Math.cos(f * 0.009 + i * 2.1) * H * 0.03
    const spd = (det.conf - 0.7) * 2
    const vx = Math.cos(f * 0.015 + i) * 60 * spd
    const vy = Math.sin(f * 0.012 + i) * 30 * spd
    const alpha = appear * (0.7 + 0.2 * Math.sin(f * 0.06 + i))
    for (let t2 = 1; t2 <= 12; t2++) {
      const tx = cx - vx * t2 * 0.06, ty = cy - vy * t2 * 0.06
      const ta = alpha * (1 - t2/12) * 0.45
      ctx.fillStyle = `rgba(${rgb},${ta})`
      ctx.beginPath(); ctx.arc(tx, ty, 1.5, 0, Math.PI*2); ctx.fill()
    }
    const ringR = 18 + 10 * Math.sin(f * 0.04 + i)
    ctx.strokeStyle = `rgba(${rgb},${alpha * 0.3})`; ctx.lineWidth = 0.8
    ctx.beginPath(); ctx.arc(cx, cy, ringR, 0, Math.PI*2); ctx.stroke()
    ctx.strokeStyle = `rgba(${rgb},${alpha * 0.15})`
    ctx.beginPath(); ctx.arc(cx, cy, ringR * 1.6, 0, Math.PI*2); ctx.stroke()
    const ds = 7 + det.w * W * 0.3
    ctx.strokeStyle = `rgba(${rgb},${alpha})`; ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(cx, cy-ds); ctx.lineTo(cx+ds, cy); ctx.lineTo(cx, cy+ds); ctx.lineTo(cx-ds, cy); ctx.closePath()
    ctx.stroke()
    ctx.fillStyle = `rgba(${rgb},${alpha*0.12})`; ctx.fill()
    ctx.strokeStyle = `rgba(${rgb},${alpha*0.8})`; ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + vx*0.25, cy + vy*0.25); ctx.stroke()
    ctx.font = '10px monospace'; ctx.fillStyle = `rgba(${rgb},${alpha*0.9})`
    ctx.fillText(`${det.cls} · ${Math.round(det.conf*100)}%`, cx - ds, cy - ds - 6)
  })
  const sweepAngle = (f * 0.022) % (Math.PI * 2)
  const cx0 = W * 0.5, cy0 = H * 0.5, maxR = Math.max(W, H)
  const sweepGrad = ctx.createLinearGradient(cx0, cy0, cx0 + Math.cos(sweepAngle)*maxR, cy0 + Math.sin(sweepAngle)*maxR)
  sweepGrad.addColorStop(0, `rgba(${rgb},0.12)`); sweepGrad.addColorStop(1, `rgba(${rgb},0)`)
  ctx.strokeStyle = sweepGrad; ctx.lineWidth = 2
  ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0 + Math.cos(sweepAngle)*maxR, cy0 + Math.sin(sweepAngle)*maxR); ctx.stroke()
  for (let a = 0; a < 1.2; a += 0.08) {
    const trailAngle = sweepAngle - a * 0.5
    ctx.strokeStyle = `rgba(${rgb},${0.04 * (1 - a/1.2)})`; ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0 + Math.cos(trailAngle)*maxR, cy0 + Math.sin(trailAngle)*maxR); ctx.stroke()
  }
}

function drawPersonnel(ctx, dets, W, H, f, rgb) {
  const JOINTS = [[0.5,0.05],[0.5,0.25],[0.3,0.28],[0.7,0.28],[0.2,0.50],[0.8,0.50],[0.25,0.70],[0.75,0.70],[0.38,0.55],[0.62,0.55],[0.35,0.78],[0.65,0.78],[0.33,0.97],[0.67,0.97]]
  const BONES = [[0,1],[1,2],[1,3],[2,4],[4,6],[3,5],[5,7],[2,8],[3,9],[8,9],[8,10],[9,11],[10,12],[11,13]]
  dets.forEach((det, i) => {
    const appear = Math.max(0, Math.min(1, (f - i * 10) / 24))
    if (appear === 0) return
    const bx = det.x * W, by = det.y * H, bw = det.w * W, bh = det.h * H
    const alpha = appear * (0.8 + 0.15 * Math.sin(f * 0.04 + i))
    const jitter = 1.2
    const grad = ctx.createRadialGradient(bx+bw/2, by+bh/2, 0, bx+bw/2, by+bh/2, bw*0.8)
    grad.addColorStop(0, `rgba(${rgb},${alpha*0.08})`); grad.addColorStop(1, `rgba(${rgb},0)`)
    ctx.fillStyle = grad; ctx.fillRect(bx-bw*0.2, by-bh*0.1, bw*1.4, bh*1.2)
    ctx.strokeStyle = `rgba(${rgb},${alpha*0.55})`; ctx.lineWidth = 1.2
    BONES.forEach(([a, b]) => {
      const ax = bx + JOINTS[a][0]*bw + Math.sin(f*0.02+i+a)*jitter
      const ay = by + JOINTS[a][1]*bh + Math.cos(f*0.018+i+a)*jitter
      const bpx = bx + JOINTS[b][0]*bw + Math.sin(f*0.02+i+b)*jitter
      const bpy = by + JOINTS[b][1]*bh + Math.cos(f*0.018+i+b)*jitter
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bpx, bpy); ctx.stroke()
    })
    JOINTS.forEach(([jx2, jy2], ji) => {
      const px = bx + jx2*bw + Math.sin(f*0.02+i+ji)*jitter
      const py = by + jy2*bh + Math.cos(f*0.018+i+ji)*jitter
      const isHead = ji === 0
      ctx.fillStyle = `rgba(${rgb},${alpha*(isHead?1:0.7)})`
      ctx.beginPath(); ctx.arc(px, py, isHead?3:1.8, 0, Math.PI*2); ctx.fill()
    })
    ctx.setLineDash([4, 4]); ctx.strokeStyle = `rgba(${rgb},${alpha*0.35})`; ctx.lineWidth = 0.8
    ctx.strokeRect(bx, by, bw, bh); ctx.setLineDash([])
    ctx.font = '10px monospace'; ctx.fillStyle = `rgba(${rgb},${alpha*0.9})`
    ctx.fillText(`${det.cls} · ${Math.round(det.conf*100)}%`, bx+2, by-4)
  })
}

function drawVehicles(ctx, dets, W, H, f, rgb, expanded) {
  dets.forEach((det, i) => {
    const appear = Math.max(0, Math.min(1, (f - i * 10) / 28))
    if (appear === 0) return
    const x = det.x * W, y = det.y * H, w = det.w * W, h = det.h * H
    const jx = expanded ? Math.sin(f * 0.015 + i) * 0.5 : 0
    const jy = expanded ? Math.cos(f * 0.012 + i) * 0.5 : 0
    const rx = x+jx, ry = y+jy
    const alpha = appear * (0.8 + 0.1 * Math.sin(f * 0.03 + i))
    ctx.fillStyle = `rgba(${rgb},${alpha*0.10})`; ctx.fillRect(rx, ry, w, h)
    ctx.strokeStyle = `rgba(${rgb},${alpha})`; ctx.lineWidth = 2; ctx.strokeRect(rx, ry, w, h)
    ctx.strokeStyle = `rgba(${rgb},${alpha*0.3})`; ctx.lineWidth = 0.8
    ctx.beginPath(); ctx.moveTo(rx+w*0.5, ry+h*0.1); ctx.lineTo(rx+w*0.5, ry+h*0.9); ctx.stroke()
    const trx = rx+w*0.5, trY = ry+h*0.38, trR = Math.min(w,h)*0.18
    ctx.beginPath(); ctx.arc(trx, trY, trR, 0, Math.PI*2); ctx.stroke()
    const heading = f * 0.008 + i * 1.1
    ctx.strokeStyle = `rgba(${rgb},${alpha*0.7})`; ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.moveTo(trx, trY); ctx.lineTo(trx + Math.cos(heading)*trR*2.4, trY + Math.sin(heading)*trR*2.4); ctx.stroke()
    ctx.strokeStyle = `rgba(${rgb},${alpha*0.25})`; ctx.lineWidth = 0.8
    ctx.beginPath(); ctx.moveTo(rx+w*0.1, ry+h*0.2); ctx.lineTo(rx+w*0.9, ry+h*0.2); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(rx+w*0.1, ry+h*0.75); ctx.lineTo(rx+w*0.9, ry+h*0.75); ctx.stroke()
    const tk = 12
    ctx.strokeStyle = `rgba(${rgb},${alpha})`; ctx.lineWidth = 3
    ctx.beginPath(); ctx.moveTo(rx,ry+tk); ctx.lineTo(rx,ry); ctx.lineTo(rx+tk,ry); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(rx+w-tk,ry); ctx.lineTo(rx+w,ry); ctx.lineTo(rx+w,ry+tk); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(rx,ry+h-tk); ctx.lineTo(rx,ry+h); ctx.lineTo(rx+tk,ry+h); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(rx+w-tk,ry+h); ctx.lineTo(rx+w,ry+h); ctx.lineTo(rx+w,ry+h-tk); ctx.stroke()
    const cbh = h * det.conf
    ctx.fillStyle = `rgba(${rgb},${alpha*0.15})`; ctx.fillRect(rx+w+3, ry+h-cbh, 4, cbh)
    ctx.fillStyle = `rgba(${rgb},${alpha*0.7})`; ctx.fillRect(rx+w+3, ry+h-cbh, 4, 3)
    ctx.strokeStyle = `rgba(${rgb},${alpha*0.12})`; ctx.lineWidth = 0.8
    ctx.beginPath(); ctx.arc(rx+w/2, ry+h/2, Math.max(w,h)*0.75, 0, Math.PI*2); ctx.stroke()
    ctx.font = '10px monospace'; ctx.fillStyle = `rgba(${rgb},${alpha*0.9})`
    ctx.fillText(`${det.cls} · ${Math.round(det.conf*100)}%`, rx+2, ry-4)
  })
}
