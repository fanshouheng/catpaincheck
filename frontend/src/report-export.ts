import { FGS_ITEMS } from './fgs-data'
import type { AutoFgsResult, Report, ScoreValue, YoloResult } from './types'

const FONT = '"Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif'

export function aiText(report: Report) {
  return report.ai.status !== 'ok'
    ? '本次无法分析'
    : report.ai.label === 'pain' ? '提示疼痛迹象' : '未提示疼痛迹象'
}

export function yoloText(result: YoloResult) {
  return result.status !== 'ok' || result.label === null ? '本次无法分类'
    : result.label === 'pain' ? '提示疼痛迹象' : '未提示疼痛迹象'
}

export function fgsText(result: Report['fgs'] | AutoFgsResult) {
  if (!result.assessable_count) return '本次无法评估'
  if (!result.complete) return '评分不完整'
  return result.flag === 'positive' ? '达到关注参考值' : '低于关注参考值'
}

export function scoreText(score: ScoreValue) {
  return score === null ? 'NA' : `${score} / 2`
}

export function formatTime(value: string) {
  return new Intl.DateTimeFormat('zh-CN', {
    timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).format(new Date(value))
}

function safeName(report: Report) {
  return (report.cat_name || '猫咪').replace(/[\\/:*?"<>|]/g, '_').slice(0, 36)
}

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function exportJson(report: Report) {
  saveBlob(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json;charset=utf-8' }),
    `${safeName(report)}-观察报告-${report.id.slice(0, 8)}.json`)
}

export async function exportPng(report: Report) {
  await document.fonts.ready
  const photo = new Image()
  photo.src = report.image_url
  await photo.decode()
  const canvas = document.createElement('canvas')
  const width = 1200
  canvas.width = width
  canvas.height = 6000
  const context = canvas.getContext('2d')
  if (!context) throw new Error('浏览器暂不支持报告图片导出，请下载 JSON 记录。')
  const ctx = context
  const margin = 76
  const contentWidth = width - margin * 2
  let y = 76
  ctx.fillStyle = '#f6f7f9'
  ctx.fillRect(0, 0, width, canvas.height)

  function font(size: number, bold = false) { ctx.font = `${bold ? 600 : 400} ${size}px ${FONT}` }
  function text(value: string, size: number, color = '#202936', bold = false, x = margin) {
    font(size, bold); ctx.fillStyle = color; ctx.fillText(value, x, y)
  }
  function wrapped(value: string, size: number, maxWidth: number, color = '#5f6b7b', x = margin, lineHeight = size * 1.65, bold = false) {
    font(size, bold); ctx.fillStyle = color
    let line = ''
    for (const ch of value) {
      if (ch === '\n' || (line && ctx.measureText(line + ch).width > maxWidth)) {
        ctx.fillText(line, x, y); y += lineHeight; line = ch === '\n' ? '' : ch
      } else line += ch
    }
    if (line) { ctx.fillText(line, x, y); y += lineHeight }
  }
  function rule() { ctx.fillStyle = '#dee3eb'; ctx.fillRect(margin, y, contentWidth, 1); y += 38 }
  function rounded(x: number, top: number, w: number, h: number, radius: number, fill: string) {
    ctx.beginPath(); ctx.roundRect(x, top, w, h, radius); ctx.fillStyle = fill; ctx.fill()
  }

  text('猫咪疼痛观察站', 30, '#202936', true)
  y += 64; wrapped(`${report.cat_name || '猫咪'}的观察报告`, 44, contentWidth, '#202936', margin, 60, true)
  y += 2; text(formatTime(report.completed_at), 21, '#68758a')
  y += 44

  const top = y
  const photoSize = 280
  const ratio = Math.min(photoSize / photo.width, photoSize / photo.height)
  const photoX = margin + (photoSize - photo.width * ratio) / 2
  const photoY = top + (photoSize - photo.height * ratio) / 2
  ctx.save(); ctx.beginPath(); ctx.roundRect(margin, top, photoSize, photoSize, 20); ctx.clip()
  ctx.fillStyle = '#e9edf3'; ctx.fillRect(margin, top, photoSize, photoSize)
  ctx.drawImage(photo, photoX, photoY, photo.width * ratio, photo.height * ratio)
  const localization = report.localization
  if (localization?.status === 'ok') {
    const sx = photo.width * ratio / report.image.width
    const sy = photo.height * ratio / report.image.height
    ctx.strokeStyle = '#297deb'; ctx.lineWidth = 1.6
    if (localization.bbox) {
      const [x1, y1, x2, y2] = localization.bbox
      ctx.setLineDash([6, 3]); ctx.strokeRect(photoX + x1 * sx, photoY + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
      ctx.setLineDash([])
    }
    for (const point of localization.points) {
      ctx.beginPath(); ctx.arc(photoX + point.x * sx, photoY + point.y * sy, 1.8, 0, Math.PI * 2)
      ctx.fillStyle = '#438ef5'; ctx.fill(); ctx.strokeStyle = '#ffffff'; ctx.lineWidth = .7; ctx.stroke()
    }
  }
  ctx.restore()
  if (localization?.status === 'ok') {
    font(16); ctx.fillStyle = '#68758a'; ctx.fillText(`${localization.points.length} 个模型预测关键点`, margin, top + photoSize + 25)
  }
  y = top + 36
  const rightX = margin + photoSize + 44
  wrapped(report.conclusion.title, 30, contentWidth - photoSize - 44, '#202936', rightX, 46)
  y += 10
  wrapped(report.conclusion.summary, 22, contentWidth - photoSize - 44, '#5f6b7b', rightX)
  y = Math.max(top + photoSize + 46, y + 24)
  rule()

  const fgsLabel = report.fgs.assessable_count ? `${report.fgs.total} / ${report.fgs.max} 分` : '无法计分'
  const machines = report.machines
  const cards = machines ? [
    { title: '自动 FGS', value: machines.auto_fgs.assessable_count ? `${machines.auto_fgs.total} / ${machines.auto_fgs.max} 分` : '无法计分', note: `可评估 ${machines.auto_fgs.assessable_count} / 5 项` },
    { title: 'YOLO 分类', value: yoloText(machines.yolo), note: '' },
    { title: '人工 FGS', value: fgsLabel, note: `可评估 ${report.fgs.assessable_count} / 5 项` },
  ] : [{ title: 'AI 初筛', value: aiText(report), note: '' }, { title: '人工 FGS', value: fgsLabel, note: `可评估 ${report.fgs.assessable_count} / 5 项` }]
  const boxTop = y
  const cardWidth = (contentWidth - 20 * (cards.length - 1)) / cards.length
  for (const [index, card] of cards.entries()) {
    const cardX = margin + index * (cardWidth + 20)
    rounded(cardX, boxTop, cardWidth, 172, 16, index === 0 ? '#eaf0f8' : '#eef0f4')
    y = boxTop + 37; text(card.title, 20, '#5f6b7b', false, cardX + 22)
    y += 49; text(card.value, cards.length === 3 ? 26 : 29, '#202936', true, cardX + 22)
    if (card.note) { y += 42; text(card.note, 19, '#5f6b7b', false, cardX + 22) }
  }
  y = boxTop + 210
  wrapped(`人工 FGS：${fgsText(report.fgs)}。${report.fgs.complete ? '五项完整时，≥4/10 为关注参考值。' : '存在看不清的项目，当前为部分评分，不应用完整评分阈值。'}`,
    22, contentWidth)
  if (machines) wrapped(`自动 FGS：${fgsText(machines.auto_fgs)}。自动与人工结果分别保留，不作简单投票。`, 22, contentWidth)
  if (report.fgs.normalized !== null) {
    wrapped(`标准化分数 ${report.fgs.normalized.toFixed(2)}；该数值不是疼痛概率。`, 20, contentWidth)
  }
  y += 16; rule()

  text(machines ? 'FGS 五项对照' : 'FGS 五项评分', 27, '#202936', true); y += 40
  if (machines) {
    text('自动', 20, '#5f6b7b', false, width - margin - 290)
    text('人工', 20, '#5f6b7b', false, width - margin - 125)
    y += 42
  }
  for (const item of FGS_ITEMS) {
    const score = report.fgs.scores[item.key]
    if (machines) {
      const machine = machines.auto_fgs.scores[item.key]
      const differs = score !== null && machine !== null && score !== machine
      if (differs) rounded(margin - 12, y - 29, contentWidth + 24, 55, 7, '#f8ebed')
      text(item.name, 23)
      text(scoreText(machine), 23, machine === null ? '#6e7888' : differs ? '#9a4855' : '#202936', true, width - margin - 290)
      text(scoreText(score), 23, score === null ? '#6e7888' : differs ? '#9a4855' : '#202936', true, width - margin - 125)
      y += 70
    } else {
      text(item.name, 23)
      text(score === null ? 'NA · 看不清' : scoreText(score), 23, score === null ? '#6e7888' : '#202936', true, width - margin - 170)
      y += 28
      wrapped(score === null ? '本项无法评估，不计入得分。' : item.choices[score], 19, contentWidth - 190)
      y += 16
    }
  }
  if (machines) wrapped('NA 不记为 0 分；柔红色行表示自动与人工分数不同，不代表哪一方正确。', 18, contentWidth)
  y += 2; rule()
  text('下一步建议', 27, '#202936', true); y += 42
  for (const [index, recommendation] of report.conclusion.recommendations.entries()) {
    wrapped(`${index + 1}. ${recommendation}`, 22, contentWidth)
    y += 7
  }
  if (machines) {
    y += 10
    wrapped(`自动 FGS：${machines.auto_fgs.model_version}`, 18, contentWidth)
    if (machines.auto_fgs.note) wrapped(machines.auto_fgs.note, 18, contentWidth)
    wrapped(`YOLO：${machines.yolo.model_version}`, 18, contentWidth)
    if (machines.yolo.note) wrapped(machines.yolo.note, 18, contentWidth)
    if (machines.yolo.detection_confidence !== null) wrapped(`YOLO 原始检测分数 ${machines.yolo.detection_confidence.toFixed(4)}，不是疼痛概率。`, 18, contentWidth)
  } else if (report.ai.note) { y += 10; wrapped(`AI 说明：${report.ai.note}`, 19, contentWidth) }
  if (report.collection) { y += 8; wrapped(report.collection.training_allowed ? '本次记录已授权用于模型改进。' : '本次记录未授权用于模型改进。', 18, contentWidth) }
  if (report.notes) {
    y += 14; wrapped(`观察备注：${report.notes.slice(0, 1000)}`, 20, contentWidth)
    if (report.notes.length > 1000) wrapped('图片仅展示前 1000 字备注，完整内容保留在 JSON 记录中。', 18, contentWidth)
  }
  y += 20; rule()
  wrapped('本报告记录照片中的疼痛表情线索；未见明显表情不能排除疼痛。', 18, contentWidth)
  wrapped(`记录编号 ${report.id}`, 16, contentWidth)

  const finalCanvas = document.createElement('canvas')
  finalCanvas.width = width
  finalCanvas.height = Math.ceil(y + 44)
  const finalContext = finalCanvas.getContext('2d')
  if (!finalContext) throw new Error('报告图片生成失败，请重试。')
  finalContext.drawImage(canvas, 0, 0)
  const blob = await new Promise<Blob>((resolve, reject) => finalCanvas.toBlob(
    value => value ? resolve(value) : reject(new Error('报告图片生成失败，请重试。')), 'image/png'))
  saveBlob(blob, `${safeName(report)}-观察报告-${report.id.slice(0, 8)}.png`)
}
