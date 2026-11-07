import { useId, useState, type PointerEvent, type MouseEvent } from 'react'
import { LoaderCircle } from 'lucide-react'
import type { Landmark, Localization } from './types'

interface Props {
  src: string
  alt: string
  image: { width: number; height: number }
  localization?: Localization | null
  processing?: boolean
}

export function FaceOverlay({ src, alt, image, localization, processing = false }: Props) {
  const titleId = useId()
  const [showOverlay, setShowOverlay] = useState(true)
  const [showNumbers, setShowNumbers] = useState(false)
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [hoveredId, setHoveredId] = useState<number | null>(null)
  const [imageFailed, setImageFailed] = useState(false)
  const located = localization?.status === 'ok' && localization.points.length > 0
  const points = located ? localization.points : []
  const bbox = located ? localization.bbox : null
  const active = points.find(point => point.id === (hoveredId ?? selectedId))
  const faceWidth = bbox ? bbox[2] - bbox[0] : 0
  const faceHeight = bbox ? bbox[3] - bbox[1] : 0
  const zoomToFace = showOverlay && bbox && faceWidth > 0 && faceHeight > 0
  const viewX = zoomToFace ? Math.max(0, bbox[0] - faceWidth * 0.1) : 0
  const viewY = zoomToFace ? Math.max(0, bbox[1] - faceHeight * 0.1) : 0
  const viewRight = zoomToFace ? Math.min(image.width, bbox[2] + faceWidth * 0.1) : image.width
  const viewBottom = zoomToFace ? Math.min(image.height, bbox[3] + faceHeight * 0.1) : image.height
  const viewWidth = viewRight - viewX
  const viewHeight = viewBottom - viewY
  // Zoom changes only the viewport; points and pointer mapping retain image pixels.
  const viewShortSide = Math.min(viewWidth, viewHeight)
  const markerRadius = viewShortSide * 0.006
  const textSize = viewShortSide * 0.028
  const locating = processing || localization?.status === 'processing'
  const status = locating ? '正在定位猫脸…'
    : localization ? localization.note || '本次未获得猫脸关键点。'
      : '该记录没有关键点，请重新上传。'

  function nearestPoint(event: PointerEvent<SVGSVGElement> | MouseEvent<SVGSVGElement>): Landmark | undefined {
    const svg = event.currentTarget
    const matrix = svg.getScreenCTM()
    if (!matrix) return
    const pointer = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
    const maxDistance = 22 / Math.hypot(matrix.a, matrix.b)
    let nearest: Landmark | undefined
    let distance = maxDistance
    for (const point of points) {
      const next = Math.hypot(pointer.x - point.x, pointer.y - point.y)
      if (next <= distance) { distance = next; nearest = point }
    }
    return nearest
  }

  return <div className="face-inspector">
    {located && <div className="face-toolbar">
      <div className="face-mode" role="group" aria-label="照片显示方式">
        <button type="button" aria-pressed={showOverlay} onClick={() => setShowOverlay(true)}>猫脸</button>
        <button type="button" aria-pressed={!showOverlay} onClick={() => { setShowOverlay(false); setHoveredId(null) }}>原图</button>
      </div>
      <button className="face-number-toggle" type="button" aria-pressed={showNumbers} disabled={!showOverlay}
        onClick={() => setShowNumbers(value => !value)}>编号</button>
    </div>}
    <svg className={`face-overlay-view${located && showOverlay ? ' has-landmarks' : ''}`}
      viewBox={`${viewX} ${viewY} ${viewWidth} ${viewHeight}`} preserveAspectRatio="xMidYMid meet"
      style={{ aspectRatio: `${viewWidth} / ${viewHeight}` }} role="img" aria-labelledby={titleId}
      onPointerMove={event => { if (showOverlay && event.pointerType !== 'touch') setHoveredId(nearestPoint(event)?.id ?? null) }}
      onPointerLeave={() => setHoveredId(null)}
      onClick={event => { if (showOverlay) setSelectedId(nearestPoint(event)?.id ?? null) }}>
      <title id={titleId}>{alt}{located && showOverlay ? `，显示猫脸框和 ${points.length} 个模型预测关键点` : ''}</title>
      <image href={src} width={image.width} height={image.height} onError={() => setImageFailed(true)} />
      {located && showOverlay && <g className="face-geometry" aria-hidden="true">
        {bbox && <rect className="face-bbox" x={bbox[0]} y={bbox[1]} width={bbox[2] - bbox[0]} height={bbox[3] - bbox[1]} vectorEffect="non-scaling-stroke" />}
        {points.map(point => <g key={point.id} className={point.id === active?.id ? 'face-point is-active' : 'face-point'}>
          <circle cx={point.x} cy={point.y} r={point.id === active?.id ? markerRadius * 1.7 : markerRadius} vectorEffect="non-scaling-stroke" />
          {showNumbers && <text x={point.x + markerRadius * 1.9} y={point.y - markerRadius * 1.9} fontSize={textSize}>{point.id}</text>}
        </g>)}
      </g>}
    </svg>
    {imageFailed && <p className="face-status" role="status">照片暂时无法加载，请刷新重试。</p>}
    {located ? <div className="face-coordinate-bar">
      <select aria-label="查看关键点坐标" value={selectedId ?? ''} disabled={!showOverlay}
        onChange={event => { setSelectedId(event.target.value ? Number(event.target.value) : null); setHoveredId(null) }}>
        <option value="">{points.length} 个预测点</option>
        {points.map(point => <option key={point.id} value={point.id}>点 {point.id}</option>)}
      </select>
      <output className="face-coordinate" aria-live="polite" aria-atomic="true">
        {showOverlay ? active ? `#${active.id} · (${active.x.toFixed(1)}, ${active.y.toFixed(1)}) px` : '点选查看坐标' : ''}
      </output>
    </div> : <p className="face-status" role="status">{locating && <LoaderCircle size={14} className="spin" aria-hidden="true" />}{status}</p>}
    {located && <details className="face-prediction-note"><summary>定位说明</summary><p>模型预测点；遮挡点可能失真，无逐点误差与可见性判断。</p></details>}
  </div>
}
