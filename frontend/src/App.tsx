import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import {
  ArrowDownToLine, ArrowLeft, ArrowRight, Check, CheckCircle2,
  CircleHelp, FileJson, ImagePlus, Info, LoaderCircle, PawPrint,
  RefreshCw, ScanFace, X,
} from 'lucide-react'
import { FGS_ITEMS, OFFICIAL_FGS_URL } from './fgs-data'
import { exportJson, exportPng, formatTime } from './report-export'
import { CLOUD_UPLOAD_BYTES, PhotoPreparationError, prepareCloudPhoto } from './cloud-upload'
import { FaceOverlay } from './FaceOverlay'
import { FgsBreakdown, ModelDetails, ResultsOverview } from './ReportComparison'
import type { AnalysisMeta, CollectionInfo, DraftScores, Health, Report, Scores } from './types'

const STORAGE_KEY = 'cat-pain-check.analysis-id'
const CAT_IDS_KEY = 'cat-pain-check.cat-ids'
const MAX_BYTES = 12 * 1024 * 1024
const CLOUD_BUILD = import.meta.env.VITE_CLOUD_DEPLOYMENT === 'true'

class ApiError extends Error {
  status: number
  constructor(message: string, status: number) { super(message); this.status = status }
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  if (!response.ok) {
    let message = '暂时无法完成请求，请稍后重试。'
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') message = body.detail
      else if (typeof body.message === 'string') message = body.message
      else if (Array.isArray(body.detail)) message = '提交的信息格式不正确，请检查后重试。'
    } catch { /* The server may return a non-JSON error. */ }
    throw new ApiError(message, response.status)
  }
  return response.json() as Promise<T>
}

function rememberedId() {
  try { return localStorage.getItem(STORAGE_KEY) } catch { return null }
}
function rememberId(id: string | null) {
  try { id ? localStorage.setItem(STORAGE_KEY, id) : localStorage.removeItem(STORAGE_KEY) } catch { /* Storage is optional. */ }
}

function anonymousCatId(name: string) {
  const key = `cat:${name.normalize('NFKC').trim().replace(/\s+/g, ' ').toLocaleLowerCase() || 'unknown'}`
  const fresh = crypto.randomUUID()
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(CAT_IDS_KEY) || '{}')
    const ids: Record<string, string> = stored && typeof stored === 'object' && !Array.isArray(stored) ? stored as Record<string, string> : {}
    const previous = ids[key]
    if (typeof previous === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(previous)) return previous
    ids[key] = fresh
    localStorage.setItem(CAT_IDS_KEY, JSON.stringify(ids))
  } catch { /* Analysis also works without browser storage. */ }
  return fresh
}

function CatArt() {
  return <svg className="cat-art" viewBox="0 0 420 390" fill="none" aria-hidden="true">
    <circle cx="210" cy="200" r="162" className="art-halo" />
    <circle cx="210" cy="200" r="133" className="art-orbit" />
    <path d="M116 179 103 79Q140 80 170 121Q211 108 252 122Q286 80 321 79L307 182Q325 220 305 264Q280 310 211 315Q142 310 117 268Q94 229 116 179Z" className="art-head" />
    <path d="m119 107 14 54 29-29M302 107l-15 54-28-29" className="art-line" />
    <path d="M140 204q22-20 46 0M237 204q22-20 46 0" className="art-line art-eye" />
    <path d="M199 233q12-6 24 0l-12 12Z" className="art-nose" />
    <path d="M211 245v12m0 0q-13 17-27 2m27-2q13 17 27 2" className="art-line" />
    <path d="m164 241-64-9m64 22-68 5m164-18 64-9m-64 22 68 5" className="art-whiskers" />
    <path d="M166 307q-16 23-19 48m108-48q16 23 19 48" className="art-line" />
    <circle cx="107" cy="73" r="5" className="art-dot" /><circle cx="321" cy="74" r="5" className="art-dot" />
    <circle cx="141" cy="204" r="4" className="art-dot" /><circle cx="281" cy="204" r="4" className="art-dot" />
    <circle cx="211" cy="238" r="4" className="art-dot" />
    <path d="M45 151h24m-12-12v24M345 291h20m-10-10v20" className="art-plus" />
  </svg>
}

function Stepper({ current }: { current: number }) {
  return <ol className="steps" aria-label="观察流程">
    {['上传照片', 'FGS 人工观察', '综合报告'].map((step, index) => <li key={step}
      className={index === current ? 'step current' : index < current ? 'step complete' : 'step'}
      aria-current={index === current ? 'step' : undefined}>
      <span className="step-number">{index < current ? <Check size={13} strokeWidth={2.5} /> : index + 1}</span>
      <span>{step}</span>
    </li>)}
  </ol>
}

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthFailed, setHealthFailed] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [catName, setCatName] = useState('')
  const [trainingAllowed, setTrainingAllowed] = useState(false)
  const [permissionBusy, setPermissionBusy] = useState(false)
  const [permissionError, setPermissionError] = useState('')
  const [analysis, setAnalysis] = useState<AnalysisMeta | null>(null)
  const [report, setReport] = useState<Report | null>(null)
  const [scores, setScores] = useState<DraftScores>({})
  const [notes, setNotes] = useState('')
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [preparingUpload, setPreparingUpload] = useState(false)
  const [uploadAdjusted, setUploadAdjusted] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [restoring, setRestoring] = useState(Boolean(rememberedId()))
  const [dragging, setDragging] = useState(false)
  const [pollRetry, setPollRetry] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const photoDialogRef = useRef<HTMLDialogElement>(null)
  const headingRef = useRef<HTMLHeadingElement>(null)
  const currentIdRef = useRef<string | null>(null)
  const stage = report ? 2 : analysis ? 1 : 0
  const chosenCount = FGS_ITEMS.filter(item => Object.prototype.hasOwnProperty.call(scores, item.key)).length
  const allChosen = chosenCount === 5
  const cloudMode = CLOUD_BUILD || health?.deployment === 'cloud'

  useEffect(() => {
    const controller = new AbortController()
    api<Health>('/api/health', { signal: controller.signal }).then(setHealth).catch(err => {
      if (err.name !== 'AbortError') setHealthFailed(true)
    })
    const stored = rememberedId()
    if (stored) {
      currentIdRef.current = stored
      ;(async () => {
        try {
          const existing = await api<Report>(`/api/analyses/${stored}/report`, { signal: controller.signal })
          if (currentIdRef.current === stored) { setReport(existing); setCatName(existing.cat_name || '') }
        } catch (err) {
          if (err instanceof ApiError && err.status === 404) {
            try {
              const existing = await api<AnalysisMeta>(`/api/analyses/${stored}`, { signal: controller.signal })
              if (currentIdRef.current === stored) { setAnalysis(existing); setCatName(existing.cat_name || '') }
            } catch (metaError) {
              if ((metaError as Error).name !== 'AbortError') {
                if (metaError instanceof ApiError && metaError.status === 404) rememberId(null)
                setError('上次的观察暂时无法恢复。你可以重新上传照片。')
              }
            }
          } else if ((err as Error).name !== 'AbortError') setError(CLOUD_BUILD ? '暂时无法恢复上次报告，请稍后重试。' : '暂时无法恢复上次报告，请确认分析服务已启动。')
        } finally { if (!controller.signal.aborted) setRestoring(false) }
      })()
    }
    return () => controller.abort()
  }, [])

  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview) }
  }, [preview])

  useEffect(() => {
    if (!analysis || analysis.status !== 'processing' || report) return
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    let failures = 0
    const id = analysis.id
    async function poll() {
      try {
        const next = await api<AnalysisMeta>(`/api/analyses/${id}`, { signal: controller.signal })
        if (controller.signal.aborted || currentIdRef.current !== id) return
        failures = 0
        setAnalysis(next)
        if (next.status === 'processing') timer = setTimeout(poll, 1500)
      } catch (err) {
        if (controller.signal.aborted) return
        failures += 1
        if (failures < 4) timer = setTimeout(poll, 2500)
        else setError('暂时无法获取分析进度。已选的评分会保留，请点击“重试连接”。')
      }
    }
    timer = setTimeout(poll, 900)
    return () => { controller.abort(); clearTimeout(timer) }
  }, [analysis?.id, analysis?.status, report, pollRetry])

  useEffect(() => {
    if (stage > 0) {
      headingRef.current?.focus({ preventScroll: true })
      window.scrollTo({ top: 0, behavior: 'instant' })
    }
  }, [stage])

  function selectFile(candidate: File | undefined) {
    if (!candidate || uploading) return
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(candidate.type)) {
      setError('请选择 JPG、PNG 或 WebP 图片。暂不支持 HEIC、动图或视频。'); return
    }
    if (candidate.size > MAX_BYTES) { setError('图片超过 12 MB，请换一张较小的照片。'); return }
    if (!candidate.size) { setError('这张图片是空文件，请重新选择。'); return }
    setFile(candidate); setPreview(URL.createObjectURL(candidate)); setError('')
  }

  function dropFile(event: DragEvent<HTMLElement>) {
    event.preventDefault(); setDragging(false)
    if (event.dataTransfer.files.length > 1) { setError('每次观察请上传一张照片。'); return }
    selectFile(event.dataTransfer.files[0])
  }

  async function uploadPhoto() {
    if (!file || uploading) return
    setError(''); setUploading(true); setUploadAdjusted(false)
    try {
      const serverLimit = health?.max_upload_bytes
      const maxBytes = serverLimit && serverLimit > 0 ? Math.min(CLOUD_UPLOAD_BYTES, serverLimit) : CLOUD_UPLOAD_BYTES
      setPreparingUpload(cloudMode && file.size > maxBytes)
      const uploadFile = cloudMode ? await prepareCloudPhoto(file, maxBytes) : file
      setPreparingUpload(false)
      setUploadAdjusted(uploadFile !== file)
      const form = new FormData(); form.append('image', uploadFile); form.append('cat_name', catName.trim())
      form.append('cat_id', anonymousCatId(catName))
      form.append('training_allowed', String(trainingAllowed))
      const meta = await api<AnalysisMeta>('/api/analyses', { method: 'POST', body: form })
      currentIdRef.current = meta.id; rememberId(meta.id); setAnalysis(meta); setScores({}); setReport(null)
    } catch (err) {
      setError(err instanceof ApiError || err instanceof PhotoPreparationError ? err.message : cloudMode ? '连接分析服务失败，请稍后重试。' : '连接分析服务失败，请检查服务是否已启动后重试。')
    } finally { setUploading(false); setPreparingUpload(false) }
  }

  async function submitScores() {
    if (!analysis || !allChosen || submitting || analysis.status !== 'ready') return
    const id = analysis.id
    setSubmitting(true); setError('')
    try {
      const result = await api<Report>(`/api/analyses/${id}/scores`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scores: scores as Scores, notes: notes.trim() }),
      })
      if (currentIdRef.current === id) setReport(result)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : '报告生成失败，已选评分仍然保留，请重试。')
    } finally { setSubmitting(false) }
  }

  function startOver() {
    currentIdRef.current = null; rememberId(null)
    setAnalysis(null); setReport(null); setFile(null); setPreview(''); setScores({}); setNotes(''); setError(''); setCatName(''); setUploadAdjusted(false); setTrainingAllowed(false); setPermissionError('')
    if (inputRef.current) inputRef.current.value = ''
    window.scrollTo({ top: 0, behavior: 'instant' })
  }

  async function withdrawTrainingPermission() {
    if (!report?.collection?.training_allowed || permissionBusy) return
    const id = report.id
    setPermissionBusy(true); setPermissionError('')
    try {
      const collection = await api<CollectionInfo>(`/api/analyses/${id}/training-permission`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ allowed: false }),
      })
      setReport(previous => previous?.id === id ? { ...previous, collection } : previous)
    } catch (err) {
      if (currentIdRef.current === id) setPermissionError(err instanceof ApiError ? err.message : '撤回未成功，请重试。')
    } finally { setPermissionBusy(false) }
  }

  async function downloadPng() {
    if (!report || exporting) return
    setExporting(true); setError('')
    try { await exportPng(report) }
    catch (err) { setError((err as Error).message || '报告图片生成失败，请重试。') }
    finally { setExporting(false) }
  }

  const reportConcern = report && (['agreement_positive', 'fgs_positive', 'dual_positive'].includes(report.conclusion.code) || report.fgs.flag === 'positive')
  const reportUncertain = report && ['incomplete', 'discordant', 'ai_unavailable', 'dual_discordant', 'dual_incomplete', 'dual_unavailable'].includes(report.conclusion.code)

  return <div className="site-shell">
    <header className="site-header">
      <a className="brand" href="#" onClick={event => { event.preventDefault(); if (stage > 0) startOver(); else window.scrollTo({ top: 0, behavior: 'instant' }) }} aria-label="猫咪疼痛观察站首页">
        <span className="brand-icon"><PawPrint size={23} strokeWidth={1.7} /></span>
        <span>猫咪疼痛观察站</span>
      </a>
      <a className="reference-link" href={OFFICIAL_FGS_URL} target="_blank" rel="noreferrer"><CircleHelp size={16} />FGS 参考指南<ArrowRight size={14} /></a>
    </header>

    <main>
      <div className="flow-top"><Stepper current={stage} /></div>

      {restoring ? <div className="restore-state" role="status"><LoaderCircle className="spin" size={27} /><h1>正在恢复报告</h1></div> : <>
        {error && <div className="error-banner" role="alert"><Info size={19} /><span>{error}</span>
          {analysis?.status === 'processing' && <button type="button" onClick={() => { setError(''); setPollRetry(value => value + 1) }}>重试连接</button>}
          <button className="icon-button" type="button" onClick={() => setError('')} aria-label="关闭错误提示"><X size={17} /></button>
        </div>}

        {stage === 0 && <>
          <section className="intro" aria-labelledby="intro-title">
            <div className="intro-copy"><h1 id="intro-title">猫咪疼痛<br /><span>表情观察</span></h1></div>
            <div className="intro-art"><CatArt /></div>
          </section>

          <section className="upload-layout" aria-labelledby="upload-title">
            <div className="upload-panel">
              <h2 id="upload-title" className="visually-hidden">上传猫咪照片</h2>
              <input ref={inputRef} type="file" id="cat-photo" accept="image/jpeg,image/png,image/webp" className="visually-hidden" aria-label="选择猫咪照片" disabled={uploading}
                onChange={(event: ChangeEvent<HTMLInputElement>) => { selectFile(event.target.files?.[0]); event.target.value = '' }} />
              <div className={`upload-zone ${dragging ? 'is-dragging' : ''} ${file ? 'has-photo' : ''}`}
                onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={dropFile}>
                {file ? <>
                  <img className="upload-preview" src={preview} alt="待分析的猫咪照片" onError={() => { setFile(null); setPreview(''); setError('无法读取这张图片，请选择其他照片。') }} />
                  <div className="selected-file"><span><CheckCircle2 size={16} /><span>{file.name}<small>{(file.size / 1024 / 1024).toFixed(2)} MB</small></span></span>
                    <button className="text-button" type="button" onClick={() => inputRef.current?.click()} disabled={uploading}>更换照片</button></div>
                </> : <button className="upload-trigger" type="button" onClick={() => inputRef.current?.click()}>
                  <span className="upload-icon"><ImagePlus size={29} strokeWidth={1.5} /></span><strong>点击或拖入猫咪照片</strong><span>JPG / PNG / WebP · 最大 12 MB</span>
                </button>}
              </div>
              <label className="name-label" htmlFor="cat-name">猫咪名字（选填）</label>
              <input className="text-input" id="cat-name" value={catName} onChange={event => setCatName(event.target.value)} maxLength={40} autoComplete="off" disabled={uploading} />
              <label className="training-consent"><input type="checkbox" checked={trainingAllowed} onChange={event => setTrainingAllowed(event.target.checked)} disabled={uploading} /><span>允许本次照片及评分用于模型改进</span></label>
              <button className="button button-primary start-button" type="button" disabled={!file || uploading} onClick={uploadPhoto}>
                {uploading ? <><LoaderCircle className="spin" size={18} />{preparingUpload ? '正在处理大图…' : cloudMode ? '正在上传并分析…' : '正在上传照片…'}</> : <>开始本次观察<ArrowRight size={18} /></>}
              </button>
              {(healthFailed || !health || !health.model.ready) && <p className="model-status limited" role="status"><span className="status-dot" />
                {healthFailed ? cloudMode ? '分析服务暂时未连接，请稍后重试。' : '分析服务未连接，请确认本地服务已启动。' : health ? 'AI 暂不可用，仍可完成 FGS 评分。' : '正在连接分析服务…'}
              </p>}
            </div>
            <aside className="photo-guide"><h2>拍摄要点</h2>
              <ol><li><p>清醒时拍摄，避开打哈欠和眨眼。</p></li>
                <li><p>自然光下正面拍清双耳、双眼、口鼻和胡须。</p></li>
                <li><p>留出肩部，便于判断头部姿态。</p></li></ol>
            </aside>
          </section>
        </>}

        {stage === 1 && analysis && <section className="scoring-view" aria-labelledby="scoring-title">
          <div className="page-title"><h1 ref={headingRef} tabIndex={-1} id="scoring-title">FGS 人工评分</h1><p>{uploadAdjusted && '大图已等比压缩。'}提交后查看 AI 结果。</p></div>
          <div className="scoring-layout">
            <aside className="photo-sidebar"><div className="reference-photo"><FaceOverlay src={analysis.image_url} image={analysis.image} localization={analysis.localization} processing={analysis.status === 'processing'} alt={`${analysis.cat_name || '猫咪'}本次观察的照片`} /></div>
              {analysis.cat_name && <div className="photo-sidebar-caption"><strong>{analysis.cat_name}</strong></div>}
              {analysis.status !== 'ready' && <div className="processing-status" role="status">{analysis.status === 'processing' ? <><LoaderCircle className="spin" size={16} />AI 分析中…</> : <><Info size={16} />{analysis.message || '分析遇到问题，请重新上传'}</>}</div>}
              <button className="text-button" type="button" onClick={startOver}><ArrowLeft size={15} />重新上传</button>
            </aside>
            <div className="score-form">
              <button className="mobile-photo-button" type="button" onClick={() => photoDialogRef.current?.showModal()}><img src={analysis.image_url} alt="" /><span>查看照片与定位</span><ScanFace size={20} /></button>
              <div className="score-form-top"><span className="completion-count">已选择 <strong>{chosenCount}</strong> / 5 项</span></div>
              <p className="fgs-scoring-guide">1 分：轻度或难以确定变化；NA：看不清，不计分。</p>
              {FGS_ITEMS.map(item => <fieldset key={item.key} disabled={submitting} className={`score-card ${Object.prototype.hasOwnProperty.call(scores, item.key) ? 'answered' : ''}`}>
                <legend><span>{item.name}</span>
                  {Object.prototype.hasOwnProperty.call(scores, item.key) && <CheckCircle2 className="answered-check" size={19} />}</legend>
                <div className="score-options">
                  {([0, 1, 2] as const).map(value => <label key={value} className={`score-option ${scores[item.key] === value ? 'selected' : ''}`}>
                    <input type="radio" name={`score-${item.key}`} value={value} checked={scores[item.key] === value} onChange={() => setScores(previous => ({ ...previous, [item.key]: value }))} />
                    <span className="option-score">{value}<small>分</small></span><span className="option-description">{item.choices[value]}</span><span className="option-radio" aria-hidden="true" />
                  </label>)}
                </div>
                <div className="score-bottom"><p><Info size={14} />{item.hint}</p><label className={`na-option ${Object.prototype.hasOwnProperty.call(scores, item.key) && scores[item.key] === null ? 'selected' : ''}`}>
                  <input type="radio" name={`score-${item.key}`} value="na" checked={Object.prototype.hasOwnProperty.call(scores, item.key) && scores[item.key] === null}
                    onChange={() => setScores(previous => ({ ...previous, [item.key]: null }))} /><span>看不清 / NA</span>
                </label></div>
              </fieldset>)}
              <div className="observation-notes"><label htmlFor="observation-notes">备注（选填）</label><textarea id="observation-notes" value={notes} onChange={event => setNotes(event.target.value)} maxLength={1000} rows={3} disabled={submitting} />{notes.length >= 900 && <span className="character-count">{notes.length} / 1000</span>}</div>
              <div className="score-submit">
                <button className="button button-primary" type="button" disabled={!allChosen || analysis.status !== 'ready' || submitting} onClick={submitScores}>
                  {submitting ? <><LoaderCircle className="spin" size={17} />正在生成报告…</> : analysis.status === 'processing' && allChosen ? <><LoaderCircle className="spin" size={17} />等待后台处理…</> : <>查看综合报告<ArrowRight size={17} /></>}
                </button></div>
            </div>
          </div>
          <dialog className="photo-dialog" ref={photoDialogRef} aria-labelledby="photo-dialog-title"><div className="photo-dialog-heading"><h2 id="photo-dialog-title">本次照片</h2><button className="icon-button" type="button" aria-label="关闭照片" onClick={() => photoDialogRef.current?.close()}><X size={21} /></button></div><FaceOverlay src={analysis.image_url} image={analysis.image} localization={analysis.localization} processing={analysis.status === 'processing'} alt={`${analysis.cat_name || '猫咪'}的本次照片，供 FGS 评分对照`} /></dialog>
        </section>}

        {stage === 2 && report && <section className="report-view" aria-labelledby="report-title">
          <div className="report-page-heading"><div className="page-title"><h1 ref={headingRef} tabIndex={-1} id="report-title">观察报告</h1><p>{report.cat_name || '猫咪'} · {formatTime(report.completed_at)}</p></div><button className="button button-secondary" type="button" onClick={startOver}><RefreshCw size={16} />重新上传</button></div>
          <div className={`conclusion-panel ${reportConcern ? 'concern' : reportUncertain ? 'uncertain' : ''}`}>
            <span className="conclusion-icon">{reportUncertain ? <CircleHelp size={27} strokeWidth={1.5} /> : reportConcern ? <Info size={27} strokeWidth={1.5} /> : <CheckCircle2 size={27} strokeWidth={1.5} />}</span>
            <div><h2>{report.conclusion.title}</h2><p>{report.conclusion.summary}</p></div>
          </div>
          <div className="report-grid">
            <div className="report-primary">
              <ResultsOverview report={report} />
              <FgsBreakdown report={report} />
              <section className="recommendations"><h2>建议</h2><ol>{report.conclusion.recommendations.map((item, index) => <li key={index}><span>{index + 1}</span><p>{item}</p></li>)}</ol></section>
              {report.notes && <section className="saved-notes"><span className="section-kicker">你的观察备注</span><p>{report.notes}</p></section>}
              <ModelDetails report={report} />
            </div>
            <aside className="report-sidebar"><div className="report-photo"><FaceOverlay src={report.image_url} image={report.image} localization={report.localization} alt={`${report.cat_name || '猫咪'}的本次观察照片`} /></div>
              <div className="save-report"><button className="button button-primary" type="button" onClick={downloadPng} disabled={exporting}>{exporting ? <LoaderCircle className="spin" size={17} /> : <ArrowDownToLine size={17} />}{exporting ? '正在生成图片…' : '下载报告图片'}</button><button className="button button-quiet" type="button" onClick={() => exportJson(report)}><FileJson size={16} />导出 JSON</button></div>
              {report.collection && <div className="training-permission"><p role="status">{report.collection.training_allowed ? '本次照片及评分已获准用于模型改进。' : '本次记录未授权用于模型改进。'}</p>
                {report.collection.training_allowed && <button className="text-button" type="button" onClick={withdrawTrainingPermission} disabled={permissionBusy}>{permissionBusy ? '正在撤回…' : permissionError ? '重试撤回训练授权' : '撤回训练授权'}</button>}
                {permissionError && <p className="permission-error" role="alert">{permissionError}</p>}
              </div>}
            </aside>
          </div>
        </section>}
      </>}
    </main>
    <footer className="site-footer"><span>{cloudMode ? '照片在本站处理' : '照片仅在本机处理'}</span><a href="https://github.com/fanshouheng/catpaincheck" target="_blank" rel="noreferrer">源代码</a></footer>
  </div>
}

export default App
