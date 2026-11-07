import { ChevronDown, PawPrint, ScanFace } from 'lucide-react'
import { FGS_ITEMS } from './fgs-data'
import { aiText, fgsText, scoreText, yoloText } from './report-export'
import type { AutoFgsResult, Report } from './types'

function FgsCard({ result, automatic = false }: { result: Report['fgs'] | AutoFgsResult; automatic?: boolean }) {
  return <section className={`judgment-card fgs-judgment ${automatic ? 'automatic-fgs' : ''}`}>
    <span className="section-kicker">{automatic ? <ScanFace size={17} /> : <PawPrint size={17} />}{automatic ? '自动 FGS' : '人工 FGS'}</span>
    <div className="fgs-big-score">{result.assessable_count ? <><strong>{result.total}</strong><span>/ {result.max}</span></> : <strong className="no-score">无法计分</strong>}</div>
    <p>可评估 {result.assessable_count} / 5 项{!result.complete && '，仅提供部分评分。'}</p>
    <span className={`judgment-tag ${result.flag === 'positive' ? 'needs-attention' : result.complete ? '' : 'neutral'}`}>{fgsText(result)}</span>
  </section>
}

export function ResultsOverview({ report }: { report: Report }) {
  const machines = report.machines
  return <div className={`judgment-grid ${machines ? 'dual-results' : ''}`}>
    {machines ? <>
      <FgsCard result={machines.auto_fgs} automatic />
      <section className="judgment-card yolo-judgment"><span className="section-kicker"><ScanFace size={17} />YOLO 分类</span><h2>{yoloText(machines.yolo)}</h2>
        {machines.yolo.status !== 'ok' && <p>本次没有获得可用分类。</p>}
      </section>
    </> : <section className="judgment-card"><span className="section-kicker"><ScanFace size={17} />AI 初筛</span><h2>{aiText(report)}</h2>{report.ai.status !== 'ok' && <p>{report.ai.note || '本次没有获得可靠的模型结果。'}</p>}</section>}
    <FgsCard result={report.fgs} />
  </div>
}

export function FgsBreakdown({ report }: { report: Report }) {
  const automatic = report.machines?.auto_fgs
  return <section className="fgs-breakdown"><div className="section-heading"><h2>{automatic ? 'FGS 对照' : 'FGS 明细'}</h2></div>
    {automatic ? <table className="comparison-table"><thead><tr><th scope="col">项目</th><th scope="col">自动</th><th scope="col">人工</th></tr></thead><tbody>
      {FGS_ITEMS.map(item => {
        const manual = report.fgs.scores[item.key]
        const machine = automatic.scores[item.key]
        const differs = manual !== null && machine !== null && manual !== machine
        return <tr key={item.key} className={differs ? 'score-difference' : ''}>
          <th scope="row">{item.name}{differs && <span className="difference-note">分数不同</span>}</th>
          <td className={machine === null ? 'is-na' : ''}>{scoreText(machine)}</td><td className={manual === null ? 'is-na' : ''}>{scoreText(manual)}</td>
        </tr>
      })}
    </tbody></table> : <div className="breakdown-list">{FGS_ITEMS.map(item => {
      const value = report.fgs.scores[item.key]
      return <div className="breakdown-row" key={item.key}><div><strong>{item.name}</strong><p>{value === null ? '看不清，本项不计分' : item.choices[value]}</p></div><span className={`breakdown-value ${value === null ? 'is-na' : ''}`}>{value === null ? 'NA' : <>{value}<small>/ 2</small></>}</span></div>
    })}</div>}
  </section>
}

export function ModelDetails({ report }: { report: Report }) {
  const machines = report.machines
  return <details className="model-details"><summary>评分说明<ChevronDown size={16} /></summary><div>
    {machines ? <>
      <p>自动 FGS：{machines.auto_fgs.note || '逐项预测 0–2 分；无法计算的项目记为 NA。'}</p>
      <p>YOLO：{machines.yolo.note || '独立的疼痛表情二分类结果。'}</p>
      {machines.yolo.detection_confidence !== null && <p>YOLO 原始检测分数：{machines.yolo.detection_confidence.toFixed(4)}；检测阈值：{machines.yolo.confidence_threshold}。检测分数不是疼痛概率。</p>}
      <p>自动 FGS 模型：{machines.auto_fgs.model_version}；预处理：{machines.auto_fgs.preprocess_version}</p>
      <p>YOLO 模型：{machines.yolo.model_version}；预处理：{machines.yolo.preprocess_version}</p>
      <p>机器模型可能对遮挡、眨眼或拍摄角度产生错误评分。自动与人工结果分别保留；一致不代表已经验证正确。</p>
    </> : <>
      <p>{report.ai.note}</p>
      {report.ai.raw_score !== null && <p>模型原始分值：{report.ai.raw_score.toFixed(4)}；分类阈值：{report.ai.threshold}。模型分值未经疼痛概率校准。</p>}
      <p>模型版本：{report.ai.model_version || '未提供'}</p>
    </>}
    <p>{report.fgs.normalized === null ? '人工五项均无法评估，未计算有效分数。' : `人工标准化 FGS 分数 ${report.fgs.normalized.toFixed(2)}，不代表疼痛概率。`}</p>
    <p>NA 不记为 0 分。五项完整时，FGS ≥4/10 为关注参考值；部分评分不套用该阈值。各项结果不作简单投票或加权平均。</p>
    <p>记录编号：<code>{report.id}</code></p>
  </div></details>
}
