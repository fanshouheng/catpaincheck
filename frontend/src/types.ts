export type ScoreKey = 'ears' | 'eyes' | 'muzzle' | 'whiskers' | 'head'
export type ScoreValue = 0 | 1 | 2 | null
export type Scores = Record<ScoreKey, ScoreValue>
export type DraftScores = Partial<Scores>

export interface CollectionInfo {
  cat_id: string | null
  training_allowed: boolean
  permission_version: string
  permission_at: string | null
}

export interface AnnotationInfo {
  annotation_id: string
  submitted_at: string
  label_source: 'user_fgs'
  rubric_version: string
  saw_ai_before_scoring: boolean
  supersedes_annotation_id: string | null
}

export interface YoloResult {
  status: 'ok' | 'unavailable' | 'error'
  label: 'pain' | 'no_pain' | null
  detection_confidence: number | null
  bbox: [number, number, number, number] | null
  detections: Record<string, unknown>[]
  model_version: string
  preprocess_version: string
  note: string
  confidence_threshold: number
}

export interface AutoFgsResult {
  status: 'ok' | 'partial' | 'unavailable' | 'error'
  model_version: string
  preprocess_version: string
  note: string
  scores: Scores
  raw_outputs: Partial<Record<ScoreKey, number[]>>
  regions: Partial<Record<ScoreKey, [number, number, number, number]>>
  total: number
  max: number
  assessable_count: number
  complete: boolean
  flag: 'positive' | 'negative' | 'incomplete'
}

export interface MachineResults {
  prediction_id: string
  created_at: string
  yolo: YoloResult
  auto_fgs: AutoFgsResult
}

export interface Landmark {
  id: number
  x: number
  y: number
  confidence: number | null
}

export interface Localization {
  status: 'processing' | 'ok' | 'unavailable' | 'error'
  reason: string
  note: string
  model_version: string
  coordinate_space: 'image_pixels'
  bbox: [number, number, number, number] | null
  cat_bbox: [number, number, number, number] | null
  detector_confidence: number | null
  points: Landmark[]
}

export interface AnalysisMeta {
  id: string
  created_at: string
  cat_name: string
  image_url: string
  status: 'processing' | 'ready' | 'error'
  image: { width: number; height: number }
  localization?: Localization | null
  collection?: CollectionInfo | null
  message?: string
}

export interface Health {
  status: string
  deployment?: 'cloud' | 'local'
  max_upload_bytes?: number
  model: { ready: boolean; detail: string; version: string }
}

export interface Report {
  schema_version: number
  id: string
  created_at: string
  completed_at: string
  cat_name: string
  image_url: string
  image: { width: number; height: number }
  localization?: Localization | null
  machines?: MachineResults | null
  collection?: CollectionInfo | null
  annotation?: AnnotationInfo | null
  notes: string
  ai: {
    status: 'ok' | 'unavailable' | 'error'
    label: 'pain' | 'no_pain' | null
    raw_score: number | null
    threshold: number
    model_version: string
    note: string
    crop_url: string | null
    bbox: [number, number, number, number] | null
  }
  fgs: {
    scores: Scores
    total: number
    max: number
    assessable_count: number
    normalized: number | null
    complete: boolean
    threshold: number
    flag: 'positive' | 'negative' | 'incomplete'
  }
  conclusion: {
    code: string
    title: string
    summary: string
    recommendations: string[]
  }
}
