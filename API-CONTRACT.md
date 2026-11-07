# API contract · dual models

当前线格式仍为 `schema_version: 1`，新增字段可选，以兼容旧单模型报告。默认推理组合为 `yolo-kittycheck.dual-v1`；三份比较规则为 `fgs-three-way-v2`。字段与校验以 `backend/app/schemas.py`、路由以 `backend/app/main.py` 为准。

## 通用约定

- FGS keys 固定为 `ears, eyes, muzzle, whiskers, head`；五个 key 必须全部提交。值仅允许严格整数 `0/1/2` 或 `null`；布尔、字符串、浮点不转成分数。
- `null` = NA，不计作零；总分只累加有效项，`max=2*assessable_count`。只有五项完整时，`total>=4` 才使 `flag=positive`；否则 `flag=incomplete`。
- 所有 bbox 为 `[x1,y1,x2,y2]`。公开定位坐标指向经过 EXIF 转正、净化、缩放后的照片像素，与 `image.width/height` 及 `/image` 相符，不是上传前原文件坐标。
- 时间使用带时区的 ISO 8601。记录 ID 为随机 32 位十六进制字符串。
- API 响应为 `Cache-Control: no-store`。未知 `/api/*` 返回 JSON 404，不回退为前端 HTML。

## 健康状态

`GET /api/health`

```text
{
  status: "ok", service: "catpaincheck",
  deployment: "local" | "cloud", max_upload_bytes: number,
  model: {
    ready: boolean, detail: string, version: string,
    engines?: {localization: ModelStatus, yolo: ModelStatus, auto_fgs: ModelStatus}
  }
}
```

`ModelStatus={ready,detail,version}`。`ready` 表示模型文件与依赖准备状态，不是单张照片的识别成功或精度证据，不包含预测。

## 上传与评分前 metadata

`POST /api/analyses`，multipart：

| 字段 | 类型 | 默认 |
|---|---|---|
| `image` | JPG / PNG / WebP 文件，必填 | — |
| `cat_name` | 字符串，最长 80 字 | `""` |
| `cat_id` | 可选 UUID 字符串 | 空，保存为 `null` |
| `training_allowed` | 表单布尔 | `false` |

`cat_id` 服务端规范化为 UUID hex。前端按本浏览器的猫名映射生成，属于分组提示，不认证真实猫身份，不作为已清洗的同猫划分。

本地文件上限 12 MiB，云端 4 MiB；前端接受最多 12 MiB 原图，必要时压缩云端上传体。服务端验证静态文件、最小 64×64、最多 2500 万像素，转正去元数据并保存为 RGB JPEG，最长边不超过 2560。

返回 202 `AnalysisMeta`；`GET /api/analyses/{id}` 返回同结构：

```text
{
  id, created_at, cat_name, image_url,
  status: "processing" | "ready" | "error",
  image: {width,height}, message: string,
  localization: Localization | null,
  collection: CollectionInfo | null
}
```

`AnalysisMeta` 通过字段白名单构造，**不得包含 `ai`、`machines`、自动 FGS、痛/不痛类别、原始 logits 或疼痛检测分数**。后台已处理完也不能泄漏。允许返回独立猫种检测/猫脸定位几何及其检测分数，这些不属于疼痛分类输出。

本地以后台任务处理；模型失败或不可用一般仍置 `ready`，方便提交人工评分。云端在上传请求内同步完成推理后返回 `ready`，持久记录与照片在私有 Blob；不依赖响应后的后台进程或临时目录存活。

### Localization

```text
{
  status: "processing" | "ok" | "unavailable" | "error",
  reason, note, model_version,
  coordinate_space: "image_pixels",
  bbox: Bbox | null, cat_bbox: Bbox | null,
  detector_confidence: number | null,
  points: [{id:1..48,x:number,y:number,confidence:number|null}]
}
```

`ok` 必须有合法猫脸框及 48 个唯一编号；其他状态不返回预测点。界面编号 `1..48` 对应上游 CatFLW `0..47 + 1`。回归器没有逐点可见性或置信度输出，点的 `confidence` 保持 `null`，不能补造误差。`detector_confidence` 是独立 COCO 猫检测分数，不是疼痛概率。

### CollectionInfo

```text
{
  cat_id: string | null,
  training_allowed: boolean,
  permission_version: "photo-fgs-training-v1",
  permission_at: string | null
}
```

旧记录可为 `collection:null`，不视作已经授权。

## 提交人工评分与取报告

`POST /api/analyses/{id}/scores`

```json
{"scores":{"ears":0,"eyes":1,"muzzle":null,"whiskers":0,"head":1},"notes":"选填"}
```

备注最长 2000 字。额外评分字段被拒绝。返回 `Report`；处理中返回 409、非法输入 422、记录不存在 404。

`GET /api/analyses/{id}/report` 在首次人工评分前返回 404；提交后返回当前报告。`GET /api/analyses/{id}/image` 提供净化照片；`/crop` 仅在人工评分提交后且裁剪存在时可读。

```text
Report = {
  schema_version:1,id,created_at,completed_at,cat_name,image_url,image:{width,height},notes,
  localization: Localization | null,
  machines: MachineResults | null,
  collection: CollectionInfo | null,
  annotation: AnnotationInfo | null,
  ai: LegacyAi,
  fgs: {
    scores:{ears,eyes,muzzle,whiskers,head},total,max,assessable_count,
    normalized:number|null,complete:boolean,threshold:4,
    flag:"positive"|"negative"|"incomplete"
  },
  conclusion:{code,title,summary,recommendations:string[]}
}
```

`fgs` 永远是人工评分；不得把机器分数预填或覆写进去。`normalized=total/max`，全 NA 时为 `null`。标准化分数不是疼痛概率。

### 双模型字段

```text
MachineResults = {
  prediction_id:string,created_at:string,
  yolo: {
    status:"ok"|"unavailable"|"error",
    label:"pain"|"no_pain"|null,
    detection_confidence:number|null,
    bbox:Bbox|null,
    detections:[{bbox:Bbox,label:"pain"|"no_pain",confidence:number}],
    model_version,preprocess_version,note,confidence_threshold:number
  },
  auto_fgs: {
    status:"ok"|"partial"|"unavailable"|"error",
    model_version,preprocess_version,note,
    scores:{ears,eyes,muzzle,whiskers,head},
    raw_outputs:{[FGSKey]:[number,number,number]},
    regions:{[FGSKey]:Bbox},
    total,max,assessable_count,complete,
    flag:"positive"|"negative"|"incomplete"
  }
}
```

YOLO 当前检测阈值为 0.25，768 RGB letterbox 输入；类别分数不是校准后的疼痛概率。未通过独立单猫猫脸定位、无检测、多个候选或区域不一致时不得接受分类。`status!=ok` 时主 `label` 和 `detection_confidence` 必须为 `null`；`detections` 可能保留诊断候选，仅在报告阶段可见。

自动 FGS 每个有效 `raw_outputs` 是三维原始 logits，分数为 argmax 的 0/1/2；不是概率或可见性置信度。`regions` 当前适配器为空对象，保留扩展位，不表示已确认原图各局部框。缺失模型、无效区域、异常输出记 NA。schema 重新计算总分和覆盖，不信任调用者传入的累计值。原模型无逐项可见性头，不能把“模型输出了分数”当作“区域肯定可见”。

### 旧字段与旧报告兼容

```text
LegacyAi = {
  status:"ok"|"unavailable"|"error",label:"pain"|"no_pain"|null,
  raw_score:number|null,threshold:number,model_version,note,
  crop_url:string|null,bbox:Bbox|null,
  localization?:Localization|null,machines?:MachineResults|null
}
```

新推理的 `ai` 是兼容封套，通常 `status=unavailable,label=null,raw_score=null`，同时 `machines` 有效。**存在 `machines` 时前端以双模型字段展示，不能把兼容封套当成全站“本次无法分析”。** 为兼容存储，`ai.machines` 可能与报告顶层 `machines` 同时存在。旧报告没有 `machines`，仍使用原单模型字段及旧规则；不重新生成历史预测。

### 标注与不可覆盖历史

```text
AnnotationInfo = {
  annotation_id,submitted_at,label_source:"user_fgs",rubric_version:"fgs-five-au-v1",
  saw_ai_before_scoring:boolean,
  supersedes_annotation_id:string|null
}
```

每次提交追加标注，不覆写首次人工分数。首次记录标为 `saw_ai_before_scoring=false`；存在此前报告或标注的重复提交保守标为 `true`，并链接上一条。当前报告是最新快照。SQLite 追加预测、标注和授权事件；私有 Blob 使用独立不可覆盖历史对象，首次标注槽不可覆盖。这里只记录本次交互顺序，不验证用户在记录之外的行为。

## 训练授权

`POST /api/analyses/{id}/training-permission`

```json
{"allowed":false}
```

`allowed` 必须是严格 JSON 布尔，返回 **`CollectionInfo` 本体**，不是 Report。可在上传后更新；报告刷新时使用最新授权，不受旧报告快照影响。前端仅在服务端成功后显示撤回；失败可重试。

撤回使后续候选导出排除记录，不删除报告或已经导出的离线副本。训练前重新导出，旧副本需由运营者按最新授权重新整理。无公开批量训练数据导出、标注历史列表或训练 API。

## 确定性三份比较

当前实现 `backend/app/scoring.py::compare_three_results`：

| 条件 | code |
|---|---|
| 至少一份有效结果提示关注，且有相反有效结果 | `dual_discordant` |
| 至少一份有效结果提示关注，无相反有效结果，允许其他缺失 | `dual_positive` |
| 无有效阳性，至少一份缺失/FGS不完整 | `dual_incomplete` |
| 三份全部有效且均阴性 | `dual_negative` |

FGS 阳性须五项完整且 ≥4；不完整累计 ≥4 只在文字保留关注提示，不当完整 FGS 分类。`dual_unavailable` 为 schema 保留 code，当前比较函数不产生。任何结果均不采用多数票、加权平均、准确率或疼痛概率。

旧 code 保持兼容：`agreement_positive, fgs_positive, discordant, agreement_negative, incomplete, ai_unavailable`；旧规则版本为 `fgs-comparison-v1`。

## 推理与运营者导出接口

默认 `backend/app/dual_inference.py` 暴露：

```python
get_model_status() -> dict
analyze_image(image_path: Path, crop_path: Path) -> dict
```

返回值匹配 `AiResult`，含独立 `localization`、`machines`，由 API 补 `crop_url`。两个机器引擎隔离异常，某一路失败不伪造另一路。源权重、依赖缺失、校验失败或推理出错必须明确返回不可用/错误，禁止任意回退分数。旧 `inference.py` 与 `cloud_inference.py` 仅为兼容模块。

`Store.training_candidates()` 与 `BlobStore.training_candidates()` 只供运营者 CLI 使用；`scripts/export-training-data.py --output NEW_DIR [--copy-images] [--cloud]` 导出当前授权的首次盲评弱标签。云端需环境变量 `BLOB_READ_WRITE_TOKEN`。候选不含猫名、备注，不把机器预测作为真值；包含照片 SHA256、授权版本、模型/预处理版本和原始人工分数。全 NA 或已知非盲评首次标注不导出。

输出 `dataset.json`、`candidates.jsonl` 及可选 `images/`，已有目录不覆盖。`cat_id` 仅 `browser_provided`，全部 `split=unassigned`。脚本不训练、不自动去除近重复或按猫划分，不给出独立准确率。
