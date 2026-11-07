# 猫咪疼痛观察站 · CatPainCheck

免费、非商业的猫咪疼痛表情观察网站。上传一张照片后，后端分别运行 **YOLO 二分类**和 **KittyCheck 自动 FGS**；用户先独立完成五项人工 FGS，提交后查看三份结果及逐项对照。模型在本机或本站后端运行，不调用第三方照片分析 API。

本项目处于研究复现与功能验证阶段，尚未建立本站独立疼痛识别准确率。免费、非商业和注明出处不自动补足第三方授权；模型来源、许可现状和引用见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 使用

此电脑完成安装后，双击 `启动网站.cmd`，打开 `http://127.0.0.1:8765`；双击 `停止网站.cmd` 停止后台服务。默认只绑定本机。关闭网页不会停止服务。

1. 上传单只猫的 JPG / PNG / WebP 照片，原图最大 12 MiB。清醒、自然光、正面拍摄，让双耳、双眼、口鼻、胡须和肩部入镜。
2. 独立选择耳位、眼眶收紧、口鼻紧张、胡须位置及头部姿态，每项 0 / 1 / 2 / NA，无预填答案。提交前只返回照片及猫脸几何，不返回机器标签、自动 FGS、检测分数或原始 logits。
3. 提交后显示自动 FGS、YOLO 分类和人工 FGS；五项分歧保留展示，不用机器答案改写人工答案。可下载 PNG 报告和 JSON，刷新可恢复当前记录。
4. 上传页的“允许本次照片及评分用于模型改进”默认关闭；授权后可在报告中撤回。

照片上的 48 点来自专用猫脸关键点回归器，是模型预测位置，不是人工确认点，也不自带逐点可见性、置信度或实测误差。48 点用于展示定位，不通过几何阈值冒充自动 FGS。

## 三份结果怎样比较

| 结果 | 含义 | 保留的限制 |
|---|---|---|
| YOLO | `pain` / `no_pain` 猫脸类别及检测分数 | 分数是检测输出，不是疼痛概率；不是五项 FGS |
| 自动 FGS | 五个分类头分别预测 0–2 分，失败项为 NA | 原模型没有可见性分类头，不能保证识别所有遮挡、眨眼或角度问题 |
| 人工 FGS | 用户独立观察五项 | 用户评分是未审核观察记录，不是兽医确认真值 |

NA 不补成 0。有效分为 `总分 / (2 × 可评估项数)`，同时显示覆盖项数；人工结果另保存标准化分数。仅五项完整时，使用 FGS ≥4/10 作为关注参考值。部分评分不按完整阈值判阴性。

综合逻辑版本为 `fgs-three-way-v2`：任何有效结果提示关注时保留该信号，有相反结果则提示分歧；没有阳性而存在缺项时提示不完整；只有三份均有效且均阴性时描述“本次未提示明显疼痛表情”。三份结果不作多数投票、加权平均或疼痛概率换算。一致也不证明判断正确，低分不排除照片外的疼痛。

FGS 参考：[官方培训资料](https://www.felinegrimacescale.com/)及 [原始验证研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC6911058/)。出现持续食欲、活动或行为异常时，结合实际情况联系兽医。

## 默认模型及来源

默认组合为 `yolo-kittycheck.dual-v1`，替代原单路 ResNet18；旧记录、旧报告字段及旧推理模块仍兼容，不把旧输出改造成五项 FGS。

- **YOLO**：[sophiapyx/IAT360-Computer-Vision-Project](https://github.com/sophiapyx/IAT360-Computer-Vision-Project)，提交 `dd3144e38ac414cc720d34c81f711ad231d67467`。官方 `yolov8n.pt` 微调，输入 768×768，`0=no_pain / 1=pain`。本站安全导出为 ONNX，以 OpenCV CPU 推理；网页运行时不反序列化上游 `.pt`。源权重 SHA256 为 `c1e73b7d141e3017c032096beb7a8491ec295b5bc221b6d5a4d4a508704be36d`。checkpoint 含 Ultralytics AGPL-3.0 元数据，仓库无单独 LICENSE，详见第三方说明。
- **自动 FGS**：[rmcaoile/KittyCheck](https://github.com/rmcaoile/KittyCheck)，提交 `818d4f5943f707e9fd3ec4ec420461c54e882832`。七个 LiteRT/TFLite 模型包含定位、区域回归和五项 EfficientNetB0 评分器。本站保留上游区域裁剪、对齐、归一化以及头位使用完整画面的输入方式；不读取上游开发阶段固定训练样例坐标，失败项改为 NA。Python 与原 Dart 预处理尚未逐像素证明等价。作者资料为本科 CMSC190 项目报告，不能称为同行评审临床验证；仓库未提供 LICENSE。
- **猫脸与 48 点**：[hugocornellier/cat-face-landmarks](https://huggingface.co/hugocornellier/cat-face-landmarks)，提交 `8404a7a9a98ba7752a48f55dfba0f957381ee775`，CatFLW 衍生权重为 CC BY-NC 4.0，另以 EfficientDet-Lite0 COCO 猫检测作门控。关键点不直接用于生成 KittyCheck 输入。

低 star 不能证明复制或有效。核查确认了预训练骨干和迁移学习来源，未找到 KittyCheck 直接复制其他疼痛权重的证据；训练脚本和完整标注未公开，也不足以认证全部原创。更详细证据及其他候选见 [模型来源调查](docs/model-origin-review.md)。

下载脚本固定来源提交、大小和 SHA256；模型安装后生成或保留来源 manifest。权重保存在 `models/`，不提交 Git。模型来源说明与下载脚本保留在代码仓库中，不替第三方模型补发许可。

## 数据与以后训练

本地模式使用 `data/catpain.sqlite3` 与净化后的 JPEG；云端模式使用私有 Vercel Blob，浏览器通过本站 API 读取。记录 ID 是随机访问入口；本项目没有账号或公开历史列表。

预测、每次人工提交和训练授权变更分别追加历史。首次人工评分不被后续修改覆盖；再次提交记为已可能看过机器结果，并保存 `supersedes_annotation_id`。当前报告可更新，原始预测和标注历史保留。这里的“盲评”是本条记录的交互顺序，不证明用户从未通过其他途径见过模型答案。

训练授权默认 `false`，撤回成功后后续导出排除该记录。撤回不自动删除观察记录，也不会追溯删除已经下载的离线副本；运营者在实际训练前必须重新导出并按最新授权整理已有副本。本项目目前没有训练任务或自动学习服务。

运营者可通过本地 CLI 导出弱标签候选，无公开批量训练数据导出 API：

```powershell
.venv/Scripts/python.exe scripts/export-training-data.py --output work/dataset-001
# 需要照片时显式加 --copy-images；输出目录必须尚不存在。
.venv/Scripts/python.exe scripts/export-training-data.py --output work/dataset-002 --copy-images
# 私有 Blob 导出；令牌由运营者在环境中自行配置，不放入命令或仓库。
.venv/Scripts/python.exe scripts/export-training-data.py --cloud --output work/cloud-dataset-001 --copy-images
```

脚本只选择当前明确授权、首次记录未见机器结果且至少一项有效的用户 FGS，生成 `dataset.json` 和 `candidates.jsonl`；不导出猫名、备注，不把模型预测当标签。所有样本保持 `unreviewed_user_weak_labels`、`split=unassigned`，脚本不训练、不自动划分训练/验证/测试集。

`cat_id` 仅是本浏览器按猫名生成的匿名分组提示，同名、改名、跨浏览器及无名照片可能误合并或分离，不等于真实猫身份。图片 SHA256 只能识别净化后完全相同的文件。训练前仍需人工或专门流程核对同猫、近重复图、标签质量和授权，再按猫及重复图分组划分；当前没有防泄漏拆分或独立准确率结论。

## 安装与开发

需要 Python 3.12、Node.js 20.19+ / 22.12+。项目使用 `.venv`，不修改全局 Python 包。完整安装入口为：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

安装脚本先准备运行依赖、定位与 KittyCheck；仅当 YOLO ONNX 缺失或校验失败时，才安装 CPU PyTorch/Torchvision 及 `backend/requirements-export.txt`，以 `--no-deps` 安装 `ultralytics==8.3.221`，然后导出 YOLO。这样避免额外安装另一份 OpenCV wheel。本站 ONNX/LiteRT 推理无需 Ultralytics 或 PyTorch。

各模型的独立安装/导出入口为 `scripts/setup-localization.py`、`scripts/setup-kittycheck.py`、`scripts/export-yolo-model.py`。旧 `scripts/setup-models.py` 与 `scripts/export-cloud-model.py` 仅用于兼容旧 ResNet18。

只检查已安装模型，不下载：

```powershell
.venv/Scripts/python.exe scripts/setup-localization.py --check
.venv/Scripts/python.exe scripts/setup-kittycheck.py --check
.venv/Scripts/python.exe scripts/export-yolo-model.py --check
```

开发时两个终端分别运行：

```powershell
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8765
```

```powershell
npm --prefix frontend run dev
```

构建及检查：

```powershell
npm --prefix frontend run build
.venv/Scripts/python.exe -m pytest backend/tests -q
```

生产页面构建到 `frontend/dist` 后由 FastAPI 提供。`CATPAIN_DATA_DIR`、`CATPAIN_MODELS_DIR` 可修改本地数据和模型目录。测试中的推理 fixture 只验证接口和规则；运行、导出数值检查及浏览器通过不等于诊断准确率。

```text
frontend/src/       React + TypeScript、人工盲评、48 点、三份对照及导出
backend/app/        FastAPI、双模型推理、规则、SQLite / 私有 Blob
backend/tests/      API、NA、隐藏机器结果、历史、授权及推理适配检查
scripts/           安装、哈希验证、模型导出、私有训练候选导出、启动/停止
models/            下载的第三方权重与 manifest，不提交 Git
data/              私人照片、数据库及本地日志，不提交 Git
outputs/           可审阅的验证证据
work/              临时验证文件，不提交 Git
```

## Vercel 与发布状态

网站地址为 [catpaincheck.vercel.app](https://catpaincheck.vercel.app)。2026-11-07 已部署 `yolo-kittycheck.dual-v1`，生产健康接口、真实上传与48点定位、评分前隐藏、人工评分、双模型对照、刷新恢复、授权撤回和 PNG/JSON 导出均已验证。这些是功能验证，不是疼痛识别准确率验证；之后修改须重新检查，不能沿用本次结果。

`vercel.json` 部署静态前端，`api/index.py` 提供同源 Python API，后端使用 ONNX/OpenCV 与 LiteRT，照片和记录保存到私有 Blob。云端在上传请求内完成推理，不依赖返回响应后的后台任务，也不把 SQLite 或 `/tmp` 当持久存储。

云端上传体限制为 4 MiB；浏览器仍接受最大 12 MiB 原图，必要时保留完整画面、纠正方向并压缩，最长边不超过 2560。压缩不保证细胡须信息无损。前端构建时使用 `VITE_CLOUD_DEPLOYMENT=true`，Python 版本与云端依赖见 `pyproject.toml`。环境令牌、`.env*`、`.vercel`、私人照片和源 `.pt/.pth` 不提交或作为公开文件分发。

部署前须准备新模型及 manifest，并在生产环境实际检查健康状态、上传、评分前机器结果隐藏、三份报告、刷新恢复、训练授权撤回和 PNG/JSON 导出；不能沿用旧单模型部署验证结果。
