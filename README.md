# 猫咪疼痛观察站 · CatPainCheck

上传猫咪照片，对照 **YOLO 疼痛表情分类、自动 FGS 和用户人工 FGS**，查看猫脸定位、48 个预测关键点和观察报告。

[在线体验](https://catpaincheck.vercel.app) · [模型来源与许可](THIRD_PARTY_NOTICES.md) · [API 契约](API-CONTRACT.md) · [功能验证记录](docs/validation.md)

本站免费，以非商业研究和表情观察为目标。**目前只验证了功能与推理流程，尚无本项目独立的疼痛识别准确率或临床验证。** 输出记录照片中的表情线索，不能确定疼痛病因；低分也不能排除疼痛。

## 能做什么

- 上传单只猫的 JPG、PNG 或 WebP 照片，原图最大 12 MiB。
- 检测猫、定位猫脸，显示 48 个预测点、编号及像素坐标。
- 分别运行 YOLO 二分类和 KittyCheck 五项自动 FGS，无需图像分析 API Key。
- 用户先独立选择五项人工 FGS，提交前隐藏机器答案，提交后展示三份结果和逐项对照。
- 保留分歧与 NA，支持刷新恢复、PNG 报告及 JSON 导出。
- 可选择授权照片和首次人工评分用于后续研究，并在报告页撤回。

当前只支持单张静态照片，尚未实现视频分析、账号系统、兽医标签审核、自动训练或完整的同猫身份识别。

## Windows 快速开始

需要 **Python 3.12、Node.js 20.19+ 或 22.12+、Git**，并能够访问 GitHub、Hugging Face、PyPI、PyTorch 和 npm 下载源。以下脚本面向 Windows / PowerShell；未验证 Linux、macOS 的一键安装。

确认 `python --version` 为 3.12 后运行：

```powershell
git clone https://github.com/fanshouheng/catpaincheck.git
cd catpaincheck
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Python python
.\启动网站.cmd
```

浏览器打开 `http://127.0.0.1:8765/`。停止服务：

```powershell
.\停止网站.cmd
```

安装脚本创建项目内的 `.venv`，安装运行依赖，下载并核验定位和自动 FGS 权重；YOLO ONNX 缺失或校验失败时，额外安装 CPU PyTorch 和导出工具，从固定 `.pt` 权重导出 ONNX，最后构建前端。首次安装需要下载模型和依赖，耗时取决于网络和电脑性能；无需独立显卡。

日常推理使用 **OpenCV / ONNX 与 LiteRT / TFLite**，无需加载 PyTorch 或 Ultralytics。`ultralytics==8.3.221` 仅用于本地导出，以 `--no-deps` 安装，避免同时安装两份提供 `cv2` 的 OpenCV wheel。关闭网页不会停止后台服务，本地服务只绑定 `127.0.0.1`。

权重不随 Git 仓库提供。下载脚本固定上游提交、文件大小和 SHA256；公开下载地址不等同于取得使用或再分发许可，请先阅读[第三方说明](THIRD_PARTY_NOTICES.md)。

## 怎样使用

1. 猫清醒、状态自然时拍摄，尽量拍清双耳、双眼、口鼻和胡须，并留出肩部。避开睡觉、打哈欠和瞬间眨眼。
2. 上传照片，独立评估耳位、眼眶收紧、口鼻紧张、胡须位置及头部姿态。每项选择 `0 / 1 / 2 / NA`，没有预填答案；看不清时选择 NA。
3. 提交后查看自动 FGS、YOLO 分类和人工 FGS。五项分数不同时分别展示，不用机器答案改写人工评分。
4. 可下载 PNG / JSON。训练授权默认关闭；授权后可在报告中撤回。

本地照片在本机处理；云端照片在本站 Vercel Python 后端处理并存入本站私有 Blob。两种模式均不向第三方图像或多模态 API 发送照片。

## 模型分别判断什么

| 组成 | 输入与输出 | 作用与限制 |
|---|---|---|
| EfficientDet-Lite0 + CatFLW 定位 | 照片 → 猫目标、猫脸框与 48 点 | 展示与独立门控，不直接判断疼痛 |
| YOLOv8n 微调模型 | 原图 → `pain / no_pain` 及检测分数 | 独立二分类，检测分数不是疼痛概率 |
| KittyCheck | 原图 → 自身区域回归与五项 `0 / 1 / 2 / NA` | 五个分类头预测 FGS；没有逐项可见性判断头 |
| 用户人工 FGS | 用户独立观察 → 五项分数 | 保留首次评分，是未审核的观察记录，不是专家真值 |

48 点是模型回归的预测位置，界面编号为上游 `0–47` 加 1。原模型没有逐点置信度、可见性或实测误差，本站不虚构这些数值。**48 点没有通过几何阈值被换算成 FGS**，也没有被用来替代 KittyCheck 原有的区域输入。

KittyCheck 适配保留上游区域裁剪、旋转、RGB 归一化，以及头位使用完整画面加猫脸红框的输入方式；不读取上游开发阶段固定训练样例坐标。Python 与原 Dart 预处理尚未逐像素证明等价，适配器有独立版本。

无猫、多猫、定位不一致、权重缺失或推理异常时，相关结果标为不可用或 NA；两个引擎分别处理失败。原模型仍可能对眨眼、遮挡、角度、品种及毛色误判。定位或框检查通过，不代表所有部位都可评估。

## 评分与结果对照

五项均为整数 `0 / 1 / 2` 或 `null`（NA）。NA 不补 0：显示有效总分、分母 `2 × 可评估项数` 和覆盖项数，人工结果另保存标准化分数。

只有五项完整时，才采用 **FGS ≥4/10** 作为关注参考值。部分评分不按完整阈值判为阴性，也不把 FGS 换算成疼痛概率。

规则版本为 `fgs-three-way-v2`：

- 任一有效结果提示关注，就保留该信号；存在相反结果时提示分歧。
- 没有关注信号但存在缺项或不可用结果时，说明评估不完整。
- 只有三份均有效、均未提示关注时，描述为“本次未提示明显疼痛表情”。

三份结果不作多数投票或加权平均。一致不证明正确；持续的食欲、活动或行为异常仍需结合实际情况联系兽医。评分参考 [FGS 官方资料](https://www.felinegrimacescale.com/)和[原始验证研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC6911058/)。

## 模型来源

默认组合为 `yolo-kittycheck.dual-v1`。本项目是网站与推理适配实现，**不声称自行训练了下列第三方模型**。

| 模型 | 固定来源 | 许可记录 |
|---|---|---|
| YOLO 二分类 | [sophiapyx/IAT360-Computer-Vision-Project](https://github.com/sophiapyx/IAT360-Computer-Vision-Project)，`dd3144e3` | checkpoint 含 Ultralytics AGPL-3.0 元数据；作者仓库无单独 LICENSE，其他授权仍有未明确部分 |
| 自动 FGS | [rmcaoile/KittyCheck](https://github.com/rmcaoile/KittyCheck)，`818d4f59` | 上游未提供明确 LICENSE；模型与适配来源分别记录 |
| 猫脸与 48 点 | [hugocornellier/cat-face-landmarks](https://huggingface.co/hugocornellier/cat-face-landmarks)，`8404a7a9` | 衍生权重 CC BY-NC 4.0，不能与源码 Apache-2.0 混淆 |

YOLO 实际权重与 notebook 表明它从官方 `yolov8n.pt` 微调，输入 768×768，类别为 `0=no_pain / 1=pain`。KittyCheck 报告说明采用 EfficientNetB0 微调，公开七份 TFLite；其资料为本科 CMSC190 项目报告，训练脚本和完整标签未公开，不能称为同行评审临床验证。

低 star 不能判断原创性或准确率。目前未找到 KittyCheck 直接复制其他疼痛权重的证据，也不足以认证全部原创。其他候选及证据见[来源调查](docs/model-origin-review.md)；完整提交、哈希、作者、适配改动和原始声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

旧 ResNet18 模块和下载脚本仅为兼容历史记录保留，不参与默认双模型推理。

## 数据、授权与后续训练

本地数据在 `data/catpain.sqlite3` 及净化后的 JPEG 中；云端照片、记录和历史在私有 Vercel Blob 中。记录 ID 是随机访问入口，本项目没有账号鉴权或公开历史列表；请妥善保管报告链接。原图元数据会被去除，但画面中的人、文字和背景不会被自动打码。

预测、人工标注和授权变更分别保存历史。首次评分保留，后续修改不会覆盖它，再次提交保守标为可能已见机器答案。这里的“盲评”只指本条记录的交互顺序，不证明用户从未在其他途径看过模型答案。

训练授权默认 `false`。撤回成功后，后续候选导出会排除该记录；撤回不等同于删除观察记录，也不会删除已经下载的离线副本。目前没有自动训练或边使用边学习服务。

运营者可用本地 CLI 导出候选，输出目录必须尚不存在：

```powershell
# 只导出候选元数据
.venv/Scripts/python.exe scripts/export-training-data.py --output work/dataset-001
# 同时复制符合条件的照片
.venv/Scripts/python.exe scripts/export-training-data.py --output work/dataset-002 --copy-images
# 私有 Blob；需由运营者在环境中配置 BLOB_READ_WRITE_TOKEN
.venv/Scripts/python.exe scripts/export-training-data.py --cloud --output work/cloud-dataset-001 --copy-images
```

仅选择当前明确授权、首次记录未见机器答案、至少一项有效的人工 FGS；导出前逐条复查授权。不导出猫名和备注，不把模型预测当训练真值。候选保持 `weak_label=true`、`review_status=unreviewed`、`split=unassigned`。

`cat_id` 只是本浏览器按猫名保存的匿名分组提示，不能认证猫身份；SHA256 只能发现净化后完全相同的图片。实际训练前仍需核对同猫、近重复图、标签质量与当前授权，再按组划分训练、验证和测试集。目前没有完成这套审核与防泄漏拆分。

## 开发与检查

完成快速安装后，分别在两个终端启动：

```powershell
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8765
```

```powershell
npm --prefix frontend run dev
```

构建及测试：

```powershell
npm --prefix frontend run build
.venv/Scripts/python.exe -m pytest backend/tests -q
```

只核验模型，不重新下载：

```powershell
.venv/Scripts/python.exe scripts/setup-localization.py --check
.venv/Scripts/python.exe scripts/setup-kittycheck.py --check
.venv/Scripts/python.exe scripts/export-yolo-model.py --check
```

查看健康状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8765/api/health | ConvertTo-Json -Depth 8
```

正常时 `model.version` 为 `yolo-kittycheck.dual-v1`，`model.engines` 下的 `localization`、`yolo`、`auto_fgs` 均为 `ready=true`。健康检查只说明资源就绪，上传实测才能检查一次完整推理。

| 现象 | 检查方向 |
|---|---|
| 没有猫脸框或点位 | 确认只有一只猫、照片清楚且角度合适；查看定位原因，不强行套点 |
| 某个机器结果不可用 | 查看健康接口对应引擎，运行模型 `--check`；本地日志在 `data/server-error.log` |
| 自动与人工 FGS 不同 | 保留原答案，复核照片、眨眼、遮挡及具体项目；不要为消除分歧修改类别映射 |
| 下载或哈希检查失败 | 检查下载源和网络后重试安装；不要绕过文件校验 |
| 前端尚未构建 | 运行 `npm --prefix frontend ci` 和 `npm --prefix frontend run build` |

`CATPAIN_DATA_DIR`、`CATPAIN_MODELS_DIR` 可分别修改数据和模型路径。完整接口与历史字段见 [API-CONTRACT.md](API-CONTRACT.md)。

## 部署到自己的 Vercel

**仅从 GitHub Import 这个仓库还不够。** 仓库不包含权重，当前 Vercel 构建命令只安装、构建前端，不自动下载或导出模型。已验证的路径是先在本地完成安装与模型核验，再通过 CLI 上传部署。

1. 按快速开始完成本地安装，确认三引擎就绪。
2. 在自己的 Vercel 账户创建项目和 **Private Blob** 存储，为该项目准备 `BLOB_READ_WRITE_TOKEN`。
3. 在项目根目录登录、关联自己的项目、配置生产环境变量并部署：

```powershell
npx --yes vercel@65.12.1 login
npx --yes vercel@65.12.1 link
npx --yes vercel@65.12.1 env add BLOB_READ_WRITE_TOKEN production
npx --yes vercel@65.12.1 deploy --prod
```

`env add` 通过交互输入自己的 Blob 令牌；若已在控制台配置，不必重复添加。需要预览环境时另配对应环境变量。不要把令牌提交 Git；本仓库不提供运营站点的凭据、项目关联文件或用户数据。

`vercel.json` 将 `frontend/dist` 用作静态页面，`api/index.py` 用作同源 Python API，并打包本地准备好的 ONNX / TFLite。云端使用 Python 3.12、OpenCV 和 LiteRT，不安装 PyTorch / Ultralytics；上传请求内完成推理，不使用 SQLite 或临时目录作云端持久存储。

云端上传上限为 4 MiB；浏览器仍接受最大 12 MiB 的原图，需要时保留完整画面、校正方向并压缩，最长边不超过 2560。压缩可能损失细胡须信息。模型体积、构建资源和函数运行时限仍受自己的 Vercel 套餐约束，不保证所有账户或平台都能直接部署。

部署后检查 `/api/health`，再实际验证上传、评分前隐藏、人工评分、三份报告、刷新、授权撤回及下载。已有站点的[验证记录](docs/validation.md)不能替代新部署检查。

## 目录

```text
frontend/src/       React / TypeScript，人工评分、48 点、结果对照与导出
backend/app/        FastAPI，推理、计分、SQLite / 私有 Blob
backend/tests/      接口、计分、隐藏、授权和推理适配测试
scripts/           安装、下载、导出、启动与停止
docs/              公开的来源调查和功能验证摘要
models/            本地下载的模型与 manifest，不提交 Git
data/              私人照片、数据库和本地日志，不提交 Git
work/、outputs/    本地临时文件与验证产物，不提交 Git
```

## 开源许可与贡献

**本项目有权许可的原创代码采用 [GNU AGPL-3.0-only](LICENSE)**，版权与适用范围见 [NOTICE](NOTICE)。AGPL 允许按其条款使用、修改和再分发；本站免费、非商业的运营定位，不是对 AGPL 原创代码额外增加非商业限制。修改后通过网络提供服务时，需遵守 AGPL 对对应源代码的要求。

这项许可不覆盖第三方权重、数据或未经授权的上游部分。特别是 `backend/app/kittycheck.py` 包含对 KittyCheck Dart 适配器的移植，上游没有明确 LICENSE：本项目不能替上游为改编部分授予 AGPL，也不能据公开仓库声称完整程序的所有第三方权利均已澄清。具体来源、受影响文件及未明确部分见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

欢迎通过 Issue / PR 提交可复现的问题、定位或评分流程改进、测试与文档修订。提交代码时仅贡献有权提交的内容，原创贡献沿用上述许可并保留第三方声明；请使用公开可用或明确获准的测试材料，不在 Issue / PR 中上传未获准的照片、报告链接、数据库或密钥。
