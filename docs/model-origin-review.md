# 模型来源与替代候选核查

核查日期：2026-11-07。只读核查公开仓库、提交记录、模型文件目录和作者报告；未联系作者。星标数会变化。

## 结论

- 低 star 不能判断模型是否复制、准确或无效。本次 KittyCheck 为 **2 stars**，YOLO 项目为 **0 stars**。
- 两仓库 GitHub 元数据均为 `fork=false`、无 `parent/source`。这只表示它们不是 GitHub 记录的 fork，不能证明所有代码或权重原创。
- KittyCheck 说明采用预训练 EfficientNetB0 微调；YOLO 项目明确从官方 `yolov8n.pt` 微调。使用公开预训练骨干是正常迁移学习，不等于已有疼痛模型被原样复制。
- **未找到 KittyCheck 直接复制 dominika01/pain-detection-cats 权重或代码的证据**；也没有足够资料认证其全部原创。两者采用不同训练框架、部署格式及方法，KittyCheck 未公开训练脚本与标签 CSV，完整可复现性不足。
- 免费、非商业、注明出处不能替代使用许可。两仓库均无 LICENSE 文件；YOLO 权重另自带 Ultralytics AGPL-3.0 许可字段，需与仓库作者对其自训成果的授权状态分别记录，不能一概写为完全无许可，也不能据此声称取得单独作者授权。

## KittyCheck 来源

固定提交：`818d4f5943f707e9fd3ec4ec420461c54e882832`。

- README署名 **Ralph Philip M. Caoile**，报告作者为 **Ralph Philip M. Caoile、Rodolfo C. Camaclang**；第一页明确为菲律宾大学 Los Baños 计算机学士 CMSC190 Special Problem，不能称同行评审临床验证论文。
- 报告第4–6页说明：用 PyTorch/Torchvision/TIMM，预训练 EfficientNetB0 加自定义回归头或三分类头；训练后经 Torch ONNX、ONNX、ONNX2TF 转为 TFLite。五个 FGS 评分器分别训练。
- 报告明确引用 Martvel 等人的猫脸定位/关键点研究；它复用方法和数据，不声称 EfficientNet 或 CatFLW 为自己发明。
- CatFLW约2079图训练定位；1598张来自 CatFLW、Crawford及其他公开来源的图，由作者与两名协助标注者按 FGS 标注、取中位数。报告未明确三人均为兽医。
- 仓库有连续 Flutter 开发历史，包括初始占位模型、AI集成、裁剪和旋转修正，当前七份 TFLite 普通二进制文件各约16.7MB；不是只提供外部多模态API提示词的项目。
- 当前树未见 Python 训练脚本、训练notebook、原始PyTorch checkpoint或标签CSV；作者其他公开仓库未发现对应训练工程。公开说明与模型文件不能替代可复现训练过程。
- 对已下载 `ears_scorer.tflite` 静态解析：模型描述为 `onnx2tf flatbuffer_direct`，仅有 `reduced_precision_support = fp16accfp32` 元数据，未发现作者或许可字段。转换工具痕迹与报告吻合，但无法据此证明训练者身份或使用授权；其余六份未逐一检查元数据。
- 与 dominika 项目对照：后者公开 Keras/SavedModel、notebook及标签CSV；不同框架和模型格式不足以证实不存在借鉴，但未发现直接复制的证据。本次没有开展全网代码相似度或完整权重张量取证。

来源：

- [README及署名](https://github.com/rmcaoile/KittyCheck/blob/818d4f5943f707e9fd3ec4ec420461c54e882832/README.md)
- [作者研究报告](https://github.com/rmcaoile/KittyCheck/blob/818d4f5943f707e9fd3ec4ec420461c54e882832/Journal_Paper_CMSC190_RMCaoile.pdf)
- [开发提交历史](https://github.com/rmcaoile/KittyCheck/commits/main/)
- [实际推理代码](https://github.com/rmcaoile/KittyCheck/blob/818d4f5943f707e9fd3ec4ec420461c54e882832/lib/services/ai_scoring_service.dart)
- [对照项目 dominika01](https://github.com/dominika01/pain-detection-cats/tree/d33e1e2a0928615a59c2d4817a6ead4fe1c83e17)

## YOLO 来源与移植契约

仓库：`sophiapyx/IAT360-Computer-Vision-Project`，固定提交 `dd3144e38ac414cc720d34c81f711ad231d67467`。

- `best.pt` 位于仓库根目录，6,233,514 bytes。
- README自称计算机视觉项目，公开连续 Colab notebook 开发记录、训练代码和输出；基于 CatFLW 2079张，自按 FGS 将猫脸标为两类，另收60张测试图（实际8损坏后52张）。未找到兽医标签证据或按猫个体隔离的划分说明。
- Notebook明确 `YOLO('yolov8n.pt')` 初始化后进行训练；所以可确认采用官方通用预训练骨干进行疼痛任务微调，不能称从零发明 YOLO。
- `data.yaml` 类别映射：`0: no_pain`、`1: pain`。
- 最终 `cat_pain_v3` 训练参数：`imgsz=768, batch=8, epochs=40, patience=15`；早期实验是640。预测示例显式 `conf=0.25`；最终val/test调用未设置自定义置信阈值。
- Root对实际权重作安全 `weights_only` 加载：SHA256 `c1e73b7d141e3017c032096beb7a8491ec295b5bc221b6d5a4d4a508704be36d`；checkpoint记录版本 `8.3.221`、日期 `2025-10-26`、训练输入768、基础模型 `yolov8n.pt` 和类别映射，与公开notebook一致。其 `license` 字段为 `AGPL-3.0 (https://ultralytics.com/license)`。这些是来源一致性的直接证据，不是准确率验证。
- README承认一次增广数据包含验证/测试近重复图，称用pHash删去271张后才将剩余518张只用于训练。该修正不能替代猫个体隔离验证。
- 仓库未提供 LICENSE 文件；README引用CatFLW的 CC BY-NC 4.0 数据条款。框架及实际checkpoint包含Ultralytics AGPL-3.0信息；仓库作者是否另行明确授权其自训增量/标注数据仍需确认。本站README注明出处是必要记录，不能代替遵守已有许可和澄清缺失授权。

来源：

- [README及数据来源](https://github.com/sophiapyx/IAT360-Computer-Vision-Project/blob/dd3144e38ac414cc720d34c81f711ad231d67467/README.md)
- [训练notebook](https://github.com/sophiapyx/IAT360-Computer-Vision-Project/blob/dd3144e38ac414cc720d34c81f711ad231d67467/Computer_Vision_Project_Group_36.ipynb)
- [类别映射](https://github.com/sophiapyx/IAT360-Computer-Vision-Project/blob/dd3144e38ac414cc720d34c81f711ad231d67467/data.yaml)
- [实际权重](https://github.com/sophiapyx/IAT360-Computer-Vision-Project/blob/dd3144e38ac414cc720d34c81f711ad231d67467/best.pt)
- [Ultralytics许可说明](https://www.ultralytics.com/license)

## 再检索结果

额外检索 GitHub README 中的 `feline grimace weights`、`cat pain efficientnet`、`cat pain pytorch`，检查以下候选，**没有找到比当前候选更明确、同时具备公开疼痛权重和充分验证的直接替代品**：

- [Miffka/kittyface](https://github.com/Miffka/kittyface)：使用48点模型和几何比例，README明确称临时阈值的几何代理，**不应用FGS、不作疼痛评估**。不能把其0–2代理分数列为已训练自动FGS模型。
- [vwicky/cats-in-pain-main](https://github.com/vwicky/cats-in-pain-main)：行为/姿态与音频弱标签的论文研究工程，文中强调非临床真值；论文和数据链接仍为TODO，未见可直接替换的公开疼痛checkpoint。
- [xushiexpresso15/Cat_Emotion](https://github.com/xushiexpresso15/Cat_Emotion)：MIT代码及嵌入式情绪/压力项目；主分支描述硬件、情绪和压力映射，本次未核到经独立疼痛验证的五项FGS权重。不能把代码MIT等同于证明疼痛任务有效。

因此下一步仍适合先做 KittyCheck 自动FGS与YOLO的**独立本地对照复现**；来源说明保留作者、固定提交、模型哈希、数据来源、许可状态和“本站准确率尚未验证”，不以star数替代质量评估。
