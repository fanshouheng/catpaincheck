# Third-party model sources and notices

本项目用于免费、非商业的研究复现与表情观察。以下记录区分来源、已有条款和未明确授权的部分；它不是替第三方新发的许可证。公开可下载、注明出处或没有收费，均不自动构成额外授权。模型权重不提交到本项目 Git，安装时由固定来源下载并核验。

核查日期：2026-11-07。更完整的原始来源及替代候选调查见 [模型来源调查](docs/model-origin-review.md)。本文件不以 star 数判断原创性或准确性。

## KittyCheck 自动 FGS

- 项目：[rmcaoile/KittyCheck](https://github.com/rmcaoile/KittyCheck)。固定提交 `818d4f5943f707e9fd3ec4ec420461c54e882832`。
- README 署名 Ralph Philip M. Caoile；报告作者 Ralph Philip M. Caoile、Rodolfo C. Camaclang。作者 [CMSC190 项目报告](https://github.com/rmcaoile/KittyCheck/blob/818d4f5943f707e9fd3ec4ec420461c54e882832/Journal_Paper_CMSC190_RMCaoile.pdf) 是本科 Special Problem 资料，不能表述为经过同行评审的临床验证论文。
- 作者说明使用 TIMM 预训练 EfficientNetB0，在 PyTorch 中训练，经过 ONNX / ONNX2TF 转为 TFLite；本站复现 [上游推理适配器](https://github.com/rmcaoile/KittyCheck/blob/818d4f5943f707e9fd3ec4ec420461c54e882832/lib/services/ai_scoring_service.dart) 的区域处理与五项分类方式。使用预训练骨干是迁移学习，不能仅据此推断抄袭。未找到直接复制 dominika01 疼痛权重的证据；缺少训练脚本与完整标签 CSV 也使全部原创和完整可复现性无法认证。
- **许可现状：该固定仓库未见 LICENSE，不能自行声明为 MIT、Apache 或已经取得作者使用/再分发许可。** 已检查的耳评分模型没有 author/license 元数据；这不改变仓库缺少明确许可的事实。本站的免费非商业用途及本说明不补足缺失授权。
- 作者报告称数据来自 CatFLW、Crawford 及其他公开来源，按 FGS 标注；不能据公开资料推断所有标签均经兽医确认，也不能把上游报告指标当作本站准确率。

安装脚本 `scripts/setup-kittycheck.py` 下载 `assets/models/` 下的七个模型，记录到 `models/kittycheck/manifest.json`。固定文件校验如下：

| 文件 | bytes | SHA256 |
|---|---:|---|
| `ears_scorer.tflite` | 16689496 | `e2edca443ed5c928dbf6b1328472b98aa88686a9d82f3e7a0d4af7355b918825` |
| `eyes_scorer.tflite` | 16689496 | `ae8932ca4404db5885d34b4cca5a70e58421e9127f844794aea53f46bc34a56f` |
| `face_detector.tflite` | 16689756 | `95f3ef0042a5f3c5d63c7bab82dbfe2072bd9a338ea8b5db34a0a1fe006ea96a` |
| `head_scorer.tflite` | 16689496 | `4b02ed832f7beb7e10d31b85abc14ccdbdd061442a80b7f9019cc29bf0a06f05` |
| `muzzle_scorer.tflite` | 16689496 | `88ad0785bb46e840b7458c0cf4a70ee46562fdb8937c474eb8b9d78731ef7355` |
| `region_detector.tflite` | 16691320 | `39dd520796984887326ac692b2b2fd76a82a3b00d6e59c9e6c63a35fa9461d13` |
| `whiskers_scorer.tflite` | 16689500 | `a82f4980f66f8d840e43f180fd6eed69e67b306ef72b7f0f6c8452193f110ef1` |

本站修改：Python/NumPy/LiteRT 运行，增加独立单猫门控，移除上游开发用固定训练样例坐标读取；无效区域或评分器失败为 NA，不使用上游零分回退。保留区域旋转和头位完整画面红框，但尚未证明与 Dart 逐像素相同，独立标注预处理版本。没有为模型增加真实可见性头或校准疼痛概率。

## YOLO 猫脸疼痛表情二分类

- 项目：[sophiapyx/IAT360-Computer-Vision-Project](https://github.com/sophiapyx/IAT360-Computer-Vision-Project)，固定提交 `dd3144e38ac414cc720d34c81f711ad231d67467`。
- 原权重：[best.pt](https://github.com/sophiapyx/IAT360-Computer-Vision-Project/blob/dd3144e38ac414cc720d34c81f711ad231d67467/best.pt)，6,233,514 bytes，SHA256 `c1e73b7d141e3017c032096beb7a8491ec295b5bc221b6d5a4d4a508704be36d`。
- 训练 notebook 和 checkpoint 互相印证：从官方 `yolov8n.pt` 微调，Ultralytics 版本 8.3.221，训练输入 768，类别 `0=no_pain / 1=pain`。这是官方通用预训练骨干上的任务微调，不是从零发明 YOLO，也没有证据表明直接复制另一套疼痛权重。
- **许可记录：实际 checkpoint 的 `license` 字段为 `AGPL-3.0 (https://ultralytics.com/license)`；源码仓库没有单独 LICENSE。** 不能将其简单写成“毫无许可”，也不能据该元数据保证全部自训增量、标注与来源权利已经明确。Ultralytics 的现有条款应保留并遵守；作者未明确的部分仍未由本站补发授权。
- 许可来源：[Ultralytics licensing](https://www.ultralytics.com/license)、[Ultralytics 8.3.221 LICENSE](https://github.com/ultralytics/ultralytics/blob/v8.3.221/LICENSE)。AGPL 条件不因免费网站而消失；AGPL 涉及网络交互及对应源代码的要求，不能把只写致谢当作履约。
- 原项目自按 FGS 标成两类；没有核到兽医标签或按猫个体隔离划分的证据。README 说明曾修正增广近重复数据问题；该说明不能替代本站独立无泄漏验证。

本站 `scripts/export-yolo-model.py` 固定源 SHA256，只允许已知模型类型并以 `weights_only=True` 加载导出；运行网站使用 ONNX/OpenCV，不加载 `.pt`。导出的 `models/yolo-model.json` 记录源提交、源哈希、checkpoint 许可字段、输出哈希和类别。转换格式不改变适用许可。

输入为 RGB、768 letterbox、填充值 114、float32 /255；检测阈值 0.25。增加独立猫脸区域一致性门控，多候选或失败不判痛/不痛。检测分数不是疼痛概率，不从二分类伪造五项 FGS。

## 48 点与猫脸定位：CatFLW 衍生权重

来源：[Hugo Cornellier 的 cat-face-landmarks](https://huggingface.co/hugocornellier/cat-face-landmarks)，固定提交 `8404a7a9a98ba7752a48f55dfba0f957381ee775`。本站使用：

| 文件 | SHA256 |
|---|---|
| `cat_face_landmarks_full.tflite` | `1a8b09c4b30a8211de89979244ab90e595935c27fdba1e64b433990f8a3a5b66` |
| `cat_face_localizer.tflite` | `a9621320ba8ca39b706c82649737c3cb72ef10ba37200ee91e84346600bee0c7` |

**权重为 CC BY-NC 4.0，源码 Apache-2.0 的说明不能覆盖权重许可。** 下载器同时保留上游权重声明到 `models/catflw-LICENSE.txt`，不将模型、源码和数据混成一个许可。

引用：Martvel et al., *CatFLW* (2023)；Martvel et al., *Automated Detection of Cat Facial Landmarks*, International Journal of Computer Vision (2024)。完整作者条目及论文链接以 [CatFLW 数据集](https://github.com/martvelge/CatFLW)和上游模型 README 为准。原始数据集由 Tech4Animals Lab, University of Haifa 的作者提供；本站使用衍生模型，不将整个数据集随仓库再分发。

保留的上游权重声明如下：

```text
Trained model weights: Creative Commons Attribution-NonCommercial 4.0
International (CC BY-NC 4.0)

Copyright 2026 Hugo Cornellier

This license applies to the TRAINED MODEL WEIGHTS produced by this repository,
including every .tflite, .keras, and .weights.h5 file, wherever they are
distributed. It does not apply to the source code in this repository, which is
licensed under the Apache License 2.0; see LICENSE.

Full license text:
https://creativecommons.org/licenses/by-nc/4.0/legalcode
Summary:
https://creativecommons.org/licenses/by-nc/4.0/

WHY THE WEIGHTS ARE NON-COMMERCIAL

These weights were trained on the CatFLW dataset, which is licensed CC BY-NC 4.0
by its authors (Martvel et al., Tech4Animals Lab, University of Haifa). The
dataset authors were asked directly how they wished weights derived from their
annotations to be licensed, and asked that those weights be released under
CC BY-NC 4.0 to remain consistent with the non-commercial terms of the source
data. They granted permission to publish the weights on that basis.

The weights are therefore offered for non-commercial use only. If you want to
use them commercially, that permission is not mine alone to give: contact the
dataset authors.

ATTRIBUTION

Any use or redistribution of these weights must retain this notice and cite the
CatFLW papers. The citations are in the README of this repository.

Dataset: https://github.com/martvelge/CatFLW
Kaggle:  https://www.kaggle.com/datasets/georgemartvel/catflw
```

本站在上游预测外增加猫 ROI、坐标还原和几何检查，界面编号为原编号 +1；没有提供模型本身没有的逐点可见性或误差。该模型只定位，不是疼痛分类器或 FGS 评分器。

## 通用猫检测门控

来源为 TensorFlow 发布的 [EfficientDet-Lite0 COCO TFLite 文件](https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_efficientdet_lite0_detection_metadata_1.tflite)。SHA256 `2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b`，4,563,519 bytes；下载来源和完整校验见 `scripts/setup-localization.py`。

只用于独立确认猫目标，不能判断疼痛。模型和运行依赖保持各自上游条款；本站没有把通用检测分数解释为疼痛概率。

## 旧 ResNet18 兼容资源

旧记录使用 [tongkasuyo/cat-pain-recog](https://github.com/tongkasuyo/cat-pain-recog)，提交 `b48eeee42596e26537fe4cc15bdf7a7954d2dac6`，源文件 `cat-pr-rn18-adamW-FL.pth`，SHA256 `4ace5922b92e8c89c14b4d275cd41d4b1b9884de14bf8c14bac41e32c8c73581`。旧脚本及模块为兼容保留，当前默认双模型不使用它。

该仓库核查时未提供明确项目 LICENSE。本站替换了其外部关键点 API 及裁剪流程，改变输入分布，未验证诊断准确率；旧代码或记录不能作为自动 FGS 五项模型的依据。原权重同样不随本项目 Git 分发。

## FGS 资料

评分依据为 [Feline Grimace Scale 官方资料](https://www.felinegrimacescale.com/)和 Evangelista et al. (2019), *Facial expressions of pain in cats: the development and validation of a Feline Grimace Scale* ([公开论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6911058/))。本站自行实现交互和结果对照，不声称得到 FGS 作者背书，不把用户或机器分数等同于兽医诊断。
