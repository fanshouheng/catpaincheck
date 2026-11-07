"""Deterministic FGS calculation and transparent v1 comparison rules."""
from __future__ import annotations

from .schemas import AiResult, Conclusion, FgsChoices, FgsResult


RULE_VERSION = "fgs-comparison-v1"
DUAL_RULE_VERSION = "fgs-three-way-v2"


def calculate_fgs(scores: FgsChoices) -> FgsResult:
    observed = [v for v in scores.model_dump().values() if v is not None]
    count = len(observed)
    total = sum(observed)
    complete = count == 5
    return FgsResult(
        scores=scores,
        total=total,
        max=2 * count,
        assessable_count=count,
        normalized=total / (2 * count) if count else None,
        complete=complete,
        flag=("positive" if total >= 4 else "negative") if complete else "incomplete",
    )


def combine_results(ai: AiResult, fgs: FgsResult) -> Conclusion:
    if ai.machines is not None:
        return compare_three_results(ai, fgs)
    if not fgs.complete:
        concern = fgs.total >= 4 or (ai.status == "ok" and ai.label == "pain")
        return Conclusion(
            code="incomplete",
            title="评分信息不完整，请补拍或复评",
            summary=(
                f"FGS 有 {fgs.assessable_count}/5 项可以判断；缺失项未记为 0。"
                "本次仅提供有效项目的部分评分，不套用完整五项的 4/10 规则。"
                + ("已有评分或模型提示值得关注，不能因缺项而忽略。" if concern else "")
            ),
            recommendations=[
                "在自然光下重新拍摄，让双耳、双眼、口鼻、胡须和肩部清楚入镜。",
                "对照 FGS 标准，补充目前选择“看不清”的项目；无法判断时保留缺项。",
                "同时观察食欲、活动、躲藏及触碰反应；若出现明显异常，联系兽医。",
            ],
        )

    if ai.status != "ok":
        positive = fgs.flag == "positive"
        return Conclusion(
            code="ai_unavailable",
            title="FGS 达到关注参考值，AI 本次无法判断" if positive else "FGS 评分已完成，AI 本次无法判断",
            summary=(
                f"人工 FGS 为 {fgs.total}/10，"
                + ("达到 ≥4/10 的关注参考值。" if positive else "未达到 ≥4/10 的关注参考值。")
                + "AI 未提供有效判断，不能据此称为双重确认。"
            ),
            recommendations=[
                "保留本次人工评分，查看 AI 无法判断的原因后补拍或重试。",
                "结合食欲、活动和行为变化观察；持续或明显异常时联系兽医。",
            ],
        )

    if fgs.flag == "positive" and ai.label == "pain":
        return Conclusion(
            code="agreement_positive",
            title="两项均提示疼痛表情，需要关注",
            summary=f"AI 初筛提示疼痛迹象；人工 FGS 为 {fgs.total}/10，达到 ≥4/10 的关注参考值。",
            recommendations=[
                "结合食欲、活动、躲藏和触碰反应，联系兽医说明观察到的变化。",
                "保存本次记录，必要时在相近拍摄条件下复评以便比较。",
            ],
        )

    if fgs.flag == "positive":
        return Conclusion(
            code="fgs_positive",
            title="FGS 提示疼痛表情，请保留关注",
            summary=(f"人工 FGS 为 {fgs.total}/10，达到关注参考值；AI 初筛未提示疼痛。"
                     "两项结果不同，AI 的阴性结果不抵消人工评分提示。"),
            recommendations=[
                "对照标准复核五项评分，并结合行为变化联系兽医。",
                "补拍清晰、自然状态的照片，避免闪光、遮挡和过度侧脸。",
            ],
        )

    if ai.label == "pain":
        return Conclusion(
            code="discordant",
            title="两项判断不一致，建议复评",
            summary=(f"AI 初筛提示疼痛迹象；人工 FGS 为 {fgs.total}/10，未达到关注参考值。"
                     "本次不强行合并成确定的有痛或无痛判断。"),
            recommendations=[
                "对照标准重新观察眼睛、耳位、口鼻、胡须和头部姿态。",
                "待猫清醒且自然放松时补拍；持续出现异常行为时联系兽医。",
            ],
        )

    return Conclusion(
        code="agreement_negative",
        title="本次未见明显疼痛表情",
        summary=(f"AI 初筛未提示疼痛；人工 FGS 为 {fgs.total}/10，未达到关注参考值。"
                 "该结果描述本次照片中的表情，不能排除照片外的疼痛或其他异常。"),
        recommendations=[
            "继续观察食欲、活动和行为；有变化时在相似条件下重新记录。",
            "若已有持续不适或其他明显异常，不因本次低分而推迟联系兽医。",
        ],
    )


def compare_three_results(ai: AiResult, fgs: FgsResult) -> Conclusion:
    """Keep three observations separate; shared errors are not extra votes."""
    machines = ai.machines
    assert machines is not None
    automatic, yolo = machines.auto_fgs, machines.yolo
    signals = {
        "人工 FGS": fgs.flag == "positive" if fgs.complete else None,
        "自动 FGS": automatic.flag == "positive" if automatic.complete else None,
        "YOLO": yolo.label == "pain" if yolo.status == "ok" else None,
    }
    positives = [name for name, value in signals.items() if value is True]
    negatives = [name for name, value in signals.items() if value is False]
    missing = [name for name, value in signals.items() if value is None]
    summaries = [
        f"人工 FGS {fgs.total}/{fgs.max}（{fgs.assessable_count}/5 项）；",
        f"自动 FGS {automatic.total}/{automatic.max}（{automatic.assessable_count}/5 项）；",
        "YOLO " + ("提示疼痛表情。" if signals["YOLO"] is True else "未提示疼痛表情。" if signals["YOLO"] is False else "本次无法判断。"),
    ]
    if positives:
        code = "dual_discordant" if negatives else "dual_positive"
        title = "结果存在分歧，请保留关注并复评" if negatives else "已有结果提示疼痛表情，需要关注"
        summaries.append("、".join(positives) + "达到关注条件，其他结果不抵消该提示。")
    elif missing:
        code, title = "dual_incomplete", "部分结果不完整，请补拍或复评"
        if fgs.total >= 4 or automatic.total >= 4:
            summaries.append("已有项目累计达到 4 分，不能因缺项而忽略；未套用完整五项的 4/10 规则。")
    else:
        code, title = "dual_negative", "本次三份结果均未提示明显疼痛表情"
        summaries.append("本次结果不排除照片之外的疼痛或其他异常。")
    if missing:
        summaries.append("、".join(missing) + "缺少有效判断；NA 未记为 0。")
    summaries.append("三份结果仅作对照，不作多数投票，也不转换为疼痛概率。")
    return Conclusion(
        code=code, title=title, summary="".join(summaries),
        recommendations=[
            "对照原图复核存在分歧或无法判断的项目，清醒、自然状态下补拍双耳、双眼、口鼻、胡须及肩部。",
            "结合食欲、活动和行为变化观察；持续或明显异常时联系兽医。",
        ],
    )
