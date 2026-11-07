from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Score = Annotated[int, Field(strict=True, ge=0, le=2)] | None


class FgsChoices(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ears: Score
    eyes: Score
    muzzle: Score
    whiskers: Score
    head: Score


class ScoreSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: FgsChoices
    notes: str = Field(default="", max_length=2000, strict=True)


class LandmarkPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(strict=True, ge=1, le=48)
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    # The regressor does not necessarily supply per-point confidence.
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class Localization(BaseModel):
    """Public image geometry, independent of the hidden pain prediction."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["processing", "ok", "unavailable", "error"]
    reason: str = ""
    note: str = ""
    model_version: str = "unknown"
    coordinate_space: Literal["image_pixels"] = "image_pixels"
    bbox: tuple[float, float, float, float] | None = None
    cat_bbox: tuple[float, float, float, float] | None = None
    detector_confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    points: list[LandmarkPoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_geometry(self) -> "Localization":
        for box in (self.bbox, self.cat_bbox):
            if box is not None:
                if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError("Invalid localization box")
        ids = [point.id for point in self.points]
        if len(set(ids)) != len(ids):
            raise ValueError("Landmark IDs must be unique")
        if self.status == "ok":
            if self.bbox is None or len(ids) != 48:
                raise ValueError("Successful localization requires a face box and 48 landmarks")
        elif self.points:
            raise ValueError("Unconfirmed localization cannot expose landmarks")
        return self


class YoloResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["ok", "unavailable", "error"]
    label: Literal["pain", "no_pain"] | None = None
    detection_confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    bbox: tuple[float, float, float, float] | None = None
    detections: list[dict] = Field(default_factory=list)
    model_version: str = "unknown"
    preprocess_version: str = "unknown"
    note: str = ""
    confidence_threshold: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_detection(self) -> "YoloResult":
        if self.bbox is not None and (not all(math.isfinite(v) for v in self.bbox) or self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]):
            raise ValueError("Invalid YOLO box")
        if self.status == "ok" and (self.label is None or self.detection_confidence is None):
            raise ValueError("Available YOLO result requires label and detection score")
        if self.status != "ok":
            self.label = None
            self.detection_confidence = None
        return self


def empty_choices() -> FgsChoices:
    return FgsChoices(ears=None, eyes=None, muzzle=None, whiskers=None, head=None)


class AutoFgsResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["ok", "partial", "unavailable", "error"]
    model_version: str = "unknown"
    preprocess_version: str = "unknown"
    note: str = ""
    scores: FgsChoices = Field(default_factory=empty_choices)
    raw_outputs: dict[str, list[float]] = Field(default_factory=dict)
    regions: dict[str, tuple[float, float, float, float]] = Field(default_factory=dict)
    total: int = 0
    max: int = 0
    assessable_count: int = 0
    complete: bool = False
    flag: Literal["positive", "negative", "incomplete"] = "incomplete"

    @model_validator(mode="after")
    def calculate_result(self) -> "AutoFgsResult":
        keys = set(self.scores.model_dump())
        for key, output in self.raw_outputs.items():
            if key not in keys or len(output) != 3 or not all(math.isfinite(v) for v in output):
                raise ValueError("Invalid automatic FGS logits")
        for key, box in self.regions.items():
            if key not in keys or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Invalid automatic FGS region")
        observed = [v for v in self.scores.model_dump().values() if v is not None]
        if self.status in {"unavailable", "error"} and observed:
            raise ValueError("Unavailable FGS result cannot contain scores")
        self.total = sum(observed)
        self.assessable_count = len(observed)
        self.max = len(observed) * 2
        self.complete = len(observed) == 5
        self.flag = ("positive" if self.total >= 4 else "negative") if self.complete else "incomplete"
        if self.status == "ok" and not self.complete:
            self.status = "partial" if observed else "unavailable"
        if self.status == "partial" and self.complete:
            self.status = "ok"
        return self


class MachineResults(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prediction_id: str
    created_at: str
    yolo: YoloResult
    auto_fgs: AutoFgsResult


class CollectionInfo(BaseModel):
    cat_id: str | None = None
    training_allowed: bool = False
    permission_version: str = "photo-fgs-training-v1"
    permission_at: str | None = None


class TrainingPermission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed: bool = Field(strict=True)


class AnnotationInfo(BaseModel):
    annotation_id: str
    submitted_at: str
    label_source: Literal["user_fgs"] = "user_fgs"
    rubric_version: str = "fgs-five-au-v1"
    saw_ai_before_scoring: bool
    supersedes_annotation_id: str | None = None


class AiResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["ok", "unavailable", "error"]
    label: Literal["pain", "no_pain"] | None = None
    raw_score: float | None = None
    threshold: float = 0.4
    model_version: str = "unknown"
    note: str = ""
    crop_url: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    localization: Localization | None = None
    machines: MachineResults | None = None

    @model_validator(mode="after")
    def valid_prediction(self) -> "AiResult":
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Invalid inference threshold")
        if self.status == "ok":
            if self.raw_score is None or not math.isfinite(self.raw_score):
                raise ValueError("An available prediction requires a finite score")
            if not 0 <= self.raw_score <= 1:
                raise ValueError("Model score is outside [0,1]")
            expected = "pain" if self.raw_score >= self.threshold else "no_pain"
            if self.label != expected:
                raise ValueError("Prediction label does not match model threshold")
        else:
            self.label = None
            self.raw_score = None
        if self.bbox is not None and not all(math.isfinite(v) for v in self.bbox):
            raise ValueError("Invalid crop coordinates")
        return self


class ImageSize(BaseModel):
    width: int
    height: int


class AnalysisMeta(BaseModel):
    id: str
    created_at: str
    cat_name: str
    image_url: str
    status: Literal["processing", "ready", "error"]
    image: ImageSize
    message: str = ""
    localization: Localization | None = None
    collection: CollectionInfo | None = None


class FgsResult(BaseModel):
    scores: FgsChoices
    total: int
    max: int
    assessable_count: int
    normalized: float | None
    complete: bool
    threshold: int = 4
    flag: Literal["positive", "negative", "incomplete"]


class Conclusion(BaseModel):
    code: Literal[
        "agreement_positive", "fgs_positive", "discordant", "agreement_negative",
        "incomplete", "ai_unavailable", "dual_positive", "dual_negative",
        "dual_discordant", "dual_incomplete", "dual_unavailable",
    ]
    title: str
    summary: str
    recommendations: list[str]


class Report(BaseModel):
    schema_version: Literal[1] = 1
    id: str
    created_at: str
    completed_at: str
    cat_name: str
    image_url: str
    image: ImageSize
    notes: str
    ai: AiResult
    localization: Localization | None = None
    machines: MachineResults | None = None
    collection: CollectionInfo | None = None
    annotation: AnnotationInfo | None = None
    fgs: FgsResult
    conclusion: Conclusion
