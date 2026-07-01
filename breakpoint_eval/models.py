from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


AnswerType = Literal[
    "extractive",
    "abstain",
    "refusal",
    "json",
    "tool_decision",
    "reasoned",
]


class CategorySpec(BaseModel):
    id: str
    name: str
    description: str
    failure_modes: list[str]
    eval_families: list[str]
    risk_level: Literal["low", "medium", "high"] = "medium"


class HiddenTrap(BaseModel):
    kind: str
    description: str
    trigger: str
    expected_behavior: str


class ExpectedAnswer(BaseModel):
    value: str
    answer_type: AnswerType
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    citations_required: bool = False


class RubricCriterion(BaseModel):
    name: str
    points: float = Field(ge=0)
    description: str


class GradingRubric(BaseModel):
    total_points: float = Field(default=10, gt=0)
    criteria: list[RubricCriterion]
    fail_conditions: list[str]


class AdversarialVariant(BaseModel):
    id: str
    mutation: str
    prompt: str
    expected_answer: str
    trap: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    family: str
    title: str
    task: str
    context: str
    expected_answer: ExpectedAnswer
    hidden_traps: list[HiddenTrap]
    rubric: GradingRubric
    adversarial_variants: list[AdversarialVariant] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def prompt(self) -> str:
        return f"{self.task}\n\nContext:\n{self.context}".strip()

    def to_dataset_row(self) -> dict[str, Any]:
        row = self.model_dump(mode="json")
        row["prompt"] = self.prompt
        row["expected"] = self.expected_answer.value
        row["variant_count"] = len(self.adversarial_variants)
        return row


class ModelVote(BaseModel):
    model_name: str
    passed: bool
    confidence: float = Field(ge=0, le=1)
    rationale: str


class ValidationReport(BaseModel):
    item_id: str
    votes: list[ModelVote]
    ambiguity_score: float = Field(ge=0, le=1)
    answerability_score: float = Field(ge=0, le=1)
    trap_coverage_score: float = Field(ge=0, le=1)
    format_score: float = Field(ge=0, le=1)
    passed: bool
    reasons: list[str] = Field(default_factory=list)

    @property
    def agreement(self) -> float:
        if not self.votes:
            return 0.0
        majority = sum(1 for vote in self.votes if vote.passed)
        return majority / len(self.votes)


class DatasetBundle(BaseModel):
    dataset_id: str
    items: list[EvalItem]
    validation_reports: list[ValidationReport]
    rejected_reports: list[ValidationReport] = Field(default_factory=list)
    metrics: dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def rows(self) -> list[dict[str, Any]]:
        return [item.to_dataset_row() for item in self.items]
