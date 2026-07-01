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


ReviewStatus = Literal["pending", "approved", "rejected", "needs_changes"]
RunStatus = Literal["queued", "running", "passed", "failed", "needs_review"]
CheckType = Literal["exact", "contains", "json_schema", "llm_judge", "trace", "human"]
GateSeverity = Literal["info", "warning", "blocking"]


class Project(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetVersion(BaseModel):
    id: str
    project_id: str
    dataset_id: str
    version: str
    item_ids: list[str]
    parent_version: str | None = None
    lineage_notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class OracleSpec(BaseModel):
    type: str = "expected_answer"
    required_claims: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    evidence_rules: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.8, ge=0, le=1)


class MutationLineage(BaseModel):
    seed_id: str
    parent_case_id: str | None = None
    mutator: str = "base"
    generation: int = 0
    path: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    id: str
    suite_id: str
    eval_item: EvalItem
    original_failure_seed: str | None = None
    oracle: OracleSpec = Field(default_factory=OracleSpec)
    mutation_lineage: MutationLineage
    validation_report: ValidationReport | None = None
    failure_family: str
    review_status: ReviewStatus = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dataset_row(self) -> dict[str, Any]:
        row = self.eval_item.to_dataset_row()
        row.update(
            {
                "case_id": self.id,
                "suite_id": self.suite_id,
                "original_failure_seed": self.original_failure_seed,
                "failure_family": self.failure_family,
                "review_status": self.review_status,
                "mutation_lineage": self.mutation_lineage.model_dump(mode="json"),
                "oracle": self.oracle.model_dump(mode="json"),
            }
        )
        return row


class Check(BaseModel):
    id: str
    suite_id: str
    name: str
    check_type: CheckType
    config: dict[str, Any] = Field(default_factory=dict)
    severity: GateSeverity = "blocking"
    enabled: bool = True


class EvalSuite(BaseModel):
    id: str
    project_id: str
    name: str
    description: str
    dataset_version_id: str
    case_ids: list[str]
    checks: list[Check] = Field(default_factory=list)
    failure_families: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelOutput(BaseModel):
    id: str
    run_id: str
    case_id: str
    model_name: str
    output: str
    latency_ms: float = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    trace: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Judgment(BaseModel):
    id: str
    run_id: str
    case_id: str
    check_id: str
    passed: bool
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    rationale: str
    judge_name: str = "local"
    needs_human_review: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HumanReview(BaseModel):
    id: str
    case_id: str
    reviewer: str
    status: ReviewStatus
    notes: str = ""
    rubric_edits: dict[str, Any] = Field(default_factory=dict)
    ambiguity_flag: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FailureCluster(BaseModel):
    id: str
    project_id: str
    family: str
    title: str
    seed_ids: list[str]
    case_ids: list[str]
    centroid_terms: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegressionGate(BaseModel):
    id: str
    suite_id: str
    name: str
    min_pass_rate: float = Field(default=0.9, ge=0, le=1)
    min_mutation_survival_rate: float = Field(default=0.8, ge=0, le=1)
    max_needs_review: int = Field(default=0, ge=0)
    blocking_families: list[str] = Field(default_factory=list)
    severity: GateSeverity = "blocking"


class Run(BaseModel):
    id: str
    suite_id: str
    model_names: list[str]
    status: RunStatus = "queued"
    model_outputs: list[ModelOutput] = Field(default_factory=list)
    judgments: list[Judgment] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    gate_results: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
