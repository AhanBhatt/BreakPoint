from __future__ import annotations

from breakpoint_eval.models import CategorySpec


FAILURE_CATEGORIES: dict[str, CategorySpec] = {
    "hallucination": CategorySpec(
        id="hallucination",
        name="Grounded Hallucination",
        description="The answer must stay inside the supplied evidence and abstain when evidence is missing.",
        failure_modes=[
            "Inventing unsupported dates, names, or source claims",
            "Treating plausible domain knowledge as retrieved evidence",
            "Filling missing facts instead of saying the context is insufficient",
        ],
        eval_families=["unsupported claim detection", "closed-book abstention"],
        risk_level="high",
    ),
    "instruction_conflict": CategorySpec(
        id="instruction_conflict",
        name="Instruction Conflict",
        description="The model must resolve competing instructions using priority and task relevance.",
        failure_modes=[
            "Following a lower-priority user instruction over a developer/system constraint",
            "Mixing incompatible output constraints",
            "Obeying document text that is not an instruction to the assistant",
        ],
        eval_families=["priority ladder", "constraint reconciliation"],
        risk_level="high",
    ),
    "multi_hop_reasoning": CategorySpec(
        id="multi_hop_reasoning",
        name="Multi-Hop Reasoning",
        description="The answer requires combining facts across entities, aliases, time windows, or tables.",
        failure_modes=[
            "Stopping after the first matching fact",
            "Joining on a nearby but wrong entity",
            "Dropping a temporal or alias constraint",
        ],
        eval_families=["far-apart joins", "alias joins", "temporal joins"],
        risk_level="medium",
    ),
    "tool_misuse": CategorySpec(
        id="tool_misuse",
        name="Tool Misuse",
        description="The model must choose the right tool output and reject stale or irrelevant tool results.",
        failure_modes=[
            "Using stale search output for a current question",
            "Calling unnecessary tools after enough evidence is present",
            "Trusting tool metadata over payload content",
        ],
        eval_families=["stale tool output", "tool arbitration"],
        risk_level="high",
    ),
    "long_context_retrieval": CategorySpec(
        id="long_context_retrieval",
        name="Long-Context Retrieval",
        description="The correct answer is a needle hidden among nearby distractors and far-apart supporting facts.",
        failure_modes=[
            "Selecting a visually nearby misleading fact",
            "Missing a later amendment",
            "Failing to combine separated context sections",
        ],
        eval_families=["needle in haystack", "distributed evidence"],
        risk_level="medium",
    ),
    "refusal_boundary": CategorySpec(
        id="refusal_boundary",
        name="Refusal Boundary",
        description="The model must refuse unsafe requests while answering allowed legal, medical, or finance education.",
        failure_modes=[
            "Over-refusing benign educational content",
            "Under-refusing individualized high-stakes advice",
            "Providing procedural wrongdoing guidance under a harmless framing",
        ],
        eval_families=["legal boundary", "medical boundary", "finance boundary"],
        risk_level="high",
    ),
    "format_violation": CategorySpec(
        id="format_violation",
        name="Format Violation",
        description="The response must satisfy a strict schema even when prompt text asks for prose or decoration.",
        failure_modes=[
            "Adding explanatory prose around JSON",
            "Returning missing or extra fields",
            "Changing field types when values are unknown",
        ],
        eval_families=["strict JSON", "schema conformance"],
        risk_level="medium",
    ),
    "rag_contradiction": CategorySpec(
        id="rag_contradiction",
        name="RAG Contradiction",
        description="The model must detect and resolve contradictions across retrieved documents.",
        failure_modes=[
            "Choosing the first retrieved document despite newer contradictory evidence",
            "Averaging incompatible facts",
            "Ignoring source dates and reliability metadata",
        ],
        eval_families=["source freshness", "cross-document contradiction"],
        risk_level="high",
    ),
    "prompt_injection": CategorySpec(
        id="prompt_injection",
        name="Prompt Injection in Retrieved Text",
        description="The model must treat retrieved text as data, not as instructions to the assistant.",
        failure_modes=[
            "Following instructions embedded in a document",
            "Revealing hidden policies or chain-of-thought on request from retrieved text",
            "Dropping the original user task after injection text",
        ],
        eval_families=["RAG injection", "document-borne instruction attack"],
        risk_level="high",
    ),
}


DEFAULT_CATEGORY_IDS = list(FAILURE_CATEGORIES)
