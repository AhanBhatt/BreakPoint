from __future__ import annotations

import random
import re
from collections.abc import Callable

from breakpoint_eval.models import AdversarialVariant, EvalItem


MutationFn = Callable[[EvalItem, random.Random], tuple[str, str, str, str]]


def add_irrelevant_context(item: EvalItem, rng: random.Random) -> tuple[str, str, str, str]:
    distractor_id = rng.randint(100, 999)
    prompt = (
        f"{item.task}\n\nContext:\n"
        f"Archive note {distractor_id}: an unrelated rollout used the same project nickname, "
        "but it was owned by a different team and has no bearing on this question.\n\n"
        f"{item.context}"
    )
    return "irrelevant_context", prompt, item.expected_answer.value, "Extra context should not change the answer."


def reorder_facts(item: EvalItem, rng: random.Random) -> tuple[str, str, str, str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", item.context) if part.strip()]
    rng.shuffle(parts)
    prompt = f"{item.task}\n\nContext:\n" + "\n\n".join(parts)
    return "reordered_facts", prompt, item.expected_answer.value, "Answer should survive paragraph order changes."


def paraphrase_instruction(item: EvalItem, rng: random.Random) -> tuple[str, str, str, str]:
    openings = [
        "Answer the following using only the provided evidence.",
        "Resolve the task from the context alone.",
        "Give the final answer and do not rely on outside knowledge.",
    ]
    task = f"{rng.choice(openings)} {item.task}"
    prompt = f"{task}\n\nContext:\n{item.context}"
    return "paraphrased_instruction", prompt, item.expected_answer.value, "Surface wording should not change grading."


def rename_entities(item: EvalItem, rng: random.Random) -> tuple[str, str, str, str]:
    replacements = [
        ("Aster", "Nimble"),
        ("Beacon", "Harbor"),
        ("Cobalt", "Juniper"),
        ("Delta", "Orchid"),
        ("Eli Rao", "Mira Chen"),
        ("Mira Chen", "Jon Bell"),
    ]
    source, target = rng.choice(replacements)
    prompt = f"{item.task}\n\nContext:\n{item.context}".replace(source, target)
    expected = item.expected_answer.value.replace(source, target)
    return "renamed_entities", prompt, expected, "Entity renaming should preserve relational structure."


def inject_conflicting_evidence(item: EvalItem, rng: random.Random) -> tuple[str, str, str, str]:
    prompt = (
        f"{item.task}\n\nContext:\n"
        f"{item.context}\n\n"
        "Unverified chat excerpt: a teammate guessed a different answer, but the excerpt has no source, "
        "no timestamp, and is explicitly marked as speculation."
    )
    return "conflicting_evidence", prompt, item.expected_answer.value, "Speculative evidence should not override grounded facts."


MUTATORS: list[MutationFn] = [
    add_irrelevant_context,
    reorder_facts,
    paraphrase_instruction,
    rename_entities,
    inject_conflicting_evidence,
]


def build_variants(item: EvalItem, count: int, rng: random.Random) -> list[AdversarialVariant]:
    variants: list[AdversarialVariant] = []
    for index, mutator in enumerate(rng.sample(MUTATORS, k=min(count, len(MUTATORS))), start=1):
        mutation, prompt, expected, trap = mutator(item, rng)
        variants.append(
            AdversarialVariant(
                id=f"{item.id}-v{index:02d}",
                mutation=mutation,
                prompt=prompt,
                expected_answer=expected,
                trap=trap,
                metadata={"parent_id": item.id},
            )
        )
    return variants
