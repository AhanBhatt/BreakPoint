from breakpoint_eval import BreakPoint, TraceBuilder


trace = TraceBuilder.rag_failure(
    id="example-rag-stale-policy",
    question="What is the current refund window for Plan Atlas?",
    bad_answer="The refund window is 7 days according to the cached summary.",
    expected_behavior="Use the official policy effective 2026-06-01 and answer that the refund window is 30 days.",
    retrieved_docs=[
        {
            "id": "cached-summary",
            "source_type": "cached_summary",
            "content": "Plan Atlas refunds are available for 7 days.",
            "effective_date": "2025-09-15",
            "reliability": 0.35,
            "rank": 0,
        },
        {
            "id": "official-policy",
            "source_type": "official_policy",
            "content": "Plan Atlas refunds are available for 30 days.",
            "effective_date": "2026-06-01",
            "reliability": 0.96,
            "rank": 1,
        },
    ],
    summary="Model trusted stale retrieved context over the newer official policy.",
)

client = BreakPoint(variants_per_item=2)
build = client.build_pack([trace], output_dir="artifacts/examples/rag_failure")
print(build.model_dump_json(indent=2))
