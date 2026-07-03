from breakpoint_eval import BreakPoint, TraceBuilder


trace = TraceBuilder.tool_failure(
    id="example-tool-stale-cache",
    task="Can customer Delta be upgraded today?",
    bad_answer="Yes. The cache says the account is eligible.",
    expected_behavior="Use the live eligibility API. It says the account is blocked until compliance review is complete.",
    tool_calls=[
        {
            "name": "eligibility_cache",
            "result": {"eligible": True, "checked_at": "2026-01-03"},
            "stale": True,
        },
        {
            "name": "live_eligibility_api",
            "result": {"eligible": False, "reason": "compliance review"},
            "stale": False,
        },
    ],
    summary="Agent trusted stale cache over live tool output.",
)

client = BreakPoint(variants_per_item=2)
build = client.build_pack([trace], output_dir="artifacts/examples/tool_failure")
print(build.model_dump_json(indent=2))
