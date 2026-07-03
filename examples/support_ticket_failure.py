from breakpoint_eval import BreakPoint, TraceBuilder


trace = TraceBuilder.support_ticket_failure(
    id="example-support-overpromise",
    subject="Enterprise customer asks for a data deletion SLA",
    customer_message="Can you confirm our data will be deleted within 24 hours?",
    bad_response="Yes, all data is always deleted within 24 hours.",
    correct_resolution="Do not overpromise. State that deletion timing depends on the contract and cite the customer's data-processing addendum.",
    summary="Support assistant promised an SLA that was not grounded in the account contract.",
    severity="high",
)

client = BreakPoint(variants_per_item=2)
build = client.build_pack([trace], output_dir="artifacts/examples/support_ticket_failure")
print(build.model_dump_json(indent=2))
