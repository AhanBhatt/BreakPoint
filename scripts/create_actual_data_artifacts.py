from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from breakpoint_eval.actual_data import build_actual_dataset_artifacts
from breakpoint_eval.env import env_flag, load_env
from breakpoint_eval.reports import write_live_judge_report


def main() -> None:
    load_env()
    result = build_actual_dataset_artifacts(
        include_external_judges=env_flag("BREAKPOINT_EXTERNAL_JUDGES", False),
    )
    write_live_judge_report(Path(result.output_dir) / "trace2eval_results.json", ROOT / "artifacts" / "reports")
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
