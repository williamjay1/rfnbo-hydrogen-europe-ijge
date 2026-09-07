"""Run exactly one isolated LP solve from a persisted experiment specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esb_upgrade_core import solve_public_cache_case, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = solve_public_cache_case(spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({"scenario_id": result["scenario_id"], "status": result["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
