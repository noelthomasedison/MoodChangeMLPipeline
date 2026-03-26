from __future__ import annotations

import argparse
from pprint import pprint

from src.pipeline import refresh_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run artifact-based mood inference for one athlete.")
    parser.add_argument("--athlete-id", type=str, default=None)
    args = parser.parse_args()
    payload = refresh_payload(args.athlete_id)
    pprint({
        "summary": payload.get("summary"),
        "current_moods": payload.get("current_moods"),
        "previous_moods": payload.get("previous_moods"),
        "metrics": payload.get("metrics"),
    })


if __name__ == "__main__":
    main()
