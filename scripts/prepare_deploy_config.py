"""Bake deploy-time values into ``config/costsense.json`` before ``docker build``.

GitHub Actions calls this so secrets land in the image config JSON instead of ECS
task-definition environment variables.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    path = Path("config/costsense.json")
    config = json.loads(path.read_text(encoding="utf-8"))

    token = os.environ.get("COSTSENSE_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        config.setdefault("github", {})["token"] = token

    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
