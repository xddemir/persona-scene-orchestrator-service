"""Write SceneSpec's JSON Schema for Unity to validate against.

    python -m scene_orchestrator.models.schema            # -> schema/scene_spec.schema.json
    python -m scene_orchestrator.models.schema out.json   # -> elsewhere

Re-run after changing scene_spec.py. A test fails while the committed file is
out of date, so Unity never validates against a stale contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .scene_spec import SceneSpec

DEFAULT_PATH = Path("schema/scene_spec.schema.json")


def scene_spec_schema_text() -> str:
    return json.dumps(SceneSpec.model_json_schema(), indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scene_spec_schema_text(), encoding="utf-8", newline="\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
