from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_sim2sim_report(path: str | Path, metrics: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status = metrics.get("status", "UNKNOWN")
    lines = [
        "# Quadrotor Sim2Sim Report",
        "",
        f"Status: **{status}**",
        "",
        "## Metrics",
        "",
    ]
    for key in sorted(metrics.keys()):
        lines.append(f"- {key}: {metrics[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
