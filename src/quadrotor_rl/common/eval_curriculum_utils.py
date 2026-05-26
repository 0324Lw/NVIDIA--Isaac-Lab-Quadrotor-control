from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCurriculumSpec:
    name: str
    task_id: int
    description: str
    kwargs: dict


def task3_eval_curriculum(name: str) -> EvalCurriculumSpec:
    """Return standard Task3 obstacle-navigation eval curriculum."""

    name = str(name).lower().strip()

    table = {
        "fixed_easy": EvalCurriculumSpec(
            name="fixed_easy",
            task_id=3,
            description="Task3 easy evaluation: few static obstacles, no dynamic obstacles.",
            kwargs={"num_static": 5, "num_dynamic": 0, "max_sg_dist": 25.0},
        ),
        "fixed_medium": EvalCurriculumSpec(
            name="fixed_medium",
            task_id=3,
            description="Task3 medium evaluation: static + dynamic obstacles.",
            kwargs={"num_static": 10, "num_dynamic": 2, "max_sg_dist": 25.0},
        ),
        "fixed_hard": EvalCurriculumSpec(
            name="fixed_hard",
            task_id=3,
            description="Task3 hard evaluation: dense static + dynamic obstacles.",
            kwargs={"num_static": 25, "num_dynamic": 4, "max_sg_dist": 45.0},
        ),
    }

    if name not in table:
        raise KeyError(f"Unknown Task3 eval curriculum: {name}. Available: {sorted(table)}")

    return table[name]


def describe_task(task_id: int) -> str:
    names = {
        1: "hover stabilization",
        2: "waypoint tracking",
        3: "dynamic obstacle navigation",
        4: "vision gate racing",
    }
    return names.get(int(task_id), f"task{task_id}")
