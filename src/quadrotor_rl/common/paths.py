from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def get_project_root() -> Path:
    """Return repository root.

    File location:
        src/quadrotor_rl/common/paths.py

    parents:
        common -> quadrotor_rl -> src -> project root
    """

    return Path(__file__).resolve().parents[3]


def get_src_root() -> Path:
    return get_project_root() / "src"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    src_root: Path
    logs_root: Path
    outputs_root: Path
    assets_root: Path
    configs_root: Path
    scripts_root: Path
    tests_root: Path

    @classmethod
    def default(cls) -> "ProjectPaths":
        root = get_project_root()
        return cls(
            project_root=root,
            src_root=root / "src",
            logs_root=root / "logs",
            outputs_root=root / "outputs",
            assets_root=root / "assets",
            configs_root=root / "configs",
            scripts_root=root / "scripts",
            tests_root=root / "tests",
        )

    def ensure_standard_dirs(self) -> None:
        for p in [
            self.logs_root,
            self.outputs_root,
            self.assets_root,
            self.configs_root,
            self.scripts_root,
            self.tests_root,
        ]:
            p.mkdir(parents=True, exist_ok=True)
