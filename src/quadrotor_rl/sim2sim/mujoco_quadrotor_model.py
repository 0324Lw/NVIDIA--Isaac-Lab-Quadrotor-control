from __future__ import annotations

from pathlib import Path


def build_minimal_quadrotor_xml(mass: float = 0.0282, arm_length: float = 0.046) -> str:
    arm = float(arm_length)
    mass_value = float(mass)
    return f"""
<mujoco model="quadrotor_policy_check">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="quadrotor" pos="0 0 1">
      <freejoint/>
      <geom name="body" type="box" size="0.03 0.03 0.01" mass="{mass_value}" rgba="0.2 0.2 0.8 1"/>
      <site name="rotor_1" pos="{arm} {arm} 0"/>
      <site name="rotor_2" pos="{-arm} {arm} 0"/>
      <site name="rotor_3" pos="{-arm} {-arm} 0"/>
      <site name="rotor_4" pos="{arm} {-arm} 0"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def write_minimal_quadrotor_xml(path: str | Path, mass: float = 0.0282, arm_length: float = 0.046) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_minimal_quadrotor_xml(mass=mass, arm_length=arm_length), encoding="utf-8")
    return path
