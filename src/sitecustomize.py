"""Project-local Python startup customizations.

The Isaac/ROS development environment can expose external pytest plugins built
for a different Python version.  Project tests and smoke scripts should not
auto-load system ROS pytest plugins, because they can import Python 3.12 ROS
packages from a Python 3.11 IsaacLab environment.

Setting PYTEST_DISABLE_PLUGIN_AUTOLOAD is harmless for normal training/eval
processes and only affects pytest when it is launched in the same environment.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
