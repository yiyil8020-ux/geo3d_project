#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_run_pipeline.py — 运行端到端流水线（配置驱动）
==================================================

用法：
    .venv-gempy/bin/python scripts/11_run_pipeline.py configs/synth_base.json
    .venv-gempy/bin/python scripts/11_run_pipeline.py configs/synth_base.json --interactive
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from geo2model.pipeline import run_case  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    interactive = "--interactive" in sys.argv
    if not args:
        print(__doc__)
        raise SystemExit(1)
    cfg_path = Path(args[0])
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    # case_dir 相对路径基于项目根
    if not Path(cfg["case_dir"]).is_absolute():
        cfg["case_dir"] = str(PROJECT_ROOT / cfg["case_dir"])
    cfg["_project_root"] = str(PROJECT_ROOT)
    run_case(cfg, interactive=interactive)


if __name__ == "__main__":
    main()
