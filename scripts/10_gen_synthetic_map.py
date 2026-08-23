#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_gen_synthetic_map.py — 生成合成地质图基准案例（含真值）
==========================================================

用法：
    .venv-gempy/bin/python scripts/10_gen_synthetic_map.py            # 全部内置场景
    .venv-gempy/bin/python scripts/10_gen_synthetic_map.py base      # 只生成指定场景

输出：data/output/geo2model/synth_<场景名>/{input,truth}/...
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from geo2model.mapgen import builtin_scenarios, generate_case  # noqa: E402


def main() -> None:
    scns = builtin_scenarios()
    wanted = sys.argv[1:] or list(scns.keys())
    for name in wanted:
        if name not in scns:
            print(f"未知场景: {name}，可选: {list(scns.keys())}")
            continue
        case_dir = PROJECT_ROOT / "data" / "output" / "geo2model" / f"synth_{name}"
        generate_case(scns[name], case_dir)
        if name == "base":
            # 额外复制一份 base 的输入与真值到 synth_base_deep：
            # 与 configs/synth_base(_deep).json 配套做"无/有深部约束"的
            # A/B 消融实验（两个案例目录互不覆盖）
            import shutil

            deep_dir = case_dir.parent / "synth_base_deep"
            for sub in ("input", "truth"):
                dst = deep_dir / sub
                if dst.exists():
                    shutil.rmtree(dst)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(case_dir / sub, dst)
            print(f"[mapgen] 已复制 base 输入/真值 → {deep_dir}（A/B 消融用）")


if __name__ == "__main__":
    main()
