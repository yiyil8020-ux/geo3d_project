# -*- coding: utf-8 -*-
"""
geodatabase.py — 空间数据库构建与人机交互校正
==============================================

对应申报书「数据库构建」阶段：把矢量化结果整理成统一的结构化数据，
并提供人机交互校正接口（本原型以"可编辑的 units.csv/审核配置"实现，
算法置信度不足的字段——层序 order、单元名称、断层确认、产状读数——
交由人工填写/修改后再进入建模环节）。

人机交互的三种使用方式：
1. 配置驱动（批量实验）：case 配置 JSON 里给出 review 字典；
2. 文件驱动（日常使用）：直接编辑 db/units.csv 与 extract/faults.json；
3. 真值模拟（定量评估）：review="from_truth" 时按合成真值自动填表，
   等价于"一个不会填错的用户"，用于无人值守的批量评测。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

UNIT_COLUMNS = [
    "unit_id", "name", "role", "order", "series",
    "color_r", "color_g", "color_b", "dip", "azimuth",
]


def _color_match_to_truth(units_auto: pd.DataFrame, truth_units: list[dict]) -> dict:
    """把提取单元按颜色与真值单元一一匹配（匈牙利算法，ΔE 最小）。

    模拟"用户对照图例给每个色块命名"的人工步骤。
    """
    from scipy.optimize import linear_sum_assignment
    from skimage.color import rgb2lab

    ext = units_auto[["color_r", "color_g", "color_b"]].to_numpy(dtype=float)
    tru = np.array(
        [[u["color_r"], u["color_g"], u["color_b"]] for u in truth_units], dtype=float
    )
    lab_e = rgb2lab(ext.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    lab_t = rgb2lab(tru.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    cost = np.linalg.norm(lab_e[:, None, :] - lab_t[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    # ΔE 太大的匹配不可信（比如河流簇没有真值对应）
    return {
        int(units_auto.iloc[i]["unit_id"]): truth_units[j]
        for i, j in zip(ri, ci)
        if cost[i, j] < 25.0
    }


def apply_review(
    extract_dir: str | Path,
    db_dir: str | Path,
    review: dict | str | None,
    truth_meta_path: str | Path | None = None,
) -> pd.DataFrame:
    """把人工审核信息合入自动单元表，产出 db/units.csv。

    review 取值：
    - "from_truth"：按真值模拟人工填表（需 truth_meta_path）
    - dict：{"units": [覆盖行...], "fault_confirm": [fault_id...] 或 "all",
             "fault_edits": {fault_id: {dip/azimuth/name}}}
    - None：不改动（units_auto 原样升级为 units.csv，通常仅调试用）
    """
    extract_dir, db_dir = Path(extract_dir), Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(extract_dir / "units_auto.csv")

    # 确保所有契约列都在（pandas 3.0 起 dtype 严格：字符串列必须建成
    # object 类型，否则后续填入字符串会 TypeError）
    _str_cols = {"name", "role", "series"}
    for col in UNIT_COLUMNS:
        if col not in units.columns:
            if col in _str_cols:
                units[col] = pd.Series([None] * len(units), dtype=object)
            else:
                units[col] = np.nan

    if review == "from_truth":
        if truth_meta_path is None:
            raise ValueError("review='from_truth' 需要提供 truth_meta_path")
        with open(truth_meta_path, encoding="utf-8") as f:
            truth_units = json.load(f)["units"]
        mapping = _color_match_to_truth(units, truth_units)
        for idx, row in units.iterrows():
            uid = int(row["unit_id"])
            if uid in mapping:
                t = mapping[uid]
                units.loc[idx, "name"] = t["name"]
                units.loc[idx, "order"] = t["order"]
                units.loc[idx, "role"] = "strata"
            else:
                # 没有真值对应的簇（河流残留/文字簇等）→ 忽略
                units.loc[idx, "name"] = f"misc_{uid}"
                units.loc[idx, "order"] = -1
                units.loc[idx, "role"] = "ignore"
        # 图上未出露但图例/剖面中存在的单元（典型：从不出露的基底）：
        # 真实流程中用户按图例补录 → from_truth 模式等价地自动补入。
        matched_names = {t["name"] for t in mapping.values()}
        next_fake_id = -100
        for t in truth_units:
            if t["name"] not in matched_names:
                units = pd.concat([units, pd.DataFrame([{
                    "unit_id": next_fake_id, "name": t["name"],
                    "role": "strata", "order": t["order"],
                    "color_r": t["color_r"], "color_g": t["color_g"],
                    "color_b": t["color_b"],
                    "dip": np.nan, "azimuth": np.nan,
                }])], ignore_index=True)
                print(f"[geodatabase] 按图例补录未出露单元: {t['name']} "
                      f"(order={t['order']})")
                next_fake_id -= 1
    elif isinstance(review, dict):
        for edit in review.get("units", []):
            # 定位方式一：unit_id 直接指定；方式二：match_color 按 ΔE 最近的
            # 簇代表色匹配（对重跑更稳，推荐真实地图使用）
            if "unit_id" in edit:
                sel = units["unit_id"] == int(edit["unit_id"])
            elif "match_color" in edit:
                from skimage.color import rgb2lab

                tgt = np.asarray(edit["match_color"], dtype=float)
                cols = units[["color_r", "color_g", "color_b"]].to_numpy(float)
                lab_c = rgb2lab(cols.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
                lab_t = rgb2lab(tgt.reshape(1, 1, 3) / 255.0).reshape(3)
                de = np.linalg.norm(lab_c - lab_t, axis=1)
                if de.min() > 30.0:
                    print(f"[geodatabase] 警告: match_color={edit['match_color']} "
                          f"最近 ΔE={de.min():.1f} 过大，跳过该条")
                    continue
                sel = units.index == units.index[int(np.argmin(de))]
            else:
                print(f"[geodatabase] 警告: review 条目缺少 unit_id/match_color: {edit}")
                continue
            if not np.asarray(sel).any():
                print(f"[geodatabase] 警告: review 条目未匹配到单元: {edit}")
                continue
            for k, v in edit.items():
                if k not in ("unit_id", "match_color"):
                    units.loc[sel, k] = v
        # 附加单元（图上未出露但图例中存在，如隐伏基底）
        next_fake_id = -100
        for extra in review.get("extra_units", []):
            row = {"unit_id": next_fake_id, "role": "strata",
                   "color_r": 128, "color_g": 128, "color_b": 128,
                   "dip": np.nan, "azimuth": np.nan, "series": "Strat_Series"}
            row.update(extra)
            units = pd.concat([units, pd.DataFrame([row])], ignore_index=True)
            next_fake_id -= 1
    elif review is None:
        pass
    else:
        raise ValueError(f"不支持的 review 类型: {review!r}")

    # 规范化
    units["order"] = units["order"].fillna(-1).astype(int)
    units["role"] = units["role"].fillna("strata")
    if "series" not in units.columns:
        units["series"] = np.nan
    units["series"] = units["series"].fillna("Strat_Series")
    units = units[UNIT_COLUMNS]
    _validate_units(units)
    units.to_csv(db_dir / "units.csv", index=False)

    # ---- 断层确认/修订 ----
    faults_path = extract_dir / "faults.json"
    if faults_path.is_file():
        with open(faults_path, encoding="utf-8") as f:
            faults = json.load(f)
        if isinstance(review, dict):
            confirm = review.get("fault_confirm", [])
            edits = review.get("fault_edits", {})
            for fl in faults:
                fid = fl["fault_id"]
                if confirm == "all" or fid in confirm:
                    fl["confirmed"] = True
                for k, v in edits.get(str(fid), {}).items():
                    fl[k] = v
        elif review == "from_truth":
            # 真值模拟：有断层参数就确认最长候选并用真值倾角/倾向
            if truth_meta_path is not None:
                with open(truth_meta_path, encoding="utf-8") as f:
                    fp = json.load(f).get("fault_params")
                if fp and faults:
                    faults[0]["confirmed"] = True
                    faults[0]["dip"] = fp["dip"]
                    faults[0]["azimuth"] = fp["azimuth"]
                    faults[0]["name"] = fp.get("name", faults[0]["name"])
        with open(db_dir / "faults.json", "w", encoding="utf-8") as f:
            json.dump(faults, f, ensure_ascii=False, indent=1)

    n_strata = int((units["role"] == "strata").sum())
    print(f"[geodatabase] units.csv 已生成: 地层单元 {n_strata} 个 "
          f"(共 {len(units)} 个分区)")
    return units


def _validate_units(units: pd.DataFrame) -> None:
    """数据质量校验（问题只警告不中断——人机交互原则）。

    - 同名多簇是**合法输入**（一个地层被分成多个色簇，同名即合并），
      只提示不告警；层序连续性按"去重后的 order"检查，避免误报。
    - role=strata 但 order<0 说明层序漏审，建模阶段会剔除，这里先提醒。
    """
    strata = units[units["role"] == "strata"]
    dup = strata[strata["name"].duplicated()]["name"].unique().tolist()
    if dup:
        print(f"[geodatabase] 提示: 同名多簇将合并处理: {dup}")
    no_order = strata[strata["order"] < 0]["name"].tolist()
    if no_order:
        print(f"[geodatabase] 警告: 下列地层单元未填层序(order<0)，"
              f"建模时将被剔除: {no_order}")
    orders = sorted(set(int(t) for t in strata["order"] if t >= 0))
    if orders and orders != list(range(min(orders), min(orders) + len(orders))):
        print(f"[geodatabase] 警告: 层序编号不连续: {orders}")
    if len(strata) < 2:
        print("[geodatabase] 警告: 地层单元少于 2 个，无法建模")


def load_units(db_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(db_dir) / "units.csv")


def load_confirmed_faults(db_dir: str | Path) -> list[dict]:
    p = Path(db_dir) / "faults.json"
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as f:
        return [fl for fl in json.load(f) if fl.get("confirmed")]
