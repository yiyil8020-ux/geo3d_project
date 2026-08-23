#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""消融辅助：对指定案例 db 用指定 kriging range 重算模型并输出体素一致率。"""
import sys, json
sys.path.insert(0, "/Users/yiyi/lzu/校创/geo3d_project")
import numpy as np
import gempy as gp
from geo2model.mapgen import Scenario, truth_voxel_units
from geo2model.metrics import array_agreement

case, rng = sys.argv[1], float(sys.argv[2])
db = f"{case}/db"
cfg = json.load(open(f"{db}/model_config.json"))
tmeta = json.load(open(f"{case}/truth/truth_meta.json"))
scn = Scenario(**tmeta["scenario"])
gt_units, below = truth_voxel_units(scn, cfg["extent"], cfg["resolution"])
id2name_gt = {u["unit_id"]: u["name"] for u in tmeta["units"]}
gt_names = np.vectorize(lambda t: id2name_gt.get(int(t), "?"))(gt_units)

geo_model = gp.create_geomodel(
    project_name="abl", extent=cfg["extent"], resolution=cfg["resolution"],
    importer_helper=gp.data.ImporterHelper(
        path_to_surface_points=f"{db}/surface_points.csv",
        path_to_orientations=f"{db}/orientations.csv",
        coord_x_name="X", coord_y_name="Y", coord_z_name="Z",
        surface_name="formation"))
mapping = {k: (tuple(v) if len(v) > 1 else v[0]) for k, v in cfg["series"].items()}
gp.map_stack_to_surfaces(gempy_model=geo_model, mapping_object=mapping)
if cfg.get("fault_series"):
    gp.set_is_fault(frame=geo_model, fault_groups=cfg["fault_series"])
geo_model.interpolation_options.kernel_options.range = rng
sol = gp.compute_model(geo_model)
nx, ny, nz = cfg["resolution"]
lith = np.round(np.asarray(sol.raw_arrays.lith_block)).astype(int).reshape(nx, ny, nz)
els = [el for g in geo_model.structural_frame.structural_groups for el in g.elements]
id2n = {i + 1: el.name for i, el in enumerate(els)}
id2n[len(els) + 1] = cfg["basement_name"]
pred = np.vectorize(lambda t: id2n.get(int(t), "?"))(lith)
res = array_agreement(pred, gt_names, mask=below)
basement = tmeta["units"][-1]["name"]
nb = below & (gt_names != basement)
print(f"ABL case={case.split('/')[-1]} range={rng}: "
      f"micro={res['agreement']:.4f} "
      f"macro={np.mean(list(res['per_value'].values())):.4f} "
      f"excl_basement={(pred[nb]==gt_names[nb]).mean():.4f}")
