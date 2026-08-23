# -*- coding: utf-8 -*-
"""
geo2model — 基于平面地质图的智能化三维构建 · 原型包
====================================================

流水线四阶段（对应申报书技术路线）：
1. 矢量化   segment / vectorize / terrain      地质图 → 分区标签 + 界线/断层/等高线折线
2. 数据库   geodatabase / constraints          折线 + 单元表 → surface_points / orientations CSV
3. 三维建模 model3d                            CSV → GemPy 隐式模型 → 3D/剖面/导出
4. 应用扩展 apps                               体素解 → 虚拟钻孔 / 任意剖面 / 平切图

支撑模块：
- mapgen    合成地质图生成器（含像素级真值，用于定量验证）
- degrade   扫描退化模拟（鲁棒性测试）
- metrics   定量指标（分区准确率 / IoU / 边界F1 / 三维体素一致率）
- georef    像素 ↔ 世界坐标
- sectionreg 剖面图深部约束配准（像素比例映射）
"""

__version__ = "0.1.0"
