#!/bin/zsh
# geo2model 图形界面启动器 —— 双击即可（macOS）
# 会在默认浏览器打开 http://127.0.0.1:7860
cd "$(dirname "$0")"
echo "正在启动 geo2model 界面（首次加载约 10 秒）…"
exec .venv-gempy/bin/python app/geo2model_app.py
