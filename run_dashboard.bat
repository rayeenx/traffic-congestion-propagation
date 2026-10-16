@echo off
cd /d %~dp0
if not exist results\dashboard\val_preds.npz .venv\Scripts\python.exe -m src.precompute_val
.venv\Scripts\python.exe -m streamlit run dashboard\app.py
