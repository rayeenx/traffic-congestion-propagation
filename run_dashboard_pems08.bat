@echo off
cd /d %~dp0
set DSO4_DATASET=pems08
if not exist results\pems08\dashboard\val_preds.npz .venv\Scripts\python.exe -m src.precompute_val --dataset pems08
.venv\Scripts\python.exe -m streamlit run dashboard\app.py
