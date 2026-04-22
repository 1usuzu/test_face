# Local onboarding (Windows-first) — design spec

**Date:** 2026-04-11  
**Status:** Approved for implementation  
**Scope:** Máy dev cá nhân; ưu tiên bước “API + AI verify” trước, full stack blockchain + MetaMask sau.

## Goal

Sau `git clone`, một luồng rõ ràng để có backend chạy với detector AI (`/api/health` → `detector: true`), tối thiểu ma sát và ít lệnh tay lặp (nhất là `cd`/`venv` sai).

## Prerequisites (điều kiện bắt buộc)

1. **Python** 3.11 hoặc 3.12 (khuyến nghị; khớp `backend/requirements.txt` và wheel PyTorch). Python 3.13+ có thể cần xử lý riêng nếu `pip install -r` thất bại.
2. **Node.js** 18+ (frontend và Hardhat).
3. **Virtual environment** tại root repo (`.venv`); mọi `pip`/`python` cho backend dùng interpreter trong `.venv`.
4. **Dependencies backend:** `pip install -r backend/requirements.txt` trong `.venv`.
5. **Model weights:** `ai_deepfake/models/best_model.pth` và `best_model_v2.pth` phải tồn tại (hoặc quy trình tải/ghi rõ — ngoài phạm vi script nếu không có trong repo).
6. **Biến môi trường backend:** `SERVER_PRIVATE_KEY` bắt buộc để app khởi động; có thể dùng `backend/.env` (được `api.py` load) hoặc export shell. Dev có thể bật `ALLOW_INSECURE_DEV_KEY` + `INSECURE_DEV_PRIVATE_KEY` (chỉ local, không production).

## Optional (full demo on-chain)

- Hardhat node + deploy contract + `frontend/.env` (`VITE_*`) + MetaMask — chỉ sau khi backend AI đã ổn.

## Artifacts

- `scripts/setup-local.ps1` — tạo venv (nếu thiếu), cài deps, kiểm tra `import torch`, gợi ý copy `backend/.env` từ `.env.example`.
- README cập nhật mục điều kiện chạy và tham chiếu script.

## Verification

- `GET http://127.0.0.1:8000/api/health` → `detector: true`, `detector_version: "ensemble"` (sau khi chạy uvicorn với env đúng).

## Out of scope (this spec)

- Docker/production deploy; chỉ đề cập ngắn trong README nếu cần phân biệt local vs server.
