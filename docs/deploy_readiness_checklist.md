# Deploy Readiness Checklist (Backend Server)

Checklist smoke-test và deploy tối thiểu cho backend FastAPI.

## 1) Environment setup

- [ ] Python version tương thích (khuyến nghị 3.11+).
- [ ] Dependencies cài từ `backend/requirements.txt`.
- [ ] Tạo `backend/.env` từ `backend/.env.example` hoặc `.env.production.example`.

Biến bắt buộc tối thiểu:

- `SERVER_PRIVATE_KEY`
- `ALLOWED_ORIGINS`

Biến cần cho endpoint blockchain record:

- `RPC_URL`
- `CHAIN_ID`
- `CONTRACT_ADDRESS`

## 2) Local startup smoke test

```bash
cd backend
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Trong terminal khác:

```bash
curl http://127.0.0.1:8000/api/health
```

Kỳ vọng:

- HTTP 200
- JSON chứa `status`, `detector`, `zkp_oracle`, `did_service`, `blockchain`

## 3) Core API smoke tests

```bash
# Verify endpoint with image sample
curl -X POST "http://127.0.0.1:8000/api/verify" \
  -F "file=@<path_to_image>" \
  -F "user_address=0xYourWalletAddress"
```

Kiểm tra:

- `status=ok` thì có `label`, `confidence`, `signature`
- `status!=ok` thì response non-classifiable và không có decision fail-open

## 4) Pre-deploy checks

- [ ] CORS `ALLOWED_ORIGINS` đúng domain frontend deploy.
- [ ] Không bật `ALLOW_INSECURE_DEV_KEY=true` trên production.
- [ ] Private key production là key riêng cho môi trường deploy.
- [ ] Log startup không báo lỗi import/module critical.

## 5) Post-deploy quick checks

- [ ] `/api/health` trả 200 trên URL production.
- [ ] `/api/verify` xử lý được 1 ảnh test end-to-end.
- [ ] Nếu dùng blockchain writer: `/api/blockchain/record` hoạt động với env production.
