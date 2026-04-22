# Deployment Guide (Vercel Frontend + Render Backend)

Tài liệu này là checklist triển khai production cho nhánh `test`, theo kiến trúc:

- Frontend: Vite MPA deploy trên Vercel
- Backend: FastAPI deploy trên Render
- Smart contract: deploy riêng (ví dụ Polygon Amoy), frontend/backend chỉ consume địa chỉ + RPC

## 1) Điều kiện trước khi deploy

- Có quyền truy cập repo GitHub.
- Có tài khoản [Vercel](https://vercel.com) và [Render](https://render.com).
- Đã có RPC endpoint + chain id + contract address cho blockchain môi trường deploy.
- Có `SERVER_PRIVATE_KEY` hợp lệ cho backend ký oracle.

## 2) Pre-flight release checklist (bắt buộc)

Chạy từ root repo:

```bash
git checkout test
git pull
```

Smoke test khuyến nghị:

```bash
cd frontend
npm ci
npm run lint
npm run build
cd ..

py -m py_compile backend/api.py backend/blockchain_client.py backend/consumer_api.py backend/zkp_oracle.py
py -m pytest ai_deepfake/tests/test_detect.py
```

Tiêu chí pass:

- `npm run build` thành công.
- Unit test `ai_deepfake/tests/test_detect.py` pass.
- Không còn thay đổi ngoài ý muốn trong `git status`.

## 3) Chuẩn bị biến môi trường

### 3.1 Backend (Render)

Thiết lập các biến sau trên Render service:

- `SERVER_PRIVATE_KEY`: private key oracle (bắt buộc)
- `ALLOWED_ORIGINS`: danh sách domain frontend, cách nhau bằng dấu phẩy
  - Ví dụ: `https://your-main.vercel.app,https://your-consumer.vercel.app`
- `RPC_URL`: endpoint blockchain
- `CHAIN_ID`: chain id (ví dụ `80002` cho Polygon Amoy)
- `CONTRACT_ADDRESS`: địa chỉ contract đã deploy
- `ALLOW_INSECURE_DEV_KEY=false`

Lưu ý:

- Không dùng `ALLOW_INSECURE_DEV_KEY=true` trong production.
- `backend/api.py` đã đọc `ALLOWED_ORIGINS` từ env, không hardcode localhost.

### 3.2 Frontend (Vercel)

Thiết lập env cho project frontend:

- `VITE_API_URL=https://<render-service-domain>`
- `VITE_CHAIN_ID=<chain_id>`
- `VITE_CONTRACT_ADDRESS=<contract_address>`

Nếu cần tách main app và consumer app thành 2 project, đảm bảo cả hai đều trỏ đúng `VITE_API_URL`.

## 4) Deploy Backend lên Render

### 4.1 Tạo service

- New Web Service -> Connect GitHub repo.
- Root directory: repo root.
- Runtime: Docker.
- Branch: `test` (hoặc branch release).

### 4.2 Build/Start

Repo đã có `backend/Dockerfile` với lệnh chạy:

```dockerfile
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Render sẽ map cổng nội bộ và expose public URL.

### 4.3 Health check sau deploy

Kiểm tra:

- `GET https://<render-domain>/api/health`
- Kết quả mong muốn:
  - `status: ok`
  - API phản hồi JSON hợp lệ

Nếu lỗi CORS:

- Kiểm tra lại `ALLOWED_ORIGINS` đã chứa đúng domain Vercel chưa.

## 5) Deploy Frontend lên Vercel

### 5.1 Tạo project

- Add New Project -> import repo.
- Root Directory: `frontend`.
- Framework Preset: Vite.

### 5.2 Build settings

- Install Command: `npm ci`
- Build Command: `npm run build`
- Output Directory: `dist`

### 5.3 Env vars

Khai báo đầy đủ `VITE_API_URL`, `VITE_CHAIN_ID`, `VITE_CONTRACT_ADDRESS` cho:

- Production
- Preview (khuyến nghị cùng backend staging)

### 5.4 Verify route MPA

Sau deploy, kiểm tra:

- `https://<vercel-domain>/`
- `https://<vercel-domain>/consumer/`

Repo đang dùng Vite MPA và config rewrite middleware cho local dev; build output phải có cả `index.html` và `consumer/index.html`.

## 6) Cấu hình CI/CD và release flow khuyến nghị

- Merge code vào `test`.
- Chạy smoke test local.
- Push lên GitHub.
- Render/Vercel tự động deploy từ branch.
- Verify production endpoints + UI flows.
- Gắn tag release sau khi verify xong.

Flow đề xuất:

1. `feature/*` -> PR -> `test`
2. `test` verified -> deploy staging/prod
3. Tag: `vX.Y.Z`

## 7) Runbook rollback nhanh

### Frontend rollback

- Vercel -> Deployments -> Promote deployment trước đó.

### Backend rollback

- Render -> chọn deployment cũ -> rollback (hoặc redeploy commit cũ).

### Sau rollback

- Kiểm tra lại:
  - `/api/health`
  - Luồng verify ảnh
  - Luồng consumer `/consumer/`

## 8) Troubleshooting nhanh

### Vercel build fail vì `__dirname`

- Đã xử lý trong `frontend/vite.config.js` bằng ESM-safe:
  - `fileURLToPath(new URL('.', import.meta.url))`

### Backend không start trên Render

- Kiểm tra biến `SERVER_PRIVATE_KEY`.
- Kiểm tra log import model/detector.
- Kiểm tra cổng chạy: app dùng `0.0.0.0`.

### OOM khi cài AI deps

- `backend/requirements.txt` đã chuyển torch CPU-only và thêm:
  - `--extra-index-url https://download.pytorch.org/whl/cpu`

### CORS fail trên frontend

- Đảm bảo `ALLOWED_ORIGINS` có domain chính xác của Vercel (kể cả https).

## 9) Security checklist trước khi public

- Không commit `.env` thật.
- Rotate key nếu từng lộ private key.
- Giới hạn CORS theo domain cụ thể (không để `*` lâu dài).
- Bật branch protection cho nhánh release.
- Theo dõi log lỗi trong 24h đầu sau release.