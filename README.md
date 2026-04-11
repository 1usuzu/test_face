# DeepfakeVerify - Hệ thống định danh phi tập trung với AI phát hiện deepfake

> Đồ án tốt nghiệp: hệ thống định danh phi tập trung tích hợp AI phát hiện deepfake và xác thực không kiến thức.

## Trạng thái hiện tại (sau Stabilize + Harden)

### Baseline và claim

- **Frozen baseline (source of truth): 83.33% accuracy** theo báo cáo đo thực tế đã chốt.
- Con số **93.33%** trong tài liệu cũ chỉ là **legacy/historical README claim**, không được dùng làm frozen baseline.
- Khi báo cáo kết quả mới, luôn dùng output đo từ `ai_deepfake/evaluate.py`, không overclaim.

### Kiến trúc detector hiện tại (high-level)

- Inference dùng ensemble 2 model ảnh:
  - EfficientNet-B0 (`best_model.pth`)
  - EfficientNet-B4 head tùy biến (`best_model_v2.pth`)
- Kết quả chính là `fake_probability` (xác suất ảnh giả), sau đó phân lớp theo threshold.
- `confidence` là độ chắc chắn theo lớp dự đoán:
  - nếu dự đoán FAKE: `confidence = fake_probability`
  - nếu dự đoán REAL: `confidence = 1 - fake_probability`
- Có face detection (MTCNN) nếu môi trường có `facenet-pytorch`.

### Threshold policy (C1)

- Default operating threshold: **`0.65`** (balanced mode, runtime default).
- High-recall fake-detection mode: `0.40`.
- High-precision fake-detection mode: `0.75`.
- Source of truth: `docs/c1_operating_points.json`.

### Hành vi hardening theo status

- Detector trả `status` để phân biệt kết quả phân loại hợp lệ và lỗi không-classifiable:
  - `ok`
  - `no_face`
  - `face_detection_error`
  - `no_model`
  - `error`
- **Quy tắc an toàn bắt buộc ở API layer**: nếu `status != ok` thì không coi là REAL/APPROVED.

## Công nghệ chính

| Layer | Công nghệ |
| --- | --- |
| AI Model | PyTorch, EfficientNet-B0 + EfficientNet-B4 ensemble |
| Backend | FastAPI, Uvicorn, Python |
| Frontend | React, Vite, ethers.js |
| Blockchain | Solidity, Hardhat |
| DID | W3C DID Core, Verifiable Credentials |
| ZKP | Circom, SnarkJS, Groth16 |

## Cấu trúc dự án (thực tế hiện có)

```text
ai_deepfake/
  detect.py
  ai_config.py
  evaluate.py
  models/
backend/
  api.py
  consumer_api.py
  zkp_oracle.py
blockchain/
did_system/
frontend/
zkp/
```

## Cài đặt

### Yêu cầu hệ thống

- Python 3.11 or 3.12 (recommended; validated for current backend requirements)
- Node.js 18+
- CUDA (khuyến nghị, để chạy AI trên GPU)
- MetaMask extension

### Điều kiện tối thiểu để chạy được (local)

Để backend AI và API hoạt động (kể cả `/api/verify-zkp`), cần đồng thời:

| Điều kiện | Giải thích |
| --- | --- |
| **Đúng Python + venv** | Cài dependency bằng `pip` **trong** `.venv` của repo (tránh lỗi `No module named 'torch'` do cài nhầm sang Python global). |
| **`backend/requirements.txt`** | Bao gồm PyTorch và stack FastAPI; chạy `pip install -r backend/requirements.txt` sau khi kích hoạt venv. |
| **Model** | File `ai_deepfake/models/best_model.pth` và `best_model_v2.pth` phải có mặt. |
| **`SERVER_PRIVATE_KEY`** | Bắt buộc khi khởi động API. Có thể đặt trong `backend/.env` (xem `backend/.env.example`) hoặc biến môi trường. Chỉ **local dev**: có thể bật `ALLOW_INSECURE_DEV_KEY` + `INSECURE_DEV_PRIVATE_KEY` như trong hướng dẫn dưới — **không** dùng kiểu này trên server thật. |
| **Restart backend** | Sau khi cài thêm package (ví dụ `torch`), cần restart process `uvicorn` để `/api/health` phản ánh `detector: true`. |

**Python 3.13+:** wheel PyTorch trong `requirements.txt` có thể chưa khớp; nếu `pip install -r` thất bại, nên dùng Python **3.11 hoặc 3.12** hoặc cài PyTorch theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/) rồi cài các gói còn lại từ `requirements.txt`.

**Cài nhanh trên Windows (một script):** từ thư mục gốc repo chạy `powershell -ExecutionPolicy Bypass -File .\scripts\setup-local.ps1` (tùy chọn `-SkipNpm` nếu đã `npm install`). Thiết kế chi tiết: `docs/superpowers/specs/2026-04-11-local-onboarding-design.md`.

### 1. Cài đặt môi trường chung (Chạy 1 lần duy nhất)

```bash
# Clone project
git clone <repository-url>
cd <repo-directory>

# Tạo Python virtual environment (Dành cho Backend)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Cài đặt Python dependencies cho backend
pip install -r backend/requirements.txt

# Cài đặt Blockchain dependencies
cd blockchain
npm install
cd ..

# Cài đặt Frontend dependencies (Bao gồm Main App + Consumer App)
cd frontend
npm install buffer react-router-dom vite-plugin-node-polyfills
npm install
cd ..
```

---

## Chạy ứng dụng (Local Development)

Quy trình sử dụng **3 Terminal** chạy song song:

### 🪟 Terminal 1: Mạng Blockchain Cục Bộ (Hardhat Node)

```bash
cd blockchain
npx hardhat node
```

Hardhat sẽ hiển thị danh sách accounts. **Lưu lại Private Key của Account #0** (sử dụng trên Backend để làm Oracle Node ký giao dịch).

*(Chỉ làm 1 lần lúc mới bật)* Mở thêm 1 terminal phụ để Deploy Contract:
```bash
cd blockchain
npx hardhat run scripts/deploy.js --network localhost
```
Copy địa chỉ contract được mổ ra (ví dụ: `0x5FbDB2315678afecb367f032d93F642f64180aa3`). Mình sẽ cần để gán vào `.env` của Frontend lúc sau.

### 🪟 Terminal 2: Khởi chạy Backend API (Core API + Oracle)

```bash
.venv\Scripts\activate  # Nhớ active venv

# Thiết lập biến môi trường (dùng Private Key của Account #0 lấy từ output Hardhat)
$env:SERVER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
$env:ALLOW_INSECURE_DEV_KEY = "true"
$env:INSECURE_DEV_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

cd backend
python -m uvicorn api:app --reload --port 8000
```

### 🪟 Terminal 3: Khởi chạy Giao diện Frontend (Main + Consumer App)

Tạo file `frontend/.env`:
```env
VITE_CONTRACT_ADDRESS=<Địa_chỉ_contract_vừa_deploy_lúc_nãy>
VITE_API_URL=http://localhost:8000
VITE_CHAIN_ID=31337
```

Bật server giao diện MPA:
```bash
cd frontend
npm run dev
```

### Cấu hình MetaMask

1. **Thêm Network mới:**
   - Network Name: `Hardhat Local`
   - RPC URL: `http://127.0.0.1:8545`
   - Chain ID: `31337`
   - Currency Symbol: `ETH`

2. **Import Account #0:**
   - Vào MetaMask > Import Account
   - Paste Private Key: dùng key Account #0 từ output Hardhat.

### Trải Nghiệm Sử dụng Toàn Diện

1. **Main App (Identity Provider)**: Truy cập `http://localhost:5173/` để kết nối Ví Metamask, verify khuôn mặt, tải \`proof.json & public.json\` về máy.
2. **Consumer App (DApp Bầu Cử)**: Truy cập `http://localhost:5173/consumer/` để đăng nhập với tư cách cử tri vô danh. Gửi bằng chứng ZKP mà không cần lộ mặt.

---

## 🚀 Deployment (Triển khai Production)

Dự án V1 đã sẵn sàng để đưa lên Internet. Chúng tôi khuyến nghị sử dụng các nền tảng sau:

- **Backend AI**: Triển khai lên [Render.com](https://render.com) (Server Python/FastAPI).
- **Frontend App**: Triển khai lên [Vercel.com](https://vercel.com) (MPA Vite/React).
- **Blockchain**: Triển khai Smart Contract lên mạng **Polygon Amoy Testnet**.

> [!IMPORTANT]
> Trước khi deploy production, chuẩn bị đủ biến môi trường backend/frontend theo các file mẫu `.env.example`.
>
> Checklist chuẩn bị release/deploy:
>
> - `docs/release_readiness_checklist.md`
> - `docs/deploy_readiness_checklist.md`

### Checklist chuẩn bị đẩy GitHub

- [ ] Đảm bảo không commit file `.env` chứa key thật.
- [ ] Đã cấu hình biến môi trường trên Render/Vercel.
- [ ] Smart contract đã được deploy lên Testnet và cập nhật địa chỉ vào cấu hình.

---

## API endpoints

| Method | Endpoint | Mô tả |
| --- | --- | --- |
| POST | `/api/verify` | Xác thực ảnh deepfake + ký Oracle (chỉ khi `status=ok`) |
| POST | `/api/verify-zkp` | Xác thực + tạo dữ liệu ZKP (chỉ khi `status=ok`) |
| GET | `/api/health` | Health check backend |
| POST | `/api/blockchain/record` | Ghi kết quả lên blockchain (server ký tx) |
| GET | `/api/zkp-info` | Thông tin ZKP system |
| POST | `/api/did/create` | Tạo DID mới |
| GET | `/api/did/resolve/{did}` | Resolve DID document |
| GET | `/api/did/resolve-by-address/{address}` | Tìm DID theo ETH address |
| POST | `/api/credential/issue` | Xác thực + cấp VC (reject nếu detector `status != ok`) |
| POST | `/api/credential/verify` | Xác thực VC |
| GET | `/api/did/info` | Thống kê DID system |

### Blockchain end-to-end (server ký — không cần MetaMask)

Backend có thể gửi transaction lên smart contract bằng `SERVER_PRIVATE_KEY` (custodial tx) để demo end-to-end.

Yêu cầu cấu hình trong `backend/.env`:

```env
RPC_URL=http://127.0.0.1:8545
CHAIN_ID=31337
CONTRACT_ADDRESS=0x...
SERVER_PRIVATE_KEY=0x...
```

Consumer portal (`consumer_app`) có checkbox “Ghi lên Blockchain” và sẽ tự lưu `tx_hash` vào audit.

### Request

```bash
curl -X POST "http://localhost:8000/api/verify" \
  -F "file=@image.jpg" \
  -F "user_address=0xYourWalletAddress"
```

### Response mẫu `/api/verify` (status=ok)

```json
{
  "status": "ok",
  "label": "REAL",
  "confidence": "<measured_value>",
  "real_prob": "<measured_value>",
  "fake_prob": "<measured_value>",
  "image_hash": "a1b2c3...",
  "signature": "abc123...",
  "debug_msg": "0xAddress:imageHash:true",
  "risk_level": "low|medium|high|critical"
}
```

### Response mẫu non-classifiable (`status != ok`)

```json
{
  "status": "no_face",
  "label": "ERROR",
  "message": "<human_readable_message>",
  "error_code": "NO_FACE_DETECTED",
  "face_detected": false,
  "confidence": 0.0
}
```

## Đánh giá model (evaluation harness)

Script đo hiện tại: `ai_deepfake/evaluate.py`

### Cấu trúc dữ liệu đầu vào

```text
<data-dir>/
  real/
    *.jpg|*.jpeg|*.png|*.bmp|*.webp
  fake/
    *.jpg|*.jpeg|*.png|*.bmp|*.webp
```

### Cách chạy

```bash
python ai_deepfake/evaluate.py \
  --data-dir <path_to_dataset_root> \
  --output <path_to_report.json> \
  --detector-version <tag_for_comparison>
```

Tuỳ chọn:

- `--threshold <float>`: override threshold khi gọi detector.

### Hành vi script

- Báo lỗi rõ ràng nếu thiếu dữ liệu (`real/`, `fake/`) hoặc không có ảnh.
- Báo lỗi rõ ràng nếu thiếu model weights (`best_model.pth` và `best_model_v2.pth`).
- Ghi report JSON dùng cho so sánh V1 vs V2, bao gồm:
  - metadata (version tag, threshold, số ảnh, model presence)
  - metrics đo được
  - confusion matrix
  - thống kê status non-ok
  - kết quả từng ảnh (`per_image`)

### JSON report mẫu (schema-oriented, không phải số đo thật)

```json
{
  "meta": {
    "detector_version_tag": "<version_tag>",
    "total_images": "<measured_value>",
    "counted_images_for_metrics": "<measured_value>",
    "excluded_non_ok_images": "<measured_value>"
  },
  "status_summary": {
    "non_ok_breakdown": {
      "<status_name>": "<measured_value>"
    }
  },
  "metrics": {
    "accuracy": "<measured_value>",
    "precision_fake": "<measured_value>",
    "recall_fake": "<measured_value>",
    "f1_fake": "<measured_value>",
    "precision_real": "<measured_value>",
    "recall_real": "<measured_value>",
    "f1_real": "<measured_value>",
    "f1_macro": "<measured_value>"
  },
  "confusion_matrix": {
    "tp_fake_as_fake": "<measured_value>",
    "tn_real_as_real": "<measured_value>",
    "fp_real_as_fake": "<measured_value>",
    "fn_fake_as_real": "<measured_value>"
  }
}
```

## Smart Contract

Contract `DeepfakeVerification.sol`:

- **registerDID**: Đăng ký định danh phi tập trung
- **recordVerification**: Lưu kết quả xác thực (yêu cầu chữ ký Oracle)
- **getVerification**: Truy vấn kết quả theo image hash
- **getStats**: Thống kê tổng số DID và xác thực

### Bảo mật

Hệ thống sử dụng **Oracle Signature** để đảm bảo:

- Chỉ kết quả từ AI Server mới được ghi lên blockchain
- Người dùng không thể giả mạo kết quả xác thực
- Server ký message bằng private key, contract verify bằng `ecrecover`

## Tác giả

Đồ án tốt nghiệp - 2025

## License

MIT License
