# DeepfakeVerify - Hệ thống Định danh Phi tập trung với AI Phát hiện Deepfake

> Đồ án tốt nghiệp: Hệ thống định danh phi tập trung tích hợp AI phát hiện Deepfake và xác thực không kiến thức

## Giới thiệu

Hệ thống kết hợp 3 công nghệ chính:

- **AI Deepfake Detection**: Phát hiện ảnh giả mạo sử dụng EfficientNet-B0
- **Decentralized Identity (DID)**: Định danh phi tập trung theo chuẩn W3C
- **Blockchain**: Lưu trữ kết quả xác thực bất biến trên smart contract

## Tính năng

- Phát hiện ảnh Deepfake với độ chính xác 93.33%
- Tạo và quản lý DID (Decentralized Identifier)
- Cấp Verifiable Credentials cho kết quả xác thực
- Lưu trữ kết quả lên blockchain với chữ ký Oracle
- Giao diện web thân thiện với drag & drop

## Công nghệ sử dụng

| Layer      | Công nghệ                                     |
| ---------- | --------------------------------------------- |
| AI Model   | PyTorch, EfficientNet-B0, Temperature Scaling |
| Backend    | FastAPI, Uvicorn, Python 3.11                 |
| Frontend   | React 19, Vite, ethers.js v6                  |
| Blockchain | Solidity 0.8.20, Hardhat                      |
| DID        | W3C DID Core 1.0, Ed25519, Verifiable Credentials |
| ZKP        | Circom 2.1.0, SnarkJS, Groth16, Poseidon Hash |

## Cấu trúc dự án

```
├── ai_deepfake/           # AI Deepfake Detection
│   ├── detect.py          # Inference module (EfficientNet-B0)
│   ├── train.py           # Training script
│   ├── test_model.py      # Evaluation
│   └── models/            # Trained models (.pth)
├── backend/               # FastAPI Backend
│   ├── api.py             # REST API + Oracle signing + DID integration
│   └── zkp_oracle.py      # ZKP Oracle module
├── blockchain/            # Smart Contracts
│   ├── contracts/         # Solidity contracts
│   └── scripts/           # Deploy scripts
├── did_system/            # ★ DID & Verifiable Credentials (W3C Standard)
│   ├── __init__.py        # Module exports
│   ├── key_manager.py     # Ed25519/secp256k1 key management
│   ├── did_manager.py     # DID CRUD operations
│   ├── credential_issuer.py   # Verifiable Credentials issuance
│   ├── credential_verifier.py # VC verification
│   ├── did_service.py     # Integration service
│   └── test_did_system.py # 22 comprehensive tests
├── zkp/                   # ★ Zero-Knowledge Proofs (Circom + SnarkJS)
│   ├── circuits/          # Circom circuits
│   ├── contracts/         # ZK Verifier contracts
│   └── scripts/           # Test scripts
└── frontend/              # React Frontend
    └── src/               # React components
```

## Cài đặt

### Yêu cầu hệ thống

- Python 3.11+
- Node.js 18+
- CUDA (khuyến nghị, để chạy AI trên GPU)
- MetaMask extension

### 1. Clone và cài đặt dependencies

```bash
# Clone project
git clone <repository-url>
cd face

# Tạo Python virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Cài đặt Python dependencies
pip install torch torchvision fastapi uvicorn python-multipart pillow cryptography base58 web3 eth-account

# Cài đặt Frontend dependencies
cd frontend
npm install
cd ..

# Cài đặt Blockchain dependencies
cd blockchain
npm install
cd ..
```

## Chạy ứng dụng (Local Development)

Mở **3 terminal riêng biệt** và chạy theo thứ tự:

### Terminal 1: Blockchain (Hardhat Node)

```bash
cd blockchain
npx hardhat node
```

Hardhat sẽ hiển thị danh sách accounts. **Lưu lại Account #0**:

- Address: `0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266`
- Private Key: `0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80`

### Terminal 2: Deploy Contract

```bash
cd blockchain
npx hardhat run scripts/deploy.js --network localhost
```

Copy địa chỉ contract được hiển thị (ví dụ: `0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512`)

### Terminal 3: Backend API

```bash
# Thiết lập biến môi trường (dùng Private Key của Account #0)
$env:SERVER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

cd backend
python -m uvicorn api:app --reload --port 8000
```

### Terminal 4: Frontend

```bash
cd frontend
npm run dev
```

### Cấu hình Frontend (.env)

Tạo file `frontend/.env`:

```env
VITE_CONTRACT_ADDRESS=0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512
VITE_API_URL=http://localhost:8000
VITE_CHAIN_ID=31337
```

**Thay `VITE_CONTRACT_ADDRESS` bằng địa chỉ contract bạn vừa deploy.**

### Cấu hình MetaMask

1. **Thêm Network mới:**

   - Network Name: `Hardhat Local`
   - RPC URL: `http://127.0.0.1:8545`
   - Chain ID: `31337`
   - Currency Symbol: `ETH`

2. **Import Account #0:**
   - Vào MetaMask > Import Account
   - Paste Private Key: `0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80`

### Sử dụng ứng dụng

1. Truy cập `http://localhost:5173`
2. Click "Connect Wallet" và chọn account vừa import
3. Click "Register DID" để đăng ký định danh
4. Upload ảnh để xác thực Deepfake
5. Kết quả sẽ được AI phân tích và lưu lên blockchain

## Deploy lên Testnet (Polygon Amoy)

### Cấu hình

Sửa file `blockchain/hardhat.config.js`:

```javascript
networks: {
  amoy: {
    url: "https://rpc-amoy.polygon.technology",
    accounts: ["YOUR_PRIVATE_KEY"],
    chainId: 80002
  }
}
```

### Deploy

```bash
cd blockchain
npx hardhat run scripts/deploy.js --network amoy
```

## API Endpoints

| Method | Endpoint      | Mô tả                             |
| ------ | ------------- | --------------------------------- |
| POST   | `/api/verify` | Xác thực ảnh deepfake + ký Oracle |
| POST   | `/api/verify-zkp` | Xác thực + tạo ZKP input |
| GET    | `/api/zkp-info` | Thông tin ZKP system |
| POST   | `/api/did/create` | Tạo DID mới |
| GET    | `/api/did/resolve/{did}` | Resolve DID Document |
| GET    | `/api/did/resolve-by-address/{address}` | Tìm DID theo ETH address |
| POST   | `/api/credential/issue` | Xác thực + cấp Verifiable Credential |
| POST   | `/api/credential/verify` | Xác thực Verifiable Credential |
| GET    | `/api/did/info` | Thống kê DID system |

### Request

```bash
curl -X POST "http://localhost:8000/api/verify" \
  -F "file=@image.jpg" \
  -F "user_address=0xYourWalletAddress"
```

### Response

```json
{
  "label": "REAL",
  "confidence": 0.92,
  "real_prob": 0.92,
  "fake_prob": 0.08,
  "image_hash": "a1b2c3...",
  "signature": "abc123...",
  "debug_msg": "0xAddress:imageHash:true"
}
```

## Kết quả AI Model

| Metric        | Giá trị                          |
| ------------- | -------------------------------- |
| Test Accuracy | 93.33%                           |
| Model         | EfficientNet-B0                  |
| Temperature   | 3.5                              |
| Dataset       | 6000 images (50% real, 50% fake) |

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
