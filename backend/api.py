import sys
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

# Use absolute repo paths to avoid uvicorn reload/import cwd quirks.
_repo_root = Path(__file__).resolve().parent.parent

# Add ai_deepfake to path for import
_ai_path = _repo_root / "ai_deepfake"
if _ai_path.exists():
    sys.path.insert(0, str(_ai_path))

# Add did_system to path
_did_path = _repo_root / "did_system"
if _did_path.exists():
    sys.path.insert(0, str(_repo_root))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from eth_account import Account
from eth_account.messages import encode_defunct
import uvicorn
import hashlib
import tempfile
import threading

def _load_local_env_file() -> None:
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return

    # utf-8-sig strips a leading BOM; assign (not setdefault) so backend/.env wins over stale shell/IDE
    # placeholders like SERVER_PRIVATE_KEY=0x_your_... that would otherwise block a fixed .env.
    for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value

_load_local_env_file()

# Try to import ensemble detector first (best), then v4, then v1
DeepfakeDetector = None
try:
    from detect import DeepfakeDetector
    DETECTOR_VERSION = "ensemble"
except ImportError:
    import traceback
    print("Warning: failed to import AI detector (detect.DeepfakeDetector). AI detection disabled.")
    traceback.print_exc()
    DETECTOR_VERSION = "none"

from zkp_oracle import ZKPOracle
from blockchain_client import BlockchainClient

# DID System imports - optional, will work without if not installed
try:
    from did_system import DIDService, VerifiableCredential
    DID_AVAILABLE = True
except ImportError:
    DID_AVAILABLE = False
    print("Warning: DID System not available")

# --- CẤU HÌNH BẢO MẬT (PRIVATE KEY) ---
# BẮT BUỘC: Lấy từ biến môi trường, KHÔNG hardcode trong source
SERVER_PRIVATE_KEY = os.environ.get("SERVER_PRIVATE_KEY")
ALLOW_INSECURE_DEV_KEY = os.environ.get("ALLOW_INSECURE_DEV_KEY", "false").lower() == "true"
INSECURE_DEV_PRIVATE_KEY = os.environ.get("INSECURE_DEV_PRIVATE_KEY")

# CORS origins: read from ALLOWED_ORIGINS env (comma-separated).
# Example: "https://app.vercel.app,https://consumer.vercel.app"
_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [origin.strip() for origin in _origins_env.split(",") if origin.strip()]
ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS

detector = None
detector_init_error = None
detector_lock = threading.Lock()
zkp_oracle = None
did_service = None
blockchain_client = None
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def _detector_status_value(detection_result) -> str:
    status = getattr(detection_result, "status", None)
    if status is None:
        return "ok"
    return getattr(status, "value", str(status)).lower()


def _status_error_code(status_value: str, details: dict) -> str:
    if status_value == "no_face":
        return "NO_FACE_DETECTED"
    if status_value == "face_detection_error":
        return details.get("error", "FACE_DETECTION_ERROR")
    if status_value == "no_model":
        return details.get("error", "NO_MODEL")
    return details.get("error", "DETECTION_ERROR")


def _ensure_detector_ready():
    global detector, detector_init_error
    if detector is not None:
        return detector
    if DeepfakeDetector is None:
        raise HTTPException(status_code=503, detail="AI Detector not available")

    with detector_lock:
        if detector is not None:
            return detector
        if detector_init_error is not None:
            raise HTTPException(status_code=503, detail=f"AI Detector unavailable: {detector_init_error}")
        try:
            detector = DeepfakeDetector()
            print("AI Detector initialized lazily on first request")
        except Exception as e:
            detector_init_error = str(e)
            raise HTTPException(status_code=503, detail=f"AI Detector initialization failed: {e}")
    return detector

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, detector_init_error, zkp_oracle, did_service, blockchain_client
    print("Starting Deepfake Verification API...")
    print(f"Detector version: {DETECTOR_VERSION}")

    global SERVER_PRIVATE_KEY
    if not SERVER_PRIVATE_KEY and ALLOW_INSECURE_DEV_KEY:
        if not INSECURE_DEV_PRIVATE_KEY:
            raise RuntimeError(
                "ALLOW_INSECURE_DEV_KEY=true but INSECURE_DEV_PRIVATE_KEY is missing."
            )
        SERVER_PRIVATE_KEY = INSECURE_DEV_PRIVATE_KEY
        print("WARNING: Using insecure dev private key fallback. Do not use in production.")

    if not SERVER_PRIVATE_KEY:
        raise RuntimeError(
            "SERVER_PRIVATE_KEY is required. Set it in environment before starting backend."
        )
    
    print(f"Python executable: {sys.executable}")
    
    # Do not initialize detector at startup on low-memory hosts.
    detector = None
    detector_init_error = None
    if DeepfakeDetector is None:
        print("AI Detector import unavailable - running in limited mode")
    else:
        print("AI Detector will be initialized lazily on first verify request")
    
    # Initialize ZKP Oracle
    zkp_oracle = ZKPOracle(SERVER_PRIVATE_KEY)
    print(f"ZKP Oracle initialized (Address: {zkp_oracle.oracle_address})")
    
    # Initialize Blockchain client (optional)
    try:
        blockchain_client = BlockchainClient.from_env(SERVER_PRIVATE_KEY)
        if blockchain_client:
            print("Blockchain client initialized")
        else:
            print("Blockchain client disabled (RPC_URL/CHAIN_ID/CONTRACT_ADDRESS not set)")
    except Exception as e:
        print(f"Warning: Blockchain client initialization failed: {e}")
        blockchain_client = None

    # Initialize DID Service
    if DID_AVAILABLE:
        try:
            did_service = DIDService(oracle_private_key=SERVER_PRIVATE_KEY)
            print(f"DID Service initialized (Oracle DID: {did_service.oracle_did})")
        except Exception as e:
            print(f"Warning: DID Service initialization failed: {e}")
            did_service = None
    
    yield
    print("Shutting down...")
    detector = None

app = FastAPI(title="Deepfake Verification API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Consumer API router (/v1/consumer/*)
from consumer_api import router as consumer_router
app.include_router(consumer_router)

@app.get("/api/health")
async def health_check():
    """System health check"""
    return {
        "status": "ok",
        "detector": detector is not None,
        "detector_lazy": DeepfakeDetector is not None,
        "detector_init_error": detector_init_error,
        "detector_version": DETECTOR_VERSION,
        "zkp_oracle": zkp_oracle is not None,
        "did_service": did_service is not None,
        "blockchain": blockchain_client is not None,
        "python": sys.executable
    }


# ============================================================
# BLOCKCHAIN ENDPOINTS - On-chain recording (no MetaMask required)
# ============================================================
@app.post("/api/blockchain/record")
async def blockchain_record(
    image_hash: str = Form(...),        # 64 hex chars (sha256)
    user_address: str = Form(...),      # subject address (0x...)
    subject_did: str = Form(default=""),
    is_real: bool = Form(default=True),
    confidence: float = Form(default=1.0),
):
    """
    Record verification result on-chain using server/issuer key (custodial tx).

    This is for end-to-end demo without requiring users to have MetaMask.
    """
    if blockchain_client is None:
        raise HTTPException(status_code=503, detail="Blockchain client not configured")

    try:
        tx_hash = blockchain_client.record_verification_by_issuer(
            subject_address=user_address,
            image_hash_hex=image_hash,
            subject_did=subject_did,
            is_real=is_real,
            confidence=confidence,
        )
        return {"ok": True, "tx_hash": tx_hash}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/verify")
async def verify_image(
    file: UploadFile = File(...),
    user_address: str = Form(...)  # <--- BẮT BUỘC: Địa chỉ ví người dùng để ký
):
    """Verify if an image is real or deepfake and return a signed result"""
    
    runtime_detector = _ensure_detector_ready()
    
    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_file = None
    try:
        content = await file.read()
        
        # Validate file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        suffix = Path(file.filename).suffix or ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        
        # 1. AI Dự đoán
        image_hash = hashlib.sha256(content).hexdigest()
        
        # Enhanced detector with multi-method analysis
        detection_result = runtime_detector.predict(temp_file.name)
        
        status_value = _detector_status_value(detection_result)
        if status_value != "ok":
            details = detection_result.details if isinstance(detection_result.details, dict) else {}
            return {
                "status": status_value,
                "label": "ERROR",
                "message": "Không thể phân loại ảnh một cách an toàn. Vui lòng thử lại với ảnh khuôn mặt rõ hơn.",
                "error_code": _status_error_code(status_value, details),
                "face_detected": details.get("face_detected", False),
                "confidence": 0.0,
                "risk_level": "unknown",
            }

        result = {
            "label": "FAKE" if detection_result.is_fake else "REAL",
            "confidence": detection_result.confidence,
            "fake_prob": detection_result.fake_probability,
            "real_prob": 1 - detection_result.fake_probability,
            "risk_level": detection_result.risk_level.value
        }
        
        # 2. Logic Ký số (Signing)
        # Tạo chuỗi thông điệp duy nhất để ký: "UserAddress:ImageHash:IsReal"
        is_real_str = "true" if result["label"] == "REAL" else "false"
        msg_content = f"{user_address.lower()}:{image_hash}:{is_real_str}"        
        # Hash và Ký
        message = encode_defunct(text=msg_content)
        signed_message = Account.sign_message(message, private_key=SERVER_PRIVATE_KEY)
        signature = signed_message.signature.hex()
        
        return {
            "label": result["label"],
            "confidence": result["confidence"],
            "real_prob": result.get("real_prob", 1 - result.get("fake_prob", 0)),
            "fake_prob": result.get("fake_prob", 0),
            "image_hash": image_hash,
            "filename": file.filename,
            "signature": signature,   # <--- TRẢ VỀ CHỮ KÝ CHO FRONTEND
            "debug_msg": msg_content,
            "detector_version": DETECTOR_VERSION,
            "risk_level": result.get("risk_level", "unknown")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal verification error")
        
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


# ============================================================
# ZKP ENDPOINTS - Zero-Knowledge Proof Support
# ============================================================

@app.post("/api/verify-zkp")
async def verify_image_zkp(
    file: UploadFile = File(...),
    user_address: str = Form(...)
):
    """
    Verify image và trả về ZKP input
    
    Flow:
    1. AI verify ảnh
    2. Nếu REAL → tạo oracle_secret
    3. Trả về dữ liệu để user generate ZK proof
    4. User submit proof lên blockchain (không qua backend)
    
    Privacy: Backend KHÔNG lưu kết quả, chỉ cung cấp oracle_secret
    """
    
    runtime_detector = _ensure_detector_ready()
    
    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_file = None
    try:
        content = await file.read()
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        suffix = Path(file.filename).suffix or ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        
        # 1. AI Prediction
        image_hash = hashlib.sha256(content).hexdigest()
        
        detection_result = runtime_detector.predict(temp_file.name)
        
        status_value = _detector_status_value(detection_result)
        if status_value != "ok":
            details = detection_result.details if isinstance(detection_result.details, dict) else {}
            return {
                "status": status_value,
                "can_generate_proof": False,
                "label": "ERROR",
                "message": "Không thể tạo ZKP do kết quả phát hiện không hợp lệ.",
                "error_code": _status_error_code(status_value, details),
                "face_detected": details.get("face_detected", False),
                "confidence": 0.0,
            }

        fake_p = float(detection_result.fake_probability)
        result = {
            "label": "FAKE" if detection_result.is_fake else "REAL",
            "confidence": detection_result.confidence,
            "fake_prob": fake_p,
            "real_prob": 1.0 - fake_p,
            "risk_level": detection_result.risk_level.value,
        }

        timestamp = int(time.time())

        # 2. Chỉ ảnh REAL mới có thể tạo ZK Proof
        if result["label"] != "REAL":
            return {
                "status": "ok",
                "can_generate_proof": False,
                "message": "Chỉ ảnh REAL mới có thể tạo Zero-Knowledge Proof",
                "label": result["label"],
                "confidence": result["confidence"],
                "fake_prob": result["fake_prob"],
                "real_prob": result["real_prob"],
                "risk_level": result["risk_level"],
                "image_hash": image_hash,
                "filename": file.filename,
                "detector_version": DETECTOR_VERSION,
            }
        
        # 3. Tạo ZKP Input
        zkp_input = zkp_oracle.create_zkp_input(
            image_hash=image_hash,
            is_real=True,
            confidence=result["confidence"],
            timestamp=timestamp
        )
        
        # 4. Tạo signature backup (cho legacy flow)
        is_real_str = "true"
        msg_content = f"{user_address.lower()}:{image_hash}:{is_real_str}"
        message = encode_defunct(text=msg_content)
        signed_message = Account.sign_message(message, private_key=SERVER_PRIVATE_KEY)
        signature = signed_message.signature.hex()
        
        return {
            "status": "ok",
            "can_generate_proof": True,
            "label": result["label"],
            "confidence": result["confidence"],
            "fake_prob": result["fake_prob"],
            "real_prob": result["real_prob"],
            "risk_level": result["risk_level"],
            "image_hash": image_hash,
            "filename": file.filename,
            "detector_version": DETECTOR_VERSION,

            # ZKP specific data
            "zkp_input": {
                "oracle_secret": zkp_input.oracle_secret,
                "timestamp": zkp_input.timestamp,
                "oracle_address": zkp_oracle.oracle_address
            },

            # Legacy support
            "signature": signature
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal verification error")
        
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@app.get("/api/zkp-info")
async def get_zkp_info():
    """Get ZKP system information"""
    return {
        "oracle_address": zkp_oracle.oracle_address if zkp_oracle else None,
        "supported_circuits": ["simple_proof", "deepfake_verification"],
        "poseidon_version": "circomlib-2.0.5",
        "proving_system": "groth16"
    }


# ============================================================
# DID ENDPOINTS - Decentralized Identity System
# ============================================================

@app.post("/api/did/create")
async def create_did(user_address: str = Form(...)):
    """
    Create a DID for user based on Ethereum address
    
    Returns:
        DID Document following W3C DID Core 1.0
    """
    if not DID_AVAILABLE or did_service is None:
        raise HTTPException(status_code=503, detail="DID Service not available")
    
    try:
        did, did_doc = did_service.create_user_did(user_address)
        
        return {
            "did": did,
            "document": did_doc.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/did/resolve/{did}")
async def resolve_did(did: str):
    """
    Resolve a DID to its DID Document
    
    Args:
        did: The DID to resolve (e.g., did:deepfake:abc123)
    """
    if not DID_AVAILABLE or did_service is None:
        raise HTTPException(status_code=503, detail="DID Service not available")
    
    doc = did_service.resolve_did(did)
    
    if doc is None:
        raise HTTPException(status_code=404, detail="DID not found")
    
    return {
        "did": did,
        "document": doc.to_dict()
    }


@app.get("/api/did/resolve-by-address/{address}")
async def resolve_did_by_address(address: str):
    """
    Resolve a DID by Ethereum address
    """
    if not DID_AVAILABLE or did_service is None:
        raise HTTPException(status_code=503, detail="DID Service not available")
    
    doc = did_service.resolve_by_address(address)
    
    if doc is None:
        raise HTTPException(status_code=404, detail="DID not found for this address")
    
    return {
        "address": address,
        "did": doc.id,
        "document": doc.to_dict()
    }


@app.post("/api/credential/issue")
async def issue_credential(
    file: UploadFile = File(...),
    user_address: str = Form(...)
):
    """
    Verify image and issue Verifiable Credential
    
    This combines AI verification with DID-based credential issuance
    
    Returns:
        Verifiable Credential in W3C format
    """
    runtime_detector = _ensure_detector_ready()
    
    if not DID_AVAILABLE or did_service is None:
        raise HTTPException(status_code=503, detail="DID Service not available")
    
    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_file = None
    try:
        content = await file.read()
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        suffix = Path(file.filename).suffix or ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        
        # 1. AI Prediction
        image_hash = hashlib.sha256(content).hexdigest()
        detection_result = runtime_detector.predict(temp_file.name)
        
        status_value = _detector_status_value(detection_result)
        if status_value != "ok":
            raise HTTPException(
                status_code=422,
                detail=f"Detector result is non-classifiable: {status_value}"
            )

        # 2. Create Oracle Signature
        is_real = not detection_result.is_fake
        is_real_str = "true" if is_real else "false"
        msg_content = f"{user_address.lower()}:{image_hash}:{is_real_str}"
        message = encode_defunct(text=msg_content)
        signed_message = Account.sign_message(message, private_key=SERVER_PRIVATE_KEY)
        signature = signed_message.signature.hex()
        
        # 3. Issue Verifiable Credential
        credential = did_service.issue_verification_credential(
            user_address=user_address,
            image_hash=image_hash,
            is_real=is_real,
            confidence=detection_result.confidence,
            oracle_signature=signature
        )
        
        return {
            "label": "REAL" if is_real else "FAKE",
            "confidence": detection_result.confidence,
            "image_hash": image_hash,
            "credential": credential.to_dict(),
            "credential_id": credential.id,
            "signature": signature
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal verification error")
        
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@app.post("/api/credential/verify")
async def verify_credential(credential_json: str = Form(...)):
    """
    Verify a Verifiable Credential
    
    Args:
        credential_json: The credential in JSON format
    """
    if not DID_AVAILABLE or did_service is None:
        raise HTTPException(status_code=503, detail="DID Service not available")
    
    try:
        import json
        vc_dict = json.loads(credential_json)
        credential = VerifiableCredential.from_dict(vc_dict)
        
        result = did_service.verify_credential(credential)
        
        return result.to_dict()
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/did/info")
async def get_did_info():
    """Get DID system information"""
    if not DID_AVAILABLE or did_service is None:
        return {
            "available": False,
            "message": "DID Service not initialized"
        }
    
    stats = did_service.get_statistics()
    
    return {
        "available": True,
        "oracle_did": did_service.oracle_did,
        "statistics": stats
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))