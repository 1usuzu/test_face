import sys
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

# Add ai_deepfake to path for import
_ai_path = Path(__file__).parent.parent / "ai_deepfake"
if _ai_path.exists():
    sys.path.insert(0, str(_ai_path))

# Add did_system to path
_did_path = Path(__file__).parent.parent / "did_system"
if _did_path.exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from eth_account import Account
from eth_account.messages import encode_defunct
import uvicorn
import hashlib
import tempfile

# Try to import ensemble detector first (best), then v4, then v1
try:
    from detect import DeepfakeDetector
    DETECTOR_VERSION = "ensemble"
except ImportError:
    print("Error: detect.py not found")
    DETECTOR_VERSION = "none"

from zkp_oracle import ZKPOracle

# DID System imports - optional, will work without if not installed
try:
    from did_system import DIDService, VerifiableCredential
    DID_AVAILABLE = True
except ImportError:
    DID_AVAILABLE = False
    print("Warning: DID System not available")

# --- CẤU HÌNH BẢO MẬT (PRIVATE KEY) ---
# Lấy từ biến môi trường. Nếu không có, dùng Hardhat Account #0 (chỉ dùng cho dev)
SERVER_PRIVATE_KEY = os.environ.get(
    "SERVER_PRIVATE_KEY", 
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
) 

detector = None
zkp_oracle = None
did_service = None
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, zkp_oracle, did_service
    print("Starting Deepfake Verification API...")
    print(f"Detector version: {DETECTOR_VERSION}")
    
    model_dir = Path(__file__).parent.parent / "ai_deepfake" / "models"
    if (model_dir / "best_model.pth").exists():
        detector = DeepfakeDetector(model_dir=str(model_dir))
        print("Ensemble AI Detector initialized (EfficientNet-B0 + B4)")
    else:
        print(f"Model not found at {model_dir}")
    
    # Initialize ZKP Oracle
    zkp_oracle = ZKPOracle(SERVER_PRIVATE_KEY)
    print(f"ZKP Oracle initialized (Address: {zkp_oracle.oracle_address})")
    
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

app = FastAPI(title="Deepfake Verification API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/verify")
async def verify_image(
    file: UploadFile = File(...),
    user_address: str = Form(...)  # <--- BẮT BUỘC: Địa chỉ ví người dùng để ký
):
    """Verify if an image is real or deepfake and return a signed result"""
    
    if detector is None:
        raise HTTPException(status_code=503, detail="AI Detector not initialized")
    
    if not file.content_type.startswith("image/"):
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
        detection_result = detector.detect(temp_file.name, use_tta=True)
        result = {
            "label": "FAKE" if detection_result.is_fake else "REAL",
            "confidence": detection_result.confidence,
            "fake_prob": detection_result.fake_probability,
            "real_prob": 1 - detection_result.fake_probability,
            "risk_level": detection_result.risk_level.value,
            "methods_used": detection_result.methods_used,
            "method_scores": detection_result.method_scores,
            "recommendation": detection_result.recommendation
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
            "risk_level": result.get("risk_level", "unknown"),
            "methods_used": result.get("methods_used", ["neural"]),
            "recommendation": result.get("recommendation", "")
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
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
    
    if detector is None:
        raise HTTPException(status_code=503, detail="AI Detector not initialized")
    
    if not file.content_type.startswith("image/"):
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
        
        detection_result = detector.detect(temp_file.name, use_tta=True)
        result = {
            "label": "FAKE" if detection_result.is_fake else "REAL",
            "confidence": detection_result.confidence,
            "risk_level": detection_result.risk_level.value
        }
        
        timestamp = int(time.time())
        
        # 2. Chỉ ảnh REAL mới có thể tạo ZK Proof
        if result["label"] != "REAL":
            return {
                "can_generate_proof": False,
                "message": "Chỉ ảnh REAL mới có thể tạo Zero-Knowledge Proof",
                "label": result["label"],
                "confidence": result["confidence"]
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
            "can_generate_proof": True,
            "label": result["label"],
            "confidence": result["confidence"],
            "image_hash": image_hash,
            "filename": file.filename,
            
            # ZKP specific data
            "zkp_input": {
                "oracle_secret": zkp_input.oracle_secret,
                "timestamp": zkp_input.timestamp,
                "oracle_address": zkp_oracle.oracle_address
            },
            
            # Legacy support
            "signature": signature
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
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
    if detector is None:
        raise HTTPException(status_code=503, detail="AI Detector not initialized")
    
    if not DID_AVAILABLE or did_service is None:
        raise HTTPException(status_code=503, detail="DID Service not available")
    
    if not file.content_type.startswith("image/"):
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
        result = detector.predict(temp_file.name)
        
        # 2. Create Oracle Signature
        is_real = result["label"] == "REAL"
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
            confidence=result["confidence"],
            oracle_signature=signature
        )
        
        return {
            "label": result["label"],
            "confidence": result["confidence"],
            "image_hash": image_hash,
            "credential": credential.to_dict(),
            "credential_id": credential.id,
            "signature": signature
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
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
    uvicorn.run("api:app", host="0.0.0.0", port=8000)