"""
detect.py - Deepfake Detector v3.0 (Production + Face Extraction)
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Thư viện cắt mặt (Face Extraction)
try:
    from facenet_pytorch import MTCNN
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    print("Warning: 'facenet-pytorch' not found. Face extraction disabled.")

# Import config
try:
    from ai_config import settings
except ImportError:
    # Fallback config
    class Settings:
        MODEL_DIR = Path(__file__).parent / "models"
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        DEFAULT_THRESHOLD = 0.5
        V1_WEIGHT, V2_WEIGHT = 0.4, 0.6
    settings = Settings()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepfakeDetector")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not found. Signal analysis disabled.")

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DetectionResult:
    is_fake: bool
    confidence: float
    fake_probability: float
    risk_level: RiskLevel
    processing_time: float
    details: Dict[str, Any]

    def to_dict(self):
        d = asdict(self)
        d['risk_level'] = self.risk_level.value
        return d

class _EfficientNetB4(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=None)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4), nn.Linear(1792, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(p=0.2), nn.Linear(512, 2)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

class DeepfakeDetector:
    _instance = None # Singleton Instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepfakeDetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        
        self.device = settings.DEVICE
        self.threshold = settings.DEFAULT_THRESHOLD
        
        # 1. Load Deepfake Models
        self._load_models()
        
        # 2. Load Face Detector (MTCNN)
        if FACE_DETECTION_AVAILABLE:
            try:
                self.mtcnn = MTCNN(
                    keep_all=False, 
                    select_largest=True, 
                    device=self.device,
                    margin=20 # Lấy rộng ra một chút quanh mặt
                )
                logger.info("Face Detector (MTCNN) loaded.")
            except Exception as e:
                logger.error(f"Failed to load MTCNN: {e}")
                self.mtcnn = None
        else:
            self.mtcnn = None

        self._setup_transforms()
        self._initialized = True
        logger.info(f"AI Engine initialized on {self.device}")

    def _load_models(self):
        # Model V1 (EfficientNet-B0)
        try:
            self.model_v1 = models.efficientnet_b0(weights=None)
            self.model_v1.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 2))
            ckpt = torch.load(settings.MODEL_DIR / 'best_model.pth', map_location=self.device, weights_only=False)
            self.model_v1.load_state_dict(ckpt.get('model_state_dict', ckpt))
            self.model_v1.to(self.device).eval()
        except Exception as e:
            logger.error(f"Failed to load Model V1: {e}")
            self.model_v1 = None

        # Model V2 (EfficientNet-B4)
        try:
            self.model_v2 = _EfficientNetB4()
            ckpt = torch.load(settings.MODEL_DIR / 'best_model_v2.pth', map_location=self.device, weights_only=False)
            self.model_v2.load_state_dict(ckpt.get('model_state_dict', ckpt))
            self.model_v2.to(self.device).eval()
        except Exception as e:
            logger.error(f"Failed to load Model V2: {e}")
            self.model_v2 = None
            
        if not self.model_v1 and not self.model_v2:
            raise RuntimeError("CRITICAL: No deepfake detection models loaded!")

    def _setup_transforms(self):
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.tx_v1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
        self.tx_v2 = transforms.Compose([transforms.Resize((380, 380)), transforms.ToTensor(), norm])

    def _analyze_signal(self, img_np):
        """Phân tích tín hiệu ảnh (Frequency & Texture)"""
        if not CV2_AVAILABLE: return 0.0
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            # Texture analysis
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Frequency analysis
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.log1p(np.abs(fshift))
            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            # Mask che phần tần số thấp (trung tâm), chỉ lấy tần số cao (rìa)
            mask = (x - center_x)**2 + (y - center_y)**2 >= (min(h,w)//4)**2 
            high_freq_energy = magnitude[mask].mean()
            
            boost = 0.0
            # Ảnh deepfake thường: quá mịn (var thấp) hoặc nhiễu tần số cao (energy cao)
            if laplacian_var < 100: boost += 0.03 
            if high_freq_energy > 13.0: boost += 0.03 
            return boost
        except Exception:
            return 0.0

    def predict(self, image_path: str, threshold: float = None) -> DetectionResult:
        start_t = time.time()
        active_thresh = threshold or self.threshold
        face_detected = False
        
        try:
            # 1. Load ảnh
            img_original = Image.open(image_path).convert('RGB')
            img_to_process = img_original

            # 2. Face Extraction (QUAN TRỌNG)
            if self.mtcnn:
                try:
                    # MTCNN trả về tensor đã crop nếu tìm thấy mặt
                    # Trả về None nếu không tìm thấy
                    face_tensor = self.mtcnn(img_original)
                    
                    if face_tensor is not None:
                        # Convert tensor ngược lại PIL Image để đưa vào pipeline cũ
                        # Lưu ý: MTCNN đã normalize, nên cần denormalize hoặc convert khéo léo
                        # Cách đơn giản: Dùng box để crop từ ảnh gốc
                        boxes, _ = self.mtcnn.detect(img_original)
                        if boxes is not None:
                            box = boxes[0] # Lấy mặt to nhất
                            img_to_process = img_original.crop(box)
                            face_detected = True
                            logger.info("Face detected and cropped.")
                except Exception as e:
                    logger.warning(f"MTCNN Error: {e}. Using full image.")

            # 3. Chuẩn bị ảnh cho model
            img_np = np.array(img_to_process)
            
            # 4. Neural Network Inference
            preds = []
            with torch.no_grad():
                if self.model_v1:
                    t_in = self.tx_v1(img_to_process).unsqueeze(0).to(self.device)
                    # Class 0: Fake, Class 1: Real (Giả định theo training set của bạn)
                    # Nếu ngược lại thì dùng [0, 1]
                    prob = torch.softmax(self.model_v1(t_in), dim=1)[0, 0].item()
                    preds.append(prob * settings.V1_WEIGHT)
                
                if self.model_v2:
                    t_in = self.tx_v2(img_to_process).unsqueeze(0).to(self.device)
                    prob = torch.softmax(self.model_v2(t_in), dim=1)[0, 0].item()
                    preds.append(prob * settings.V2_WEIGHT)
            
            if not preds:
                return DetectionResult(False, 0.0, 0.0, RiskLevel.LOW, 0.0, {"error": "No model prediction"})

            # Weighted Average
            ensemble_prob = sum(preds) / (settings.V1_WEIGHT + settings.V2_WEIGHT)
            
            # 5. Signal Analysis Boost
            # Chỉ boost nếu ảnh là ảnh gốc hoặc ảnh crop chất lượng cao
            boost = self._analyze_signal(img_np)
            
            # 6. Final Logic
            final_prob = min(1.0, ensemble_prob + boost)
            is_fake = final_prob >= active_thresh
            
            # 7. Risk Level
            if final_prob > 0.85: r = RiskLevel.CRITICAL
            elif final_prob > 0.65: r = RiskLevel.HIGH
            elif final_prob > 0.40: r = RiskLevel.MEDIUM
            else: r = RiskLevel.LOW
            
            return DetectionResult(
                is_fake=is_fake,
                confidence=final_prob,
                fake_probability=final_prob,
                risk_level=r,
                processing_time=time.time() - start_t,
                details={
                    "face_detected": face_detected,
                    "model_score": ensemble_prob, 
                    "signal_boost": boost
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            return DetectionResult(False, 0.0, 0.0, RiskLevel.LOW, 0.0, {"error": str(e)})

# Khởi tạo instance
detector = DeepfakeDetector()