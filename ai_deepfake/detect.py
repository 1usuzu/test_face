"""
Deepfake/AI Image Detector - Production Version v2.1
=====================================================
Ensemble: EfficientNet-B0 + EfficientNet-B4 + Signal Analysis

Features:
- Dual model ensemble với weighted averaging
- Test Time Augmentation (TTA) cho độ chính xác cao hơn  
- Frequency & Texture analysis (phụ trợ)
- Configurable threshold (default: 0.50)
- Batch prediction support
- Full logging & versioning

Performance (Test Set n=450):
- Threshold 0.50: Accuracy 94.89%, Recall 97.78%, FN=5
- Threshold 0.55: Accuracy 95.11%, Recall 96.89%, FN=7
"""

__version__ = "2.1.0"
__author__ = "AI Deepfake Detection Team"

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import logging
import time

# Tắt cảnh báo weights của Torchvision
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not found. Signal analysis will be disabled.")


class RiskLevel(Enum):
    """Mức độ rủi ro của ảnh"""
    LOW = "low"         # An toàn
    MEDIUM = "medium"   # Nghi ngờ
    HIGH = "high"       # Nguy hiểm
    CRITICAL = "critical" # Chắc chắn giả


@dataclass
class DetectionResult:
    """Kết quả phát hiện deepfake"""
    is_fake: bool
    confidence: float
    fake_probability: float
    risk_level: RiskLevel
    methods_used: List[str]
    method_scores: Dict[str, float]
    recommendation: str
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['risk_level'] = self.risk_level.value
        return result
    
    def __repr__(self) -> str:
        return (f"DetectionResult(is_fake={self.is_fake}, "
                f"confidence={self.confidence:.2%}, "
                f"risk={self.risk_level.value})")


class _EfficientNetB4(nn.Module):
    """Kiến trúc EfficientNet-B4 tùy chỉnh"""
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=None)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        return self.classifier(self.backbone(x))


class DeepfakeDetector:
    """
    Bộ phát hiện Deepfake tích hợp Ensemble Model + Phân tích tín hiệu.
    
    Args:
        model_dir: Đường dẫn thư mục chứa model
        device: 'cuda' hoặc 'cpu' (auto-detect nếu None)
        threshold: Ngưỡng quyết định (0.0 - 1.0). Mặc định 0.50
        
    Example:
        >>> detector = DeepfakeDetector(threshold=0.50)
        >>> result = detector.detect("image.jpg")
        >>> print(result.is_fake, result.confidence)
    """
    
    # Cấu hình mặc định - Đã tối ưu qua testing
    DEFAULT_THRESHOLD = 0.50  # Balance giữa recall và precision
    V1_WEIGHT = 0.4  # EfficientNet-B0 (Nhẹ, nhanh)
    V2_WEIGHT = 0.6  # EfficientNet-B4 (Chính xác cao)
    
    # Recommended thresholds for different use cases
    THRESHOLDS = {
        'strict': 0.40,      # Ít bỏ sót fake (FN thấp, FP cao) - Cho identity verification
        'balanced': 0.50,    # Cân bằng - Recommended default
        'permissive': 0.60,  # Ít false alarm (FP thấp, FN cao)
    }
    
    def __init__(self, model_dir: str = None, device: str = None, threshold: float = None):
        """
        Khởi tạo Detector.
        Args:
            model_dir: Đường dẫn thư mục chứa model.
            device: 'cuda' hoặc 'cpu'.
            threshold: Ngưỡng quyết định (0.0 - 1.0). Mặc định 0.5.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.THRESHOLD = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        
        # Đường dẫn model
        if model_dir is None:
            self.model_dir = Path(__file__).parent / 'models'
        else:
            self.model_dir = Path(model_dir)
        
        self._load_models()
        self._setup_transforms()
        
        logger.info(f"DeepfakeDetector v{__version__} initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Threshold: {self.THRESHOLD}")
        logger.info(f"  Models: V1={'✓' if self.model_v1 else '✗'}, V2={'✓' if self.model_v2 else '✗'}")
    
    def _load_models(self):
        """Load weights cho cả 2 model"""
        # Model v1 (EfficientNet-B0)
        v1_path = self.model_dir / 'best_model.pth'
        self.model_v1 = models.efficientnet_b0(weights=None)
        self.model_v1.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 2)
        )
        
        try:
            ckpt1 = torch.load(v1_path, map_location=self.device, weights_only=False)
            state_dict1 = ckpt1['model_state_dict'] if 'model_state_dict' in ckpt1 else ckpt1
            self.model_v1.load_state_dict(state_dict1)
            self.temperature = ckpt1.get('temperature', 1.0) if isinstance(ckpt1, dict) else 1.0
            logger.debug(f"Model V1 loaded from {v1_path}")
        except Exception as e:
            logger.error(f"Error loading Model V1: {e}")
            self.model_v1 = None

        if self.model_v1:
            self.model_v1 = self.model_v1.to(self.device).eval()
        
        # Model v2 (EfficientNet-B4)
        v2_path = self.model_dir / 'best_model_v2.pth'
        self.model_v2 = _EfficientNetB4()
        
        try:
            ckpt2 = torch.load(v2_path, map_location=self.device, weights_only=False)
            state_dict2 = ckpt2['model_state_dict'] if 'model_state_dict' in ckpt2 else ckpt2
            self.model_v2.load_state_dict(state_dict2)
            logger.debug(f"Model V2 loaded from {v2_path}")
        except Exception as e:
            logger.error(f"Error loading Model V2: {e}")
            self.model_v2 = None
            
        if self.model_v2:
            self.model_v2 = self.model_v2.to(self.device).eval()

        if not self.model_v1 and not self.model_v2:
            raise RuntimeError("Không thể load bất kỳ model nào!")
    
    def _setup_transforms(self):
        """Thiết lập các phép biến đổi ảnh"""
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        self.transform_v1 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            norm
        ])
        
        self.transform_v2 = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            norm
        ])
        
        # TTA (Test Time Augmentation) - Flip ngang
        self.tta_transform_v1 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            norm
        ])
        
        self.tta_transform_v2 = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            norm
        ])
    
    def _predict_batch(self, model, tensor_list):
        """Predict on a batch of tensors"""
        if not model: return 0.0
        batch = torch.cat(tensor_list).to(self.device)
        with torch.no_grad():
            logits = model(batch)
            if model == self.model_v1:
                logits = logits / self.temperature
            # Class 0: Fake, Class 1: Real (theo ImageFolder sort order)
            probs = torch.softmax(logits, dim=1)[:, 0]
        return probs.mean().item()

    def _analyze_frequency(self, img_np: np.ndarray) -> Dict[str, Any]:
        if not CV2_AVAILABLE: return {'available': False, 'score': 0.5}
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        r = min(cx, cy)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        high_mask = dist >= r * 0.7
        total = magnitude.sum()
        high_energy = magnitude[high_mask].sum() / total if total > 0 else 0
        score = 0.5 + (0.15 if high_energy > 0.12 else 0) # Giảm boost xuống vì threshold thấp
        return {'available': True, 'score': min(1.0, score)}
    
    def _analyze_texture(self, img_np: np.ndarray) -> Dict[str, Any]:
        if not CV2_AVAILABLE: return {'available': False, 'score': 0.5}
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        score = 0.5
        if variance < 200: score += 0.15
        return {'available': True, 'variance': float(variance), 'score': min(1.0, score)}
    
    def detect(self, image_path: str, use_tta: bool = True) -> DetectionResult:
        """
        Phát hiện deepfake trong ảnh.
        
        Args:
            image_path: Đường dẫn đến file ảnh
            use_tta: Sử dụng Test Time Augmentation (flip) để tăng độ chính xác
            
        Returns:
            DetectionResult với các thông tin chi tiết
        """
        start_time = time.time()
        
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Cannot open image: {image_path} - {e}")
            return DetectionResult(False, 0.0, 0.0, RiskLevel.LOW, [], {}, f"Error: {str(e)}", {})

        img_np = np.array(img)
        
        # 1. Neural Network Prediction
        inputs_v1 = [self.transform_v1(img).unsqueeze(0)]
        inputs_v2 = [self.transform_v2(img).unsqueeze(0)]
        
        if use_tta:
            inputs_v1.append(self.tta_transform_v1(img).unsqueeze(0))
            inputs_v2.append(self.tta_transform_v2(img).unsqueeze(0))
            
        prob_v1 = self._predict_batch(self.model_v1, inputs_v1)
        prob_v2 = self._predict_batch(self.model_v2, inputs_v2)
        
        # Weighted Ensemble
        if self.model_v1 and self.model_v2:
            ensemble_prob = self.V1_WEIGHT * prob_v1 + self.V2_WEIGHT * prob_v2
        else:
            ensemble_prob = prob_v1 if self.model_v1 else prob_v2
            
        # 2. Signal Analysis
        freq = self._analyze_frequency(img_np)
        texture = self._analyze_texture(img_np)
        
        # 3. Smart Signal Boost (Điều chỉnh cho Threshold 0.5)
        signal_boost = 0
        # Vùng tranh chấp giờ là 0.30 - 0.60
        if 0.30 < ensemble_prob < 0.60:
            if freq.get('score', 0) > 0.6: signal_boost += 0.03
            if texture.get('score', 0) > 0.6: signal_boost += 0.03
            
        final_prob = min(1.0, ensemble_prob + signal_boost)
        is_fake = final_prob >= self.THRESHOLD
        
        # 4. Determine Risk Level & Recommendation (Updated for 0.5)
        if is_fake:
            if final_prob >= 0.80:
                risk_level = RiskLevel.CRITICAL
                rec = "CẢNH BÁO ĐỎ: Tỷ lệ giả mạo rất cao."
            elif final_prob >= 0.65:
                risk_level = RiskLevel.HIGH
                rec = "Cảnh báo: Ảnh có dấu hiệu giả mạo rõ ràng."
            else: # 0.50 - 0.64
                risk_level = RiskLevel.MEDIUM
                rec = "Nghi ngờ: Điểm số vừa vượt ngưỡng an toàn."
        else:
            if final_prob >= 0.35:
                risk_level = RiskLevel.MEDIUM
                rec = "Cần lưu ý: Ảnh thật nhưng chất lượng kém hoặc nhiễu."
            else:
                risk_level = RiskLevel.LOW
                rec = "An toàn: Không phát hiện bất thường."
        
        # 5. Metadata
        methods = ['neural_ensemble']
        if freq['available']: methods.append('freq_analysis')
        
        elapsed_time = time.time() - start_time
        
        return DetectionResult(
            is_fake=is_fake,
            confidence=final_prob,
            fake_probability=ensemble_prob,
            risk_level=risk_level,
            methods_used=methods,
            method_scores={'ensemble': ensemble_prob, 'v1': prob_v1, 'v2': prob_v2, 'boost': signal_boost},
            recommendation=rec,
            details={'threshold': self.THRESHOLD, 'inference_time_ms': elapsed_time * 1000}
        )
    
    def detect_batch(self, image_paths: List[str], use_tta: bool = True) -> List[DetectionResult]:
        """
        Phát hiện deepfake cho nhiều ảnh.
        
        Args:
            image_paths: Danh sách đường dẫn ảnh
            use_tta: Sử dụng Test Time Augmentation
            
        Returns:
            Danh sách DetectionResult
        """
        return [self.detect(path, use_tta) for path in image_paths]
    
    def get_raw_probability(self, image_path: str) -> float:
        """
        Lấy xác suất fake thô (không có signal boost).
        Hữu ích cho việc tìm threshold tối ưu.
        """
        result = self.detect(image_path, use_tta=True)
        return result.fake_probability
    
    def set_threshold(self, threshold: float) -> None:
        """Thay đổi threshold dynamically"""
        if 0.0 <= threshold <= 1.0:
            self.THRESHOLD = threshold
            logger.info(f"Threshold updated to {threshold}")
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
    
    @classmethod
    def get_version(cls) -> str:
        """Trả về version của detector"""
        return __version__

# Test function
if __name__ == '__main__':
    from pathlib import Path
    from tqdm import tqdm
    import argparse
    
    parser = argparse.ArgumentParser(description='Deepfake Detection')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--test', action='store_true', help='Run full test on dataset')
    parser.add_argument('--threshold', type=float, default=0.50, help='Detection threshold')
    args = parser.parse_args()
    
    print(f"Deepfake Detector v{__version__}")
    print("="*60)
    
    detector = DeepfakeDetector(threshold=args.threshold)
    
    if args.image:
        # Single image detection
        result = detector.detect(args.image)
        print(f"\nImage: {args.image}")
        print(f"Result: {'FAKE' if result.is_fake else 'REAL'}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Risk Level: {result.risk_level.value}")
        print(f"Method Scores: {result.method_scores}")
        print(f"Recommendation: {result.recommendation}")
    
    elif args.test:
        # Full test on dataset
        test_dir = Path(__file__).parent / 'dataset_final' / 'test'
        fake_dir = test_dir / 'fake'
        real_dir = test_dir / 'real'
        
        if not test_dir.exists():
            print(f"Test directory not found: {test_dir}")
            exit(1)
        
        fake_images = list(fake_dir.glob('*'))
        real_images = list(real_dir.glob('*'))
        
        print(f"\nTest Set: {len(fake_images)} fake, {len(real_images)} real")
        print(f"Threshold: {args.threshold}")
        print()
        
        tp = tn = fp = fn = 0
        
        all_images = [(p, True) for p in fake_images] + [(p, False) for p in real_images]
        
        for img_path, is_fake_actual in tqdm(all_images, desc="Testing"):
            result = detector.detect(str(img_path))
            
            if is_fake_actual and result.is_fake: tp += 1
            elif not is_fake_actual and not result.is_fake: tn += 1
            elif not is_fake_actual and result.is_fake: fp += 1
            else: fn += 1
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total * 100
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"RESULTS (Threshold: {args.threshold})")
        print(f"{'='*60}")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"  Accuracy:  {accuracy:.2f}%")
        print(f"  Recall:    {recall:.2f}% (Fake detection rate)")
        print(f"  Precision: {precision:.2f}%")
        print(f"  F1-Score:  {f1:.2f}%")
        
        if fn > 0:
            print(f"\n  ⚠️ {fn} fake images were misclassified as REAL (False Negatives)")
        if fp > 0:
            print(f"  ⚠️ {fp} real images were misclassified as FAKE (False Positives)")
    
    else:
        print("\nUsage:")
        print("  python detect.py --image <path>      # Detect single image")
        print("  python detect.py --test              # Run full test")
        print("  python detect.py --test --threshold 0.45")
        print()
        print("Recommended thresholds:")
        for name, val in DeepfakeDetector.THRESHOLDS.items():
            print(f"  {name}: {val}")