"""
Test Deepfake Detection Models
==============================
Script để test và so sánh các model deepfake detection.

Usage:
    python test_model.py                    # Test ensemble (default)
    python test_model.py --model v1         # Test model v1 only
    python test_model.py --model v2         # Test model v2 only
    python test_model.py --compare          # So sánh tất cả models
    python test_model.py --find-threshold   # Tìm threshold tối ưu
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse
import sys


def test_single_model(model_name: str = 'v1', threshold: float = 0.5):
    """Test a single model (v1 or v2)"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dir = Path(__file__).parent / 'models'
    
    print(f"Testing Model {model_name.upper()}")
    print(f"Device: {device}")
    print(f"Threshold: {threshold}")
    print("-" * 60)
    
    # Load model
    if model_name == 'v1':
        model_path = model_dir / 'best_model.pth'
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 2)
        )
        img_size = 224
    else:  # v2
        model_path = model_dir / 'best_model_v2.pth'
        from detect import _EfficientNetB4
        model = _EfficientNetB4()
        img_size = 380
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    val_acc = checkpoint.get('val_acc', 'N/A')
    print(f"Model loaded (Val Acc: {val_acc})")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test data
    test_dataset = datasets.ImageFolder(
        root=str(Path(__file__).parent / 'dataset_final' / 'test'),
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Test dataset: {len(test_dataset)} images")
    print(f"Classes: {test_dataset.classes}")
    print()
    
    # Test
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 0]  # Class 0 = Fake
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics with threshold
    tp = tn = fp = fn = 0
    for prob, label in zip(all_probs, all_labels):
        is_fake_pred = prob >= threshold
        is_fake_actual = label == 0  # Class 0 = Fake
        
        if is_fake_actual and is_fake_pred: tp += 1
        elif not is_fake_actual and not is_fake_pred: tn += 1
        elif not is_fake_actual and is_fake_pred: fp += 1
        else: fn += 1
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total * 100
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Model {model_name.upper()} Results (Threshold: {threshold})")
    print(f"{'='*60}")
    print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"  Accuracy:  {accuracy:.2f}%")
    print(f"  Recall:    {recall:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  F1-Score:  {f1:.2f}%")
    
    return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1, 'fn': fn, 'fp': fp}


def test_ensemble(threshold: float = 0.5):
    """Test ensemble detector"""
    from detect import DeepfakeDetector
    
    print("Testing Ensemble Detector")
    print("=" * 60)
    
    detector = DeepfakeDetector(threshold=threshold)
    
    test_dir = Path(__file__).parent / 'dataset_final' / 'test'
    fake_images = list((test_dir / 'fake').glob('*'))
    real_images = list((test_dir / 'real').glob('*'))
    
    print(f"Test Set: {len(fake_images)} fake, {len(real_images)} real")
    print(f"Threshold: {threshold}")
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
    print(f"Ensemble Results (Threshold: {threshold})")
    print(f"{'='*60}")
    print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"  Accuracy:  {accuracy:.2f}%")
    print(f"  Recall:    {recall:.2f}% (Fake detection rate)")
    print(f"  Precision: {precision:.2f}%")
    print(f"  F1-Score:  {f1:.2f}%")
    
    return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1, 'fn': fn, 'fp': fp}


def compare_models():
    """So sánh tất cả models"""
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print()
    
    results = {}
    
    for model in ['v1', 'v2']:
        print(f"\n{'='*60}")
        results[model] = test_single_model(model, threshold=0.5)
    
    print(f"\n{'='*60}")
    results['ensemble'] = test_ensemble(threshold=0.5)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<12} {'Accuracy':<12} {'Recall':<12} {'F1':<12} {'FN':<8}")
    print("-"*56)
    for name, r in results.items():
        if r:
            print(f"{name:<12} {r['accuracy']:.2f}%{'':<5} {r['recall']:.2f}%{'':<5} {r['f1']:.2f}%{'':<5} {r['fn']}")


def find_optimal_threshold():
    """Tìm threshold tối ưu cho ensemble"""
    from detect import DeepfakeDetector
    
    print("Finding Optimal Threshold")
    print("="*60)
    
    detector = DeepfakeDetector(threshold=0.5)
    
    test_dir = Path(__file__).parent / 'dataset_final' / 'test'
    fake_images = list((test_dir / 'fake').glob('*'))
    real_images = list((test_dir / 'real').glob('*'))
    
    print(f"Test Set: {len(fake_images)} fake, {len(real_images)} real")
    print()
    
    # Get all probabilities
    all_probs = []
    all_labels = []
    
    all_images = [(p, True) for p in fake_images] + [(p, False) for p in real_images]
    
    for img_path, is_fake_actual in tqdm(all_images, desc="Predicting"):
        result = detector.detect(str(img_path))
        all_probs.append(result.fake_probability)
        all_labels.append(is_fake_actual)
    
    print()
    print(f"{'Threshold':<12} {'Acc':<10} {'Recall':<10} {'Prec':<10} {'F1':<10} {'FN':<6} {'FP':<6}")
    print("-"*66)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        tp = tn = fp = fn = 0
        
        for prob, is_fake in zip(all_probs, all_labels):
            is_fake_pred = prob >= threshold
            
            if is_fake and is_fake_pred: tp += 1
            elif not is_fake and not is_fake_pred: tn += 1
            elif not is_fake and is_fake_pred: fp += 1
            else: fn += 1
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total * 100
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:<12.2f} {accuracy:<10.2f} {recall:<10.2f} {precision:<10.2f} {f1:<10.2f} {fn:<6} {fp:<6}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print()
    print(f"Best threshold for F1: {best_threshold} (F1={best_f1:.2f}%)")
    print()
    print("Recommendations:")
    print("  - Identity verification (minimize FN): threshold = 0.40")
    print("  - Balanced (default): threshold = 0.50")
    print("  - Fewer false alarms (minimize FP): threshold = 0.60")


def main():
    parser = argparse.ArgumentParser(description='Test Deepfake Detection Models')
    parser.add_argument('--model', type=str, choices=['v1', 'v2', 'ensemble'], 
                       default='ensemble', help='Model to test')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--compare', action='store_true', help='Compare all models')
    parser.add_argument('--find-threshold', action='store_true', help='Find optimal threshold')
    
    args = parser.parse_args()
    
    if args.find_threshold:
        find_optimal_threshold()
    elif args.compare:
        compare_models()
    elif args.model == 'ensemble':
        test_ensemble(args.threshold)
    else:
        test_single_model(args.model, args.threshold)


if __name__ == "__main__":
    main()
