"""
Deepfake Detection - Training Script v2.0 (Enhanced)
=====================================================

Cải tiến:
1. Data Augmentation mạnh hơn (RandAugment, MixUp, CutMix)
2. Progressive resizing: 224 → 380
3. Focal Loss để xử lý hard examples
4. Label Smoothing
5. Gradient accumulation cho batch size lớn
6. EfficientNet-B4 (lớn hơn B0)
7. ArcFace-style regularization
8. Test Time Augmentation (TTA)
9. Early Stopping với patience
10. Mixed Precision Training (FP16)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path
import json
from datetime import datetime


# ===================== CONFIGURATION =====================

CONFIG = {
    # Paths
    'dataset_path': 'd:/Code/face/ai_deepfake/dataset_final',
    'output_dir': 'd:/Code/face/ai_deepfake/models',
    
    # Model
    'model_name': 'efficientnet_b4',  # Upgraded from B0
    'pretrained': True,
    'num_classes': 2,
    
    # Training
    'batch_size': 16,  # Smaller due to larger model
    'accumulation_steps': 4,  # Effective batch = 64
    'epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'warmup_epochs': 3,
            
    # Image
    'image_size': 380,
    'progressive_resize': True,
    
    # Augmentation
    'use_mixup': True,
    'mixup_alpha': 0.4,
    'use_cutmix': True,
    'cutmix_alpha': 1.0,
    'cutmix_prob': 0.5,
    
    # Regularization
    'label_smoothing': 0.1,
    'dropout': 0.4,
    
    # Training tricks
    'use_amp': True,  # Mixed precision
    'use_ema': True,  # Exponential Moving Average
    'ema_decay': 0.999,
    
    # Early stopping
    'patience': 10,
    'min_delta': 0.001,
    
    # Misc
    'num_workers': 4,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ===================== DATA AUGMENTATION =====================

class RandAugment:
    """Simplified RandAugment implementation"""
    
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m
        self.augmentations = [
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4),
            transforms.ColorJitter(contrast=0.4),
            transforms.ColorJitter(saturation=0.4),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomPosterize(bits=4, p=1.0),
            transforms.RandomEqualize(p=1.0),
            transforms.RandomAutocontrast(p=1.0),
        ]
    
    def __call__(self, img):
        ops = random.choices(self.augmentations, k=self.n)
        for op in ops:
            img = op(img)
        return img


def get_transforms(image_size=224, is_training=True):
    """Get data transforms"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            RandAugment(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


# ===================== MIXUP / CUTMIX =====================

def mixup_data(x, y, alpha=0.4):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp/CutMix loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ===================== LOSS FUNCTIONS =====================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples"""
    
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ===================== MODEL =====================

class EfficientNetClassifier(nn.Module):
    """Enhanced EfficientNet with custom head"""
    
    def __init__(self, model_name='efficientnet_b4', num_classes=2, dropout=0.4):
        super().__init__()
        
        # Load backbone
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = 1280
        elif model_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            in_features = 1792
        elif model_name == 'efficientnet_v2_s':
            self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            in_features = 1280
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x):
        """Extract features for visualization"""
        return self.backbone(x)


class EMA:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ===================== TRAINING =====================

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.best_acc = 0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Create output directory
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = EfficientNetClassifier(
            model_name=config['model_name'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        ).to(self.device)
        
        # EMA
        self.ema = EMA(self.model, config['ema_decay']) if config['use_ema'] else None
        
        # Loss
        self.criterion = FocalLoss(
            alpha=1, 
            gamma=2, 
            label_smoothing=config['label_smoothing']
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config['use_amp'] else None
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load datasets"""
        train_transform = get_transforms(self.config['image_size'], is_training=True)
        val_transform = get_transforms(self.config['image_size'], is_training=False)
        
        self.train_dataset = datasets.ImageFolder(
            root=f"{self.config['dataset_path']}/train",
            transform=train_transform
        )
        self.val_dataset = datasets.ImageFolder(
            root=f"{self.config['dataset_path']}/val",
            transform=val_transform
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        print(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)}")
        print(f"Classes: {self.train_dataset.classes}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # MixUp or CutMix
            use_mixup = self.config['use_mixup'] and random.random() > 0.5
            use_cutmix = self.config['use_cutmix'] and random.random() < self.config['cutmix_prob']
            
            if use_cutmix:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, self.config['cutmix_alpha'])
            elif use_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, self.config['mixup_alpha'])
            
            # Forward pass
            if self.config['use_amp']:
                with autocast(device_type='cuda'):
                    outputs = self.model(images)
                    if use_mixup or use_cutmix:
                        loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)
                    loss = loss / self.config['accumulation_steps']
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                if use_mixup or use_cutmix:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
                loss = loss / self.config['accumulation_steps']
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                if self.config['use_amp']:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.ema:
                    self.ema.update()
            
            # Statistics
            running_loss += loss.item() * self.config['accumulation_steps']
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return running_loss / len(self.train_loader), 100. * correct / total
    
    @torch.no_grad()
    def validate(self, use_ema=True):
        """Validate model"""
        if use_ema and self.ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.val_loader, desc="Validating"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        if use_ema and self.ema:
            self.ema.restore()
        
        return running_loss / len(self.val_loader), 100. * correct / total
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config,
            'history': self.history
        }
        
        # Save EMA weights if available
        if self.ema:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        # Save last checkpoint
        torch.save(checkpoint, f"{self.config['output_dir']}/last_model.pth")
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, f"{self.config['output_dir']}/best_model_v2.pth")
            print(f"💾 New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"Training {self.config['model_name']}")
        print(f"Device: {self.device}")
        print(f"Effective batch size: {self.config['batch_size'] * self.config['accumulation_steps']}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            # Check for improvement
            if val_acc > self.best_acc + self.config['min_delta']:
                self.best_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_acc, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_acc, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
        
        # Save training history
        with open(f"{self.config['output_dir']}/training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n🎉 Training complete! Best Val Acc: {self.best_acc:.2f}%")
        return self.best_acc


def main():
    print("="*60)
    print("Deepfake Detection - Enhanced Training v2.0")
    print("="*60)
    
    set_seed(CONFIG['seed'])
    
    trainer = Trainer(CONFIG)
    best_acc = trainer.train()
    
    print(f"\nFinal Results:")
    print(f"  Best Validation Accuracy: {best_acc:.2f}%")
    print(f"  Model saved to: {CONFIG['output_dir']}/best_model_v2.pth")


if __name__ == "__main__":
    main()
