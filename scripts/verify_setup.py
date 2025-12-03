#!/usr/bin/env python3
"""
Verification Script for Modified LISA Implementation
Checks that all components can be imported and basic functionality works
"""

import sys
from pathlib import Path

print("="*80)
print("Modified LISA - Component Verification")
print("="*80)

# Test 1: Basic imports
print("\n[1/7] Testing basic imports...")
try:
    import torch
    import transformers as tf
    from PIL import Image
    import numpy as np
    import cv2
    from accelerate import Accelerator
    print("✓ All basic imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: CUDA availability
print("\n[2/7] Checking CUDA...")
if torch.cuda.is_available():
    print(f"✓ CUDA available")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠ CUDA not available (will use CPU - very slow)")

# Test 3: Dataset structure
print("\n[3/7] Checking dataset structure...")
data_dir = Path('.')
train_dir = data_dir / 'train'
val_dir = data_dir / 'val'
test_dir = data_dir / 'test'

if not train_dir.exists():
    print(f"✗ Train directory not found: {train_dir}")
    sys.exit(1)

train_jsons = list(train_dir.glob('*.json'))
train_images = list(train_dir.glob('*.jpg'))

print(f"✓ Train directory found")
print(f"  JSON files: {len(train_jsons)}")
print(f"  Image files: {len(train_images)}")

if val_dir.exists():
    val_jsons = list(val_dir.glob('*.json'))
    print(f"✓ Val directory found ({len(val_jsons)} samples)")
else:
    print("⚠ Val directory not found")

# Test 4: Import main script components
print("\n[4/7] Testing script imports...")
try:
    sys.path.insert(0, 'scripts')
    from train_new2 import (
        ReasonSegDataset,
        ModifiedLISA,
        CombinedSegmentationLoss,
        DiceLoss,
        IoULoss,
        compute_iou
    )
    print("✓ All classes imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 5: Dataset loading
print("\n[5/7] Testing dataset loading...")
try:
    dataset = ReasonSegDataset('.', split='train', max_samples=3)
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Test __getitem__
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Image size: {sample['image'].size}")
    print(f"  Text: {sample['text'][:50]}...")
    print(f"  Mask shape: {sample['mask'].shape}")
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")
    sys.exit(1)

# Test 6: Loss functions
print("\n[6/7] Testing loss functions...")
try:
    criterion = CombinedSegmentationLoss()
    
    # Create dummy tensors
    pred = torch.randn(2, 256, 256)
    target = torch.randint(0, 2, (2, 256, 256)).float()
    
    loss_dict = criterion(pred, target)
    print(f"✓ Loss computed successfully")
    print(f"  Total: {loss_dict['total']:.4f}")
    print(f"  BCE: {loss_dict['bce']:.4f}")
    print(f"  Dice: {loss_dict['dice']:.4f}")
    print(f"  IoU: {loss_dict['iou']:.4f}")
except Exception as e:
    print(f"✗ Loss computation failed: {e}")
    sys.exit(1)

# Test 7: Metrics
print("\n[7/7] Testing metrics...")
try:
    pred = torch.rand(256, 256)
    target = torch.randint(0, 2, (256, 256)).float()
    
    iou = compute_iou(pred, target)
    print(f"✓ Metrics computed successfully")
    print(f"  IoU: {iou:.4f}")
except Exception as e:
    print(f"✗ Metrics computation failed: {e}")
    sys.exit(1)

# Final summary
print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\n✓ All components verified successfully!")
print("\nYou can now run training with:")
print("  accelerate launch scripts/train_new2.py --max_train_samples 5 --epochs 2")
print("\nFor full training:")
print("  accelerate launch scripts/train_new2.py --epochs 20 --batch_size 1")
print("="*80)
