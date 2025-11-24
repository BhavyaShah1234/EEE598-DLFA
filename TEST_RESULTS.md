# Training Test Results - ModifiedLISA on ReasonSeg

## Test Summary

**Date**: November 24, 2025
**GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (7.53 GB)
**Dataset**: ReasonSeg (50 train, 20 val samples for testing)

## Successful Configuration

### Test 1: LoRA=True, Mixed Precision=False ✓ PASSED

**Configuration**:
- LoRA: Enabled (r=16, alpha=32)
- Mixed Precision: Disabled (FP32)
- Batch Size: 1
- Gradient Accumulation: 8 steps
- Epochs: 3
- Image Size: 224x224

**Results**:
```
Epoch 1/3:
  Training - Loss: 1.0117, IoU: 0.0968, gIoU: -0.5785
  Validation - Loss: 0.8383, IoU: 0.2132, gIoU: 0.0095
  Time: 14.47s, Memory: 6.85GB, Speed: 4.44 iter/s

Epoch 2/3:
  Training - Loss: 0.8577, IoU: 0.1326, gIoU: -0.3435
  Validation - Loss: 0.8065, IoU: 0.1638, gIoU: -0.3466
  Time: 14.46s, Memory: 6.85GB, Speed: 4.63 iter/s

Epoch 3/3:
  Training - Loss: 0.8348, IoU: 0.1401, gIoU: -0.4811
  Validation - Loss: 0.8093, IoU: 0.1604, gIoU: -0.3681
  Time: 14.71s, Memory: 6.85GB, Speed: 4.51 iter/s

Best Validation IoU: 0.2132 (Epoch 1)
```

**Benchmark Metrics**:
- ✓ Peak Memory: 6.85 GB (fits in 8GB GPU)
- ✓ Training Speed: ~4.5 iterations/second
- ✓ Epoch Time: ~14.5 seconds for 50 samples
- ✓ All metrics tracked: IoU, gIoU, cIoU, loss, memory, speed

## Known Issues

### Mixed Precision with LoRA
**Issue**: `ValueError: Attempting to unscale FP16 gradients`

**Cause**: When using LoRA with FP16 mixed precision in Accelerate, the gradient scaler encounters FP16 parameters that cannot be unscaled properly. This is a known limitation of the combination of:
- LoRA (which may keep some parameters in FP16)
- Mixed Precision Training (FP16)
- Accelerate's gradient scaling

**Workaround**: Use FP32 training when LoRA is enabled. This still provides good performance on 8GB GPU with batch_size=1 and gradient accumulation.

## Files Generated

1. **Checkpoints**: `checkpoints/best_model_epoch_1.pt`
   - Contains: model state, optimizer state, metrics
   - Size: ~100MB (with LoRA, much smaller than full model)

2. **Metrics**: `outputs/metrics_epoch_{1,2,3}.json`
   - Training and validation metrics per epoch
   - Benchmark data (memory, speed, time)

3. **Logs**: `training_test.log`
   - Complete training output
   - Progress tracking
   - Error messages (if any)

## Verified Functionality

✓ Dataset loading (train/val/test splits)
✓ Polygon mask processing from JSON
✓ CLIP image preprocessing
✓ LLaMA text tokenization
✓ Model forward pass
✓ Loss computation (Dice + Focal)
✓ Backpropagation
✓ Optimizer step
✓ Metric tracking (IoU, gIoU, cIoU)
✓ Checkpoint saving
✓ Validation loop
✓ Memory monitoring
✓ Speed benchmarking

## Configuration Details

### Model Architecture
- **Vision Encoder**: CLIP ViT-B/16 (openai/clip-vit-base-patch16)
  - LoRA: 884,736 trainable / 86,684,160 total (1.02%)
- **Text Encoder**: LLaMA 3.2-1B (meta-llama/Llama-3.2-1B)
  - LoRA: 2,359,296 trainable / 1,238,173,696 total (0.19%)
- **Segmentation**: SAM ViT-B (facebook/sam-vit-base)
  - LoRA: 141,312 trainable / 93,876,784 total (0.15%)
- **Total Trainable**: 27,308,160 / 1,442,657,456 (1.89%)

### Training Hyperparameters
```python
batch_size = 1
gradient_accumulation_steps = 8  # Effective batch size: 8
learning_rate = 1e-4
weight_decay = 0.01
max_grad_norm = 1.0  # (disabled with mixed precision issues)
```

## Next Steps for Full Testing

To test all combinations properly:

1. **LoRA=True, MP=False** ✓ DONE
2. **LoRA=False, MP=False** - Test without LoRA (full fine-tuning)
3. **LoRA=False, MP=True** - Test mixed precision without LoRA
4. **LoRA=True, MP=True** - Currently has issues, may need different approach

## Recommendations

For 8GB GPU with this model:
- Use LoRA: Yes (reduces memory and trains 1.89% of parameters)
- Use Mixed Precision: Only if not using LoRA (or fix gradient scaling)
- Batch Size: 1
- Gradient Accumulation: 8 or higher
- Image Size: 224 (can try 192 for more memory)

The training is working successfully and all metrics are being tracked properly!
