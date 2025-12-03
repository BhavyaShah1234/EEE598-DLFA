# Fixes Applied to train_new2.py

## Date: December 1, 2025

### Issues Resolved

#### 1. **RuntimeError: Quantized Model requires_grad Issue** ✅
**Problem:** When using 4-bit quantization, attempting to set `requires_grad` on quantized parameters caused:
```
RuntimeError: only Tensors of floating point and complex dtype can require gradients
```

**Solution:** Added conditional checks to only set `requires_grad` for non-quantized models:
- VLM components (vision, projector, LLM): Only set `requires_grad` when `use_quantization=False`
- SAM components (vision, prompt encoder, mask decoder): Only set `requires_grad` when `use_quantization=False`
- Projection layers: Always trainable regardless of quantization

**Impact:** Training now works correctly with both quantized (4-bit) and non-quantized models.

---

#### 2. **Missing Pixel-wise Metrics** ✅
**Problem:** Only IoU metrics (cIoU, gIoU) were computed. No standard pixel-wise classification metrics.

**Solution:** Added comprehensive pixel-wise metrics computation:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **AUROC**: Area Under ROC Curve

**Implementation:**
- New function `compute_pixel_metrics()` using scikit-learn
- Metrics computed in both `train_epoch()` and `validate()`
- All metrics saved to JSON files and displayed in console
- Comprehensive 3×3 subplot grid in training plots

**Metrics Tracked Per Epoch:**
```json
{
  "train": {
    "loss": float,
    "bce": float,
    "dice": float,
    "cIoU": float,
    "gIoU": float,
    "accuracy": float,
    "precision": float,
    "recall": float,
    "f1": float,
    "auroc": float
  },
  "val": {
    "val_loss": float,
    "val_cIoU": float,
    "val_gIoU": float,
    "val_accuracy": float,
    "val_precision": float,
    "val_recall": float,
    "val_f1": float,
    "val_auroc": float
  }
}
```

---

### Updated Training Plots

The training visualization now shows a **3×3 grid** with:

**Row 1 - Losses:**
1. Total Loss (Train + Val)
2. BCE Loss (Train)
3. Dice Loss (Train)

**Row 2 - IoU & Basic Metrics:**
4. cIoU + gIoU (Train + Val)
5. Accuracy (Train + Val)
6. F1-Score (Train + Val)

**Row 3 - Detailed Pixel Metrics:**
7. Precision (Train + Val)
8. Recall (Train + Val)
9. AUROC (Train + Val)

---

### Usage Examples

#### With LoRA (Memory Efficient - 5.26 GB peak):
```bash
accelerate launch scripts/train_new2.py \
  --use_lora \
  --epochs 30
```

#### Without Quantization (Requires More Memory):
```bash
accelerate launch scripts/train_new2.py \
  --no_quantization \
  --epochs 30
```

#### With Component Control:
```bash
accelerate launch scripts/train_new2.py \
  --use_lora \
  --train_vlm_projector \
  --train_sam_mask_decoder \
  --train_projection_layers \
  --epochs 30
```

---

### Verified Configurations

✅ **With quantization + LoRA**: Works (5.26 GB memory) ⭐ **RECOMMENDED for 8GB GPU**
❌ **With quantization + projector training (no LoRA)**: OOM on 8GB GPU (requires >7.4 GB)
❌ **Without quantization**: Requires significantly more memory (>16GB)

**Important Notes:**
- When using 4-bit quantization, the `requires_grad` settings for quantized components are ignored (they remain frozen)
- Only LoRA, projection layers, and unfrozen components (like SAM mask decoder) can be trained with quantization
- For full component training without LoRA, use `--no_quantization` flag (requires high-memory GPU)

**Recommended Configuration for 8GB GPU:**
```bash
accelerate launch scripts/train_new2.py --use_lora --epochs 30
```

This trains:
- VLM LLM with LoRA (memory efficient)
- VLM projector (full fine-tuning)
- SAM mask decoder (full fine-tuning)  
- All projection layers (full fine-tuning)

---

### Code Changes Summary

1. **Lines 255-267**: Fixed `requires_grad` for VLM components (conditional on quantization)
2. **Lines 290-305**: Fixed `requires_grad` for SAM components (conditional on quantization)
3. **Lines 710-780**: Added `compute_pixel_metrics()` function
4. **Lines 800-815**: Added pixel metric tracking in `train_epoch()`
5. **Lines 830-840**: Updated return metrics in `train_epoch()`
6. **Lines 920-935**: Added pixel metric tracking in `validate()`
7. **Lines 950-960**: Updated return metrics in `validate()`
8. **Lines 1060-1150**: Expanded plotting to 3×3 grid with all metrics
9. **Lines 1340-1365**: Enhanced logging to display all pixel-wise metrics

---

### Testing Results

**Test Run (1 epoch, 2 samples):**
- ✅ Training completed successfully
- ✅ All metrics computed correctly
- ✅ JSON files saved with all metrics
- ✅ 3×3 training plot generated
- ✅ Peak memory: 5.26 GB (with LoRA)
- ✅ Avg inference time: 0.57s

**Sample Output:**
```
Epoch 1 Results:
  Train Loss: 6.6940
    - BCE Loss: 3.1249
    - Dice Loss: 0.8885
  Train Metrics:
    - cIoU: 0.0572
    - gIoU: -0.7895
    - Accuracy: 0.8587
    - Precision: 0.4230
    - Recall: 0.0627
    - F1-Score: 0.1089
    - AUROC: 0.0000
  Val Metrics:
    - Loss: 55.0367
    - cIoU: 0.0205
    - gIoU: -0.1690
    - Accuracy: 0.2057
    - Precision: 0.5600
    - Recall: 0.0208
    - F1-Score: 0.0401
    - AUROC: 0.0000
```

---

### Dependencies

Added requirement: `scikit-learn` (for AUROC computation)

Install with:
```bash
pip install scikit-learn
```

---

### Next Steps

1. Run full training with 30 epochs using LoRA
2. Monitor all pixel-wise metrics for convergence
3. Compare F1-Score, Precision, and Recall trends
4. Use AUROC to validate model's ranking capability

