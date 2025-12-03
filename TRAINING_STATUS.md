# LISA Training Status Report

## Current Training Run

**Started:** Current Session
**Command:** `accelerate launch scripts/train_new2.py --save_best_only --use_lora --epochs 30`
**Status:** 🟢 RUNNING

### Configuration
- **Model:** LLaVA-1.5-7B (4-bit) + SAM-ViT-Base (4-bit)
- **LoRA:** Enabled (rank=8, alpha=16, targets=['q_proj', 'v_proj'])
- **Dataset:** ReasonSeg (expanded)
  - Train: 1326 samples (was 239) - **5.5x increase**
  - Val: 344 samples (was 200) - **1.7x increase**
- **Epochs:** 30
- **Batch Size:** 1
- **Learning Rate:** 1e-4
- **Scheduler:** CosineAnnealingLR
- **Early Stopping:** patience=10

### All Fixes Applied ✅

1. **LoRA OOM Issue** - FIXED
   - Reduced target modules to 2 (q_proj, v_proj)
   - Added gradient checkpointing
   - Fixed in-place operation with .clone()
   - Memory: 5.79 GB (within 8GB limit)

2. **Negative gIoU Metric** - FIXED
   - Corrected bounding box area calculation
   - Removed unnecessary mask creation
   - Simplified formula

3. **Dataset Underutilization** - FIXED
   - Now creates one sample per text query
   - Dataset expanded 5-6x
   - Each image contributes 4-5 training samples

### Expected Timeline

**Per Epoch:**
- Training: 1326 samples × 1.8s = ~40 minutes
- Validation: 344 samples × 0.5s = ~3 minutes
- Total: ~43 minutes per epoch

**Full 30 Epochs:**
- Estimated: 30 × 43min = **~21.5 hours**
- With early stopping: Could finish in 10-20 epochs if converges

### Monitoring Commands

**Watch live progress:**
```bash
tail -f training_log.txt
```

**Quick status check:**
```bash
./monitor_progress.sh
```

**View latest metrics:**
```bash
tail -n 20 training_log.txt | grep -E "(Epoch|Loss|cIoU|gIoU)"
```

**Check GPU memory:**
```bash
nvidia-smi
```

**Stop training:**
```bash
pkill -f train_new2.py
```

### Success Criteria

Training is successful if after 20-30 epochs:

1. **Loss Decreases:**
   - Train loss should steadily decrease
   - Val loss should decrease or plateau (not increase significantly)

2. **cIoU Improves:**
   - Should increase from ~0.01-0.04 toward 0.3-0.5+
   - This is the primary metric to watch

3. **gIoU Improves:**
   - Should increase from negative values toward 0 or positive
   - Less critical than cIoU

4. **No Overfitting:**
   - Val loss shouldn't diverge far from train loss
   - Val cIoU should track reasonably with train cIoU

### What to Do If Metrics Don't Improve

If after 20-50 epochs there's no improvement:

1. **Simplify Loss Function:**
   - Remove Dice loss component
   - Keep only CE + BCE
   - Adjust weights

2. **Adjust Hyperparameters:**
   - Try learning rate 5e-5 or 2e-4
   - Try different scheduler (linear, plateau)
   - Increase LoRA rank to 16

3. **Check Data Quality:**
   - Verify masks are correct
   - Check text queries are meaningful
   - Visualize predictions

4. **Model Issues:**
   - Try training without quantization (if memory allows)
   - Try larger LoRA (more target modules)
   - Check if VLM weights are actually updating

### Output Files

During training, these files are generated:

**Checkpoints:**
- `checkpoints/best_model_epoch_N.pt` - Best models by validation cIoU

**Metrics:**
- `outputs/training_history.json` - Full history
- `outputs/metrics_epoch_N.json` - Per-epoch metrics
- `outputs/training_history.png` - Visualization plots

**Logs:**
- `training_log.txt` - Complete training log

### Hardware Status

**GPU:** RTX 5060 (8GB)
- With LoRA: 5.79 GB peak (✅ Safe)
- Without LoRA: 5.00 GB peak
- Margin: ~2.2 GB free

**Training Speed:**
- ~1.8 seconds per sample
- ~0.5 seconds per validation sample

---

## Historical Context

### Previous Training Attempts

**Attempt 1: With Full LoRA (4 targets)**
- Status: ❌ FAILED - OOM
- Error: CUDA out of memory at 7.44 GB
- Issue: Too many target modules

**Attempt 2: Without LoRA**
- Status: ✅ SUCCESS
- Memory: 5.00 GB
- Issue: Metrics showed negative gIoU, dataset only used first text

**Attempt 3: Current (LoRA with 2 targets + all fixes)**
- Status: 🟢 RUNNING
- Memory: 5.79 GB
- All issues fixed

---

## Quick Reference

### Training Commands

**Start 30-epoch training:**
```bash
accelerate launch scripts/train_new2.py --save_best_only --use_lora --epochs 30
```

**Quick test (1 epoch, 10 samples):**
```bash
accelerate launch scripts/train_new2.py \
  --use_lora --epochs 1 \
  --max_train_samples 10 --max_val_samples 5 \
  --save_best_only
```

**Resume from checkpoint:**
```bash
# Edit script to load from checkpoint, then:
accelerate launch scripts/train_new2.py --save_best_only --use_lora --epochs 30
```

### Analysis Commands

**View training curves:**
```python
import json
import matplotlib.pyplot as plt

with open('outputs/training_history.json') as f:
    history = json.load(f)

epochs = range(1, len(history['train_loss']) + 1)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs, history['train_loss'], label='Train')
plt.plot(epochs, history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, history['train_ciou'], label='Train')
plt.plot(epochs, history['val_ciou'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('cIoU')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, history['train_giou'], label='Train')
plt.plot(epochs, history['val_giou'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('gIoU')
plt.legend()

plt.tight_layout()
plt.show()
```

**Check best checkpoint:**
```bash
ls -lth checkpoints/ | head -5
```

**View recent metrics:**
```bash
cat outputs/metrics_epoch_$(ls outputs/metrics_epoch_*.json | wc -l).json | jq '.'
```

---

## Next Steps After Training

1. **Evaluate Results:**
   - Check `outputs/training_history.png`
   - Review final metrics in `outputs/training_history.json`
   - Identify best epoch from validation cIoU

2. **Test Best Model:**
   - Load best checkpoint
   - Run inference on test set
   - Visualize predictions

3. **If Results Are Good:**
   - Document final performance
   - Save best model for deployment
   - Create inference script

4. **If Results Need Improvement:**
   - Follow "What to Do If Metrics Don't Improve" above
   - Adjust hyperparameters
   - Try alternative loss functions
   - Consider model architecture changes

---

Last Updated: Current session
Training Status: 🟢 RUNNING (Epoch 1/30 in progress)
