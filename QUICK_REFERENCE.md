# ModifiedLISA Training - Quick Reference

## Start Training (3 Options)

### Option 1: Bash Script (Easiest)
```bash
./run_training.sh
```

### Option 2: Accelerate (Multi-GPU)
```bash
accelerate launch scripts/train_new.py
```

### Option 3: Single GPU
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_new.py
```

## Check Results

### View Metrics
```bash
python scripts/analyze_metrics.py
```

### Check Outputs
```bash
ls outputs/           # Metrics JSON files
ls checkpoints/       # Saved models
```

## Key Metrics Tracked

| Metric | Description | Better |
|--------|-------------|--------|
| **IoU** | Intersection over Union | Higher ↑ |
| **gIoU** | Generalized IoU | Higher ↑ |
| **cIoU** | Complete IoU | Higher ↑ |
| **Iter/s** | Iterations per second | Higher ↑ |
| **Mem** | Peak GPU memory | Lower ↓ |
| **Loss** | Training loss | Lower ↓ |

## Modify Settings

Edit `scripts/train_new.py` → `TrainingConfig` class:

```python
class TrainingConfig:
    num_epochs: int = 10          # ← Change this
    batch_size: int = 2           # ← Change this
    learning_rate: float = 1e-4   # ← Change this
    # ... more options
```

## Common Adjustments

### Out of Memory?
```python
batch_size = 1
use_quantization = True
img_size = 192
```

### Slow Training?
```python
use_mixed_precision = True
batch_size = 4  # if memory allows
```

### Need Better Results?
```python
num_epochs = 20
lora_r = 32
learning_rate = 2e-4
```

## Files Created

| File | Purpose |
|------|---------|
| `scripts/train_new.py` | Main training script |
| `scripts/analyze_metrics.py` | Results analysis |
| `run_training.sh` | Easy launcher |
| `TRAINING_GUIDE.md` | Full documentation |
| `README_TRAINING.md` | Quick start |
| `IMPLEMENTATION_SUMMARY.md` | This implementation |

## Expected Output

```
Epoch 1/10
Step 0/500 | Loss: 0.45 | IoU: 0.62 | gIoU: 0.59 | Iter/s: 2.3 | Mem: 12.4GB
...
Training Metrics:
  Loss: 0.41
  IoU: 0.65
  gIoU: 0.62
  Iterations/sec: 2.35
  Peak memory: 12.51GB

Validation Metrics:
  Loss: 0.40
  IoU: 0.67
  gIoU: 0.63

✓ Checkpoint saved to checkpoints/best_model_epoch_1.pt
✓ Metrics saved to outputs/metrics_epoch_1.json
```

## Compare with LISA

All LISA metrics are tracked:
- ✓ Segmentation: IoU, gIoU, cIoU
- ✓ Performance: Speed, Memory
- ✓ Training: Loss progression

## Need Help?

1. Check `TRAINING_GUIDE.md` for details
2. View terminal output for errors
3. Run `nvidia-smi` to check GPU
4. Reduce batch_size if OOM

## That's It!

Run `./run_training.sh` to start training now! 🚀
