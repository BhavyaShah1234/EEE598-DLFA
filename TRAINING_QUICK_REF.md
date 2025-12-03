# Quick Reference: Modified LISA Training Script

## New Command-Line Arguments

### LoRA Configuration
```bash
--use_lora                    # Enable LoRA fine-tuning
--lora_r 8                    # LoRA rank (default: 8)
--lora_alpha 16               # LoRA alpha (default: 16)
--lora_dropout 0.05           # LoRA dropout (default: 0.05)
```

### Loss Weights (LISA-Style)
```bash
--ce_weight 1.0               # Text generation loss weight (default: 1.0)
--bce_weight 2.0              # Mask BCE loss weight (default: 2.0)
--dice_weight 0.5             # Mask Dice loss weight (default: 0.5)
```

### Learning Rate Scheduler
```bash
--scheduler cosine            # Cosine annealing (default)
--scheduler linear            # Linear warmup + decay
--scheduler plateau           # ReduceLROnPlateau
--scheduler none              # No scheduler

# Scheduler-specific parameters
--warmup_steps 100            # For linear scheduler (default: 100)
--plateau_patience 3          # For plateau scheduler (default: 3)
--plateau_factor 0.5          # For plateau scheduler (default: 0.5)
```

### Early Stopping & Model Saving
```bash
--early_stopping_patience 5   # Stop after N epochs without improvement (default: 5)
--save_best_only              # Only save best checkpoint (default: True)
```

## Example Commands

### 1. Quick Test (Small Dataset)
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch scripts/train_new2.py \
    --epochs 2 \
    --max_train_samples 10 \
    --max_val_samples 5
```

### 2. Training with LoRA (Recommended for 8GB GPU)
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch scripts/train_new2.py \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --scheduler cosine \
    --early_stopping_patience 5 \
    --epochs 20 \
    --batch_size 2 \
    --num_workers 4
```

### 3. Full Training (LISA Paper Settings)
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch scripts/train_new2.py \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --scheduler cosine \
    --ce_weight 1.0 \
    --bce_weight 2.0 \
    --dice_weight 0.5 \
    --lr 3e-4 \
    --epochs 20 \
    --batch_size 1 \
    --early_stopping_patience 5 \
    --save_best_only
```

### 4. Aggressive Learning with Plateau Scheduler
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch scripts/train_new2.py \
    --use_lora \
    --scheduler plateau \
    --plateau_patience 3 \
    --plateau_factor 0.5 \
    --lr 1e-3 \
    --epochs 50 \
    --early_stopping_patience 10
```

### 5. Linear Warmup Schedule (For Large Datasets)
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch scripts/train_new2.py \
    --use_lora \
    --scheduler linear \
    --warmup_steps 500 \
    --lr 5e-4 \
    --epochs 30
```

## Key Changes from Original

### Losses
- **Old**: BCE + Dice + IoU
- **New**: CE (text) + BCE (mask) + Dice (mask)
  - Matches LISA paper exactly

### Metrics
- **Old**: mIoU, cIoU (same as mIoU)
- **New**: cIoU, gIoU
  - cIoU: Cumulative IoU (primary metric)
  - gIoU: Generalized IoU (considers enclosing box)

### Training Features
- ✅ LoRA support (90% fewer trainable params)
- ✅ Multiple LR schedulers
- ✅ Early stopping
- ✅ Best-only model saving
- ✅ Training visualization (plots)
- ✅ Text generation loss

## Output Files

### During Training
```
outputs/
├── metrics_epoch_1.json      # Per-epoch metrics
├── metrics_epoch_2.json
├── ...
└── training_history.json     # Complete training log

checkpoints/
└── best_model_epoch_X.pt     # Best model only (if --save_best_only)
```

### After Training
```
outputs/
├── training_history.png      # NEW: Visualization plots
└── training_history.json     # Updated with benchmarks
```

## Training Output Format

### Console Output (Per Epoch)
```
Epoch 5 Results:
  Train Loss: 5.2073
    - CE Loss: 2.1234      # NEW: Text generation
    - BCE Loss: 2.0456     # Mask BCE
    - Dice Loss: 1.0383    # Mask Dice
  Train cIoU: 0.0537       # NEW: Cumulative IoU
  Train gIoU: 0.0502       # NEW: Generalized IoU
  Val Loss: 5.1133
  Val cIoU: 0.0587         # NEW: Primary validation metric
  Val gIoU: 0.0551
  ✓ Saved best model (cIoU: 0.0587)
```

### Early Stopping
```
Epoch 15 Results:
  ...
  No improvement for 5 epoch(s)

Early stopping triggered after 15 epochs
Best validation cIoU: 0.1234
```

## Metrics Explained

### cIoU (Cumulative IoU)
- Mean IoU across all validation samples
- Primary metric for model selection
- Higher is better
- Range: [0, 1]

### gIoU (Generalized IoU)
- Considers smallest enclosing box
- Penalizes scattered predictions
- Better for box-like segments
- Range: [-1, 1]

### CE Loss (Cross-Entropy)
- Measures text generation quality
- Ensures model learns to produce <SEG> token
- Lower is better

## LoRA Benefits

### Memory Usage
- **Without LoRA**: ~6.5 GB
- **With LoRA (r=8)**: ~4.5 GB
- **Reduction**: ~30% less memory

### Trainable Parameters
- **Without LoRA**: ~100M params
- **With LoRA (r=8)**: ~10M params
- **Reduction**: ~90% fewer params

### Training Speed
- **Faster**: Fewer gradients to compute
- **Stable**: Better gradient flow
- **Flexible**: Easy to switch LoRA adapters

## Troubleshooting

### OOM Error
```bash
# Reduce batch size
--batch_size 1

# Enable LoRA
--use_lora --lora_r 8

# Use gradient checkpointing (if available)
--gradient_checkpointing
```

### Training Not Converging
```bash
# Try different scheduler
--scheduler cosine

# Adjust loss weights
--ce_weight 0.5 --bce_weight 2.0 --dice_weight 0.5

# Reduce learning rate
--lr 1e-4
```

### Training Too Slow
```bash
# Increase batch size (if memory allows)
--batch_size 4

# Use more workers
--num_workers 8

# Reduce dataset
--max_train_samples 500
```

## Recommended Settings by GPU Memory

### 8GB GPU (RTX 3060, RTX 5060)
```bash
--use_lora --lora_r 8 --batch_size 1 --num_workers 2
```

### 12GB GPU (RTX 3080, RTX 4070)
```bash
--use_lora --lora_r 16 --batch_size 2 --num_workers 4
```

### 16GB+ GPU (RTX 4080, RTX 4090)
```bash
--use_lora --lora_r 32 --batch_size 4 --num_workers 8
# Or without LoRA:
--batch_size 2 --num_workers 8
```

## Monitoring Training

### Real-time Progress
- Watch console for epoch-by-epoch metrics
- Check `outputs/metrics_epoch_X.json` for detailed logs

### Post-training Analysis
- View `outputs/training_history.png` for visualizations
- Check `checkpoints/` for best model

### TensorBoard (Optional)
Add to script if needed:
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/lisa_training')
```

## Version Information

- **Script**: `train_new2.py`
- **Updates**: LISA paper alignment + LoRA + Schedulers + Plotting
- **Date**: December 2024
- **Compatibility**: Transformers 4.x, PEFT 0.7+, PyTorch 2.x
