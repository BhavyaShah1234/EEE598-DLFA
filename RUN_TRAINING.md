# How to Run Training - Quick Guide

## ✅ Working Configurations (8GB GPU)

### 1. **Recommended: Full Training with LoRA** (5.26 GB memory)
```bash
accelerate launch scripts/train_new2.py \
  --use_lora \
  --epochs 30
```

**What this trains:**
- VLM LLM: LoRA adapters (q_proj, v_proj)
- VLM Projector: Full fine-tuning
- SAM Mask Decoder: Full fine-tuning
- All Projection Layers: Full fine-tuning

**Trainable parameters:** ~143M / 3.7B (3.84%)

---

### 2. **Quick Test Run** (verify everything works)
```bash
accelerate launch scripts/train_new2.py \
  --use_lora \
  --epochs 1 \
  --max_train_samples 10 \
  --max_val_samples 5
```

---

### 3. **Resume from Checkpoint**
```bash
# Training auto-saves best model to checkpoints/best_model_epoch_X.pt
# To resume, copy checkpoint and modify script or manually load weights
```

---

## 📊 Metrics Tracked

### Loss Metrics:
- Total Loss (BCE + Dice weighted sum)
- BCE Loss (Binary Cross-Entropy)
- Dice Loss (Dice coefficient)

### IoU Metrics:
- **cIoU** (Cumulative IoU) - Primary metric for model selection
- **gIoU** (Generalized IoU) - Penalizes non-overlapping areas

### Pixel-wise Classification Metrics:
- **Accuracy**: Overall pixel correctness
- **Precision**: TP / (TP + FP) - How many predicted positives are correct
- **Recall**: TP / (TP + FN) - How many actual positives are found
- **F1-Score**: Harmonic mean of Precision and Recall
- **AUROC**: Area under ROC curve - Model's ranking ability

---

## 📈 Outputs Generated

### JSON Files (per epoch):
```
outputs/metrics_epoch_1.json
outputs/metrics_epoch_2.json
...
outputs/training_history.json  # Complete history
```

### Checkpoints:
```
checkpoints/best_model_epoch_X.pt  # Best model by validation cIoU
```

### Visualizations:
```
outputs/training_history.png  # 3×3 grid of all metrics
```

**Plot layout:**
- Row 1: Total Loss, BCE Loss, Dice Loss
- Row 2: cIoU+gIoU, Accuracy, F1-Score  
- Row 3: Precision, Recall, AUROC

---

## ⚙️ Common Arguments

### Dataset:
```bash
--data_dir .                    # Root directory with train/val/test folders
--max_train_samples 100         # Limit training samples (for testing)
--max_val_samples 50            # Limit validation samples
```

### Training:
```bash
--epochs 30                     # Number of epochs
--batch_size 1                  # Batch size (increase if memory allows)
--lr 1e-4                       # Learning rate
--weight_decay 0.01             # AdamW weight decay
```

### Loss Weights:
```bash
--bce_weight 2.0                # BCE loss weight
--dice_weight 0.5               # Dice loss weight
```

### LoRA:
```bash
--use_lora                      # Enable LoRA
--lora_r 8                      # LoRA rank
--lora_alpha 16                 # LoRA alpha
--lora_dropout 0.05             # LoRA dropout
```

### Scheduler:
```bash
--scheduler cosine              # cosine, linear, plateau, none
--early_stopping_patience 10    # Epochs without improvement before stopping
```

### Component Training Control:
```bash
--train_vlm_vision              # Train VLM vision encoder (ignored if quantized)
--train_vlm_projector           # Train VLM projector (default: True)
--train_vlm_llm                 # Train VLM LLM (default: True)
--train_sam_vision              # Train SAM vision encoder (ignored if quantized)
--train_sam_prompt_encoder      # Train SAM prompt encoder (ignored if quantized)
--train_sam_mask_decoder        # Train SAM mask decoder (default: True)
--train_projection_layers       # Train projection layers (default: True)
```

**Note:** Component flags only work for non-quantized models. With quantization (default), only LoRA and specific components can be trained.

---

## 🔍 Monitor Training

### Watch live progress:
```bash
tail -f outputs/training_history.json
```

### Check latest metrics:
```bash
cat outputs/metrics_epoch_*.json | jq .
```

### View latest plot:
```bash
xdg-open outputs/training_history.png  # Linux
open outputs/training_history.png      # macOS
```

---

## ⚠️ Troubleshooting

### Out of Memory Error:
**Problem:** `torch.OutOfMemoryError: CUDA out of memory`

**Solutions:**
1. ✅ Use LoRA: `--use_lora`
2. Reduce batch size: `--batch_size 1` (already default)
3. Use gradient accumulation (not implemented yet)
4. Reduce max samples during testing

### Slow Training:
- Expected: ~1.8s per batch with LoRA on RTX 5060
- If slower, check GPU utilization: `nvidia-smi`

### Low Metrics Initially:
- Normal! Model starts with random weights
- cIoU < 0.1 in early epochs is expected
- Watch for steady improvement over 10-20 epochs

---

## 📝 Example Training Session

```bash
# 1. Start training
accelerate launch scripts/train_new2.py --use_lora --epochs 30

# 2. Monitor in another terminal
watch -n 10 'cat outputs/training_history.json | jq ".train[-1], .val[-1]"'

# 3. After training completes, check best model
ls -lh checkpoints/best_model_epoch_*.pt

# 4. View metrics plot
xdg-open outputs/training_history.png
```

---

## 🎯 Expected Results

### After 1 Epoch:
- Train cIoU: 0.05-0.10
- Val cIoU: 0.02-0.05
- Memory: ~5.3 GB peak

### After 10 Epochs:
- Train cIoU: 0.20-0.40
- Val cIoU: 0.15-0.30
- Metrics should show steady improvement

### After 30 Epochs:
- Train cIoU: 0.40-0.60
- Val cIoU: 0.30-0.50
- Model should converge, early stopping may trigger

**Note:** These are rough estimates. Actual performance depends on dataset quality and hyperparameters.

---

## 🚀 Next Steps After Training

1. **Evaluate on test set** (implement inference script)
2. **Visualize predictions** (overlay masks on images)
3. **Analyze failure cases** (low IoU samples)
4. **Hyperparameter tuning** (learning rate, loss weights)
5. **Experiment with architectures** (different SAM/VLM models)

