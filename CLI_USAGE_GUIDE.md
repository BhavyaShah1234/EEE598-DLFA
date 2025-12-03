# ModifiedLISA Training Script - CLI Usage Guide

## Overview

The `scripts/train_new.py` script has been updated with the following new features:
- ✅ **CLI Arguments**: All hyperparameters configurable via command line
- ✅ **Early Stopping**: Automatically stops training when validation IoU plateaus
- ✅ **Learning Rate Scheduler**: Reduces LR when validation metrics plateau
- ✅ **Test Inference**: Runs inference on test dataset after training or independently

## Quick Start

### Basic Training with Default Settings
```bash
accelerate launch scripts/train_new.py --use_lora
```

### Training with Custom Parameters
```bash
accelerate launch scripts/train_new.py \
  --use_lora \
  --num_epochs 10 \
  --learning_rate 5e-5 \
  --batch_size 2 \
  --max_train_samples 100 \
  --max_val_samples 50 \
  --early_stopping_patience 5 \
  --lr_scheduler_patience 3 \
  --run_test
```

### Test-Only Mode (No Training)
```bash
accelerate launch scripts/train_new.py \
  --test_only \
  --test_checkpoint checkpoints/best_model_epoch_1.pt \
  --use_lora \
  --max_test_samples 100
```

## Command Line Arguments

### Data Paths
- `--data_root` - Root directory (default: `/home/bhavya-shah/Projects/EEE598-DLFA`)
- `--train_json` - Training JSON filename (default: `train.json`)
- `--train_dir` - Training images directory (default: `train`)
- `--val_dir` - Validation images directory (default: `val`)
- `--test_dir` - Test images directory (default: `test`)
- `--output_dir` - Output directory for metrics (default: `outputs`)
- `--checkpoint_dir` - Checkpoint directory (default: `checkpoints`)

### Training Hyperparameters
- `--num_epochs` - Number of epochs (default: `100`)
- `--batch_size` - Batch size per GPU (default: `1`)
- `--learning_rate` - Initial learning rate (default: `1e-4`)
- `--weight_decay` - Weight decay (default: `0.01`)
- `--gradient_accumulation_steps` - Gradient accumulation (default: `8`)
- `--max_grad_norm` - Max gradient norm for clipping (default: `1.0`)

### Model Configuration
- `--img_size` - Input image size (default: `256`)
- `--max_text_length` - Maximum text length (default: `128`)
- `--max_train_samples` - Limit training samples (default: `None` = all)
- `--max_val_samples` - Limit validation samples (default: `None` = all)
- `--max_test_samples` - Limit test samples (default: `None` = all)

### LoRA Configuration
- `--lora_r` - LoRA rank (default: `16`)
- `--lora_alpha` - LoRA alpha (default: `32`)
- `--lora_dropout` - LoRA dropout (default: `0.1`)

### Model Selection
- `--image_encoder_model_name` - Image encoder (default: `openai/clip-vit-base-patch16`)
- `--text_encoder_model_name` - Text encoder (default: `meta-llama/Llama-3.2-1B`)
- `--sam_model_name` - SAM model (default: `facebook/sam-vit-base`)

### Training Modes
- `--use_lora` - Enable LoRA adapters (flag)
- `--use_mixed_precision` - Enable mixed precision (flag)
- `--mixed_precision` - Precision mode: `no`, `fp16`, `bf16` (default: `no`)
- `--use_quantization` - Enable quantization (flag)
- `--quantization` - Quantization mode: `4bit`, `8bit` (default: `8bit`)
- `--freeze_image_encoder` - Freeze image encoder (flag)
- `--freeze_text_encoder` - Freeze text encoder (flag)
- `--freeze_sam` - Freeze SAM (flag)

### Early Stopping & LR Scheduler (NEW!)
- `--early_stopping_patience` - Epochs to wait before stopping (default: `10`)
- `--lr_scheduler_patience` - Epochs to wait before reducing LR (default: `5`)
- `--lr_scheduler_factor` - LR reduction factor (default: `0.5`)

### Test Inference (NEW!)
- `--run_test` - Run test inference after training (flag)
- `--test_only` - Only run test inference, skip training (flag)
- `--test_checkpoint` - Checkpoint path for test inference

## Features Explanation

### 1. Early Stopping
Monitors validation IoU and stops training if no improvement for N epochs:
- Saves best checkpoint automatically
- Prevents overfitting
- Reduces unnecessary computation
- Counter displayed during training

Example output:
```
Early stopping counter: 2/10
```

### 2. Learning Rate Scheduler
ReduceLROnPlateau reduces LR when validation IoU plateaus:
- Monitors validation IoU
- Reduces LR by `lr_scheduler_factor` after `lr_scheduler_patience` epochs
- Helps fine-tune the model

Example output:
```
Epoch 5/100
Current LR: 5.00e-05  # Reduced from 1e-04
```

### 3. Test Inference
Runs comprehensive evaluation on test dataset:
- Loads best checkpoint automatically (or specified checkpoint)
- Computes IoU, gIoU, cIoU metrics
- Saves results to `outputs/test_results.json`
- Can run independently with `--test_only`

### 4. CLI Arguments
All hyperparameters now configurable via command line:
- No need to modify code
- Easy experimentation
- Better reproducibility
- Arguments saved in checkpoint

## Example Workflows

### 1. Quick Test Run (3 epochs, limited samples)
```bash
accelerate launch scripts/train_new.py \
  --use_lora \
  --num_epochs 3 \
  --max_train_samples 30 \
  --max_val_samples 15 \
  --early_stopping_patience 3
```

### 2. Full Training with Test
```bash
accelerate launch scripts/train_new.py \
  --use_lora \
  --num_epochs 50 \
  --early_stopping_patience 10 \
  --lr_scheduler_patience 5 \
  --run_test
```

### 3. Mixed Precision Training (BF16)
```bash
accelerate launch scripts/train_new.py \
  --use_lora \
  --use_mixed_precision \
  --mixed_precision bf16 \
  --num_epochs 20
```

### 4. Test Existing Checkpoint
```bash
accelerate launch scripts/train_new.py \
  --test_only \
  --test_checkpoint checkpoints/best_model_epoch_5.pt \
  --use_lora
```

### 5. Custom Learning Rate Schedule
```bash
accelerate launch scripts/train_new.py \
  --use_lora \
  --learning_rate 5e-5 \
  --lr_scheduler_patience 3 \
  --lr_scheduler_factor 0.3 \
  --early_stopping_patience 7
```

## Output Files

### Training Outputs
- `outputs/metrics_epoch_{N}.json` - Per-epoch metrics including:
  - Training metrics (loss, IoU, gIoU, cIoU, speed, memory)
  - Validation metrics
  - Current learning rate

### Checkpoints
- `checkpoints/best_model_epoch_{N}.pt` - Best model saved containing:
  - Model state dict
  - Optimizer state dict
  - Validation IoU
  - Training/validation metrics
  - CLI arguments (for reproducibility)

### Test Results
- `outputs/test_results.json` - Test inference results:
  - Loss, IoU, gIoU, cIoU
  - Total inference time
  - Average iteration time
  - Peak memory usage

## Training Output Example

```
============================================================
Epoch 1/10
Current LR: 1.00e-04
============================================================
Step 0/50 | Loss: 3.5124 | IoU: 0.0234 | gIoU: -0.5234 | cIoU: -0.6123 | Iter/s: 4.32 | Mem: 6.50GB
Step 5/50 | Loss: 2.8456 | IoU: 0.0567 | gIoU: -0.3456 | cIoU: -0.4234 | Iter/s: 4.45 | Mem: 6.64GB
...

Training Metrics:
  Loss: 2.1234
  IoU: 0.1456
  gIoU: -0.2345
  cIoU: -0.3123
  Iterations/sec: 4.52
  Epoch time: 45.23s
  Peak memory: 6.84GB

Validation Metrics:
  Loss: 2.3456
  IoU: 0.1234
  gIoU: -0.2567
  cIoU: -0.3345

🎯 New best validation IoU: 0.1234
Checkpoint saved to checkpoints/best_model_epoch_1.pt
```

## Notes

1. **GPU Memory**: Default settings optimized for 8GB GPU
   - `batch_size=1` with `gradient_accumulation_steps=8`
   - Reduce `img_size` if OOM occurs

2. **LoRA + Mixed Precision**: 
   - FP16 may cause gradient scaling errors with LoRA
   - Use BF16 or disable mixed precision with LoRA

3. **Early Stopping**:
   - Set patience high enough to allow learning
   - Typical values: 5-15 epochs depending on dataset size

4. **Learning Rate Scheduler**:
   - Patience should be less than early stopping patience
   - Factor of 0.5 halves LR, 0.1 reduces to 10%

5. **Test Inference**:
   - Automatically uses best checkpoint when `--run_test` is used
   - Requires `--test_checkpoint` when using `--test_only`
