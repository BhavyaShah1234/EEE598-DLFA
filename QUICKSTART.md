# Modified LISA - Quick Start Guide

## Installation
```bash
pip install torch transformers accelerate pillow opencv-python numpy tqdm bitsandbytes
```

## Quick Test (2 minutes)
```bash
cd /home/bhavya-shah/Projects/EEE598-DLFA
accelerate launch scripts/train_new2.py \
    --epochs 2 \
    --batch_size 1 \
    --max_train_samples 5 \
    --max_val_samples 3
```

## Full Training (8GB GPU)
```bash
accelerate launch scripts/train_new2.py \
    --epochs 20 \
    --batch_size 1 \
    --lr 3e-4 \
    --dtype bf16 \
    --use_vlm_vision
```

## Monitor Training
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# View latest metrics
cat outputs/training_history.json | jq '.train[-1], .val[-1]'

# Monitor training log
tail -f nohup.out  # if running in background
```

## Common Issues

### Out of Memory
```bash
# Use quantization (enabled by default)
# OR reduce samples
--max_train_samples 100

# OR clear cache first
python3 -c "import torch; torch.cuda.empty_cache()"
```

### Slow Model Loading
```bash
# Normal for first run (2-5 min)
# Models cached in ~/.cache/huggingface/
```

## Output Files

```
outputs/
  ├── metrics_epoch_1.json      # Per-epoch metrics
  ├── metrics_epoch_2.json
  └── training_history.json     # Complete history + benchmarks

checkpoints/
  ├── best_model_epoch_5.pt     # Best validation mIoU
  └── best_model_epoch_8.pt
```

## Key Hyperparameters

| Parameter | Default | For 8GB GPU | For 24GB+ GPU |
|-----------|---------|-------------|---------------|
| Batch Size | 1 | 1 | 2-4 |
| Learning Rate | 3e-4 | 3e-4 | 1e-4 to 5e-4 |
| Dtype | bf16 | bf16 | bf16 or fp32 |
| Quantization | On | On | Off (optional) |

## Expected Performance

| Metric | Value |
|--------|-------|
| Training Time | ~5-10 min/epoch (small dataset) |
| Peak Memory | ~7-8 GB (with quantization) |
| Inference Time | ~0.2-0.3 s/image |
| Validation mIoU | 0.6-0.8 (dataset dependent) |

## Help

```bash
python3 scripts/train_new2.py --help
```

## Documentation

- Full guide: `MODIFIED_LISA_README.md`
- Implementation details: `MODIFIED_LISA_IMPLEMENTATION.md`
