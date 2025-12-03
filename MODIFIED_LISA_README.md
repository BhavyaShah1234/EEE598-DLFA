# Modified LISA Model - Training Guide

## Overview

This implementation provides a **Modified LISA model** that shares image embeddings between the Vision-Language Model (VLM) and Segment Anything Model (SAM) for efficient reasoning segmentation. The key innovation is **reducing computational cost by eliminating redundant vision encoding**, while maintaining the mask-as-an-embedding paradigm.

### Key Features

1. **Shared Vision Encoding**: Single forward pass through either VLM's or SAM's vision encoder (configurable)
2. **4-bit Quantization**: Memory-efficient model loading for 8GB GPU RAM
3. **<SEG> Token Paradigm**: Vocabulary expansion with special segmentation token
4. **Original LISA Losses**: BCE + Dice + IoU loss combination
5. **Comprehensive Metrics**: mIoU, cIoU, gIoU tracking
6. **Memory & Compute Benchmarking**: Automatic performance tracking

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Modified LISA Architecture                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Image + Text Query with <SEG> token                 │
│           │                                                  │
│           ├──→ [SHARED VISION ENCODER] ←──┐                 │
│           │    (VLM or SAM vision tower)  │                 │
│           │                                │                 │
│           ├──→ VLM Path:                   │                 │
│           │    ├─ Projector                │                 │
│           │    ├─ LLM (with expanded vocab)│                 │
│           │    └─ Extract <SEG> embedding  │                 │
│           │                                │                 │
│           └──→ SAM Path:                   │                 │
│                ├─ Projection layer         │                 │
│                ├─ <SEG> → SAM embedding    │                 │
│                ├─ Mask Decoder             │                 │
│                └─ Output Mask              │                 │
│                                                              │
│  Output: Segmentation Mask + IoU Prediction                 │
└─────────────────────────────────────────────────────────────┘
```

### Differences from Original LISA

| Aspect | Original LISA | Modified LISA |
|--------|--------------|---------------|
| Vision Encoding | Separate for VLM & SAM | **Shared** (single pass) |
| Memory Usage | Higher | **Lower** (4-bit quantization) |
| Inference Speed | Standard | **Faster** (reduced forward passes) |
| Training Params | Full model | **Projection layers + embeddings only** |

## Installation

```bash
# Required packages
pip install torch transformers accelerate pillow opencv-python numpy tqdm bitsandbytes
```

## Dataset Structure

The ReasonSeg dataset should be organized as follows:

```
.
├── train/
│   ├── image_1.jpg
│   ├── image_1.json  # Contains text query and polygon annotations
│   ├── image_2.jpg
│   ├── image_2.json
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### JSON Format

```json
{
  "text": ["What does this image represent?"],
  "shapes": [
    {
      "label": "target",
      "shape_type": "polygon",
      "points": [[x1, y1], [x2, y2], ...]
    }
  ]
}
```

## Training

### Basic Training

```bash
accelerate launch scripts/train_new2.py \
    --epochs 20 \
    --batch_size 1 \
    --lr 3e-4 \
    --dtype bf16 \
    --use_vlm_vision
```

### Full Training Command with All Options

```bash
accelerate launch scripts/train_new2.py \
    --vlm_name llava-hf/llava-1.5-7b-hf \
    --sam_name facebook/sam-vit-base \
    --use_vlm_vision \
    --dtype bf16 \
    --data_dir . \
    --max_train_samples 1000 \
    --max_val_samples 200 \
    --epochs 20 \
    --batch_size 1 \
    --lr 3e-4 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --warmup_steps 100 \
    --bce_weight 1.0 \
    --dice_weight 1.0 \
    --iou_weight 1.0 \
    --output_dir outputs \
    --checkpoint_dir checkpoints \
    --save_every 1 \
    --seed 42 \
    --num_workers 0
```

### Quick Test Run (5 samples)

```bash
accelerate launch scripts/train_new2.py \
    --epochs 2 \
    --batch_size 1 \
    --max_train_samples 5 \
    --max_val_samples 3 \
    --lr 3e-4 \
    --dtype bf16 \
    --use_vlm_vision
```

### Without 4-bit Quantization (requires more memory)

```bash
accelerate launch scripts/train_new2.py \
    --no_quantization \
    --epochs 20 \
    --batch_size 1
```

### Use SAM's Vision Encoder Instead of VLM's

```bash
accelerate launch scripts/train_new2.py \
    --epochs 20 \
    --batch_size 1
# Note: --use_vlm_vision is a flag, omit it to use SAM's encoder
```

## Command-Line Arguments

### Model Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--vlm_name` | str | `llava-hf/llava-1.5-7b-hf` | Hugging Face VLM model |
| `--sam_name` | str | `facebook/sam-vit-base` | Hugging Face SAM model |
| `--use_vlm_vision` | flag | True | Use VLM's vision encoder (omit for SAM's) |
| `--dtype` | str | `bf16` | Data type (`bf16`, `fp16`, `fp32`) |
| `--no_quantization` | flag | False | Disable 4-bit quantization |

### Data Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | `.` | Root directory containing train/val/test |
| `--max_train_samples` | int | None | Limit training samples |
| `--max_val_samples` | int | None | Limit validation samples |

### Training Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 20 | Number of training epochs |
| `--batch_size` | int | 1 | Batch size (recommend 1 for 8GB GPU) |
| `--lr` | float | 3e-4 | Learning rate |
| `--weight_decay` | float | 0.01 | Weight decay |
| `--grad_clip` | float | 1.0 | Gradient clipping value |
| `--warmup_steps` | int | 100 | Warmup steps |

### Loss Weights

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bce_weight` | float | 1.0 | Binary cross-entropy weight |
| `--dice_weight` | float | 1.0 | Dice loss weight |
| `--iou_weight` | float | 1.0 | IoU loss weight |

### Output Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | `outputs` | Directory for metrics/logs |
| `--checkpoint_dir` | str | `checkpoints` | Directory for model checkpoints |
| `--save_every` | int | 1 | Save checkpoint every N epochs |
| `--seed` | int | 42 | Random seed |
| `--num_workers` | int | 0 | DataLoader workers |

## Output Files

### During Training

- `outputs/metrics_epoch_N.json` - Metrics for each epoch
- `outputs/training_history.json` - Complete training history
- `checkpoints/best_model_epoch_N.pt` - Best model checkpoint

### Metrics JSON Format

```json
{
  "epoch": 1,
  "train": {
    "loss": 0.5234,
    "bce": 0.2100,
    "dice": 0.1567,
    "iou_loss": 0.1567,
    "mIoU": 0.6543,
    "cIoU": 0.6543
  },
  "val": {
    "val_loss": 0.4892,
    "val_mIoU": 0.6789,
    "val_cIoU": 0.6789
  }
}
```

### Final Benchmark Results

```json
{
  "benchmarks": {
    "peak_memory_gb": 7.2,
    "avg_inference_time_s": 0.245
  }
}
```

## Loss Functions

The model uses a combination of three losses (same as original LISA):

1. **Binary Cross-Entropy (BCE)**: Pixel-wise classification loss
2. **Dice Loss**: Overlap-based segmentation loss
3. **IoU Loss**: Intersection-over-Union loss

**Total Loss** = `bce_weight × BCE + dice_weight × Dice + iou_weight × IoU`

## Metrics

- **mIoU (mean Intersection over Union)**: Average IoU across all predictions
- **cIoU (class IoU)**: Per-class IoU metric
- **gIoU (Generalized IoU)**: Bounding box-based metric

## Memory Requirements

### With 4-bit Quantization (Default)

- **GPU Memory**: ~7-8 GB
- **System RAM**: 16-32 GB recommended
- **Batch Size**: 1 (for 8GB GPU)

### Without Quantization

- **GPU Memory**: ~24 GB+
- **System RAM**: 32 GB+
- **Batch Size**: 1-2

## Training Tips

1. **Start Small**: Test with `--max_train_samples 5` first
2. **Monitor Memory**: Watch GPU utilization with `nvidia-smi`
3. **Adjust Learning Rate**: If loss oscillates, reduce `--lr`
4. **Gradient Accumulation**: For larger effective batch size, modify the code to accumulate gradients
5. **Mixed Precision**: Use `bf16` for better stability than `fp16` on modern GPUs

## Inference

To use the trained model for inference:

```python
import torch
from train_new2 import ModifiedLISA
from PIL import Image

# Load model
model = ModifiedLISA(
    vlm_name='llava-hf/llava-1.5-7b-hf',
    sam_name='facebook/sam-vit-base',
    use_vlm_vision_encoder=True,
    dtype='bf16'
)

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model_epoch_20.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input
image = Image.open('test_image.jpg')
text = "What object is shown? <SEG>"

# Forward pass
with torch.no_grad():
    outputs = model([image], [text], return_mask=True)
    pred_mask = outputs['pred_masks'][0]  # [1, 256, 256]
    iou_pred = outputs['iou_predictions'][0]
```

## Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size
--batch_size 1

# Enable quantization
# (remove --no_quantization flag)

# Reduce samples
--max_train_samples 100
```

### Model Loading Takes Too Long

- This is normal for 4-bit quantization
- First load takes 2-5 minutes
- Models are cached afterward in `~/.cache/huggingface/`

### CUDA Out of Memory During Training

```bash
# Clear cache before training
python3 -c "import torch; torch.cuda.empty_cache()"

# Then run training
accelerate launch scripts/train_new2.py ...
```

## Comparison with Original LISA

To benchmark against original LISA:

1. **Memory Usage**: Check `peak_memory_gb` in `training_history.json`
2. **Inference Time**: Check `avg_inference_time_s` in `training_history.json`
3. **Metrics**: Compare `val_mIoU` and `val_cIoU` with original LISA paper

Expected improvements:
- **Memory**: ~30-40% reduction
- **Speed**: ~20-30% faster inference
- **Accuracy**: Comparable to original LISA

## Citation

If you use this modified LISA implementation, please cite:

```bibtex
@article{modified-lisa-2024,
  title={Modified LISA: Efficient Reasoning Segmentation with Shared Vision Encoding},
  author={Your Name},
  year={2024}
}
```

And the original LISA paper:

```bibtex
@article{lisa2023,
  title={LISA: Reasoning Segmentation via Large Language Model},
  author={Lai, Xin and Tian, Zhuotao and Chen, Yukang and Li, Yanwei and Yuan, Yuhui and Liu, Shu and Jia, Jiaya},
  journal={arXiv preprint arXiv:2308.00692},
  year={2023}
}
```

## License

See LICENSE file.

## Support

For issues or questions:
1. Check this README
2. Review error messages carefully
3. Test with minimal dataset first (`--max_train_samples 5`)
4. Monitor GPU memory with `nvidia-smi`
