# ModifiedLISA Training Guide

## Overview
This guide explains how to train the ModifiedLISA model on the ReasonSeg dataset with comprehensive benchmarking metrics.

## Features Implemented

### 1. Complete Training Pipeline
- **Dataset Loading**: Custom `ReasonSegDataset` class that loads images and polygon masks from JSON files
- **Data Processing**: Automatic image resizing, normalization, and mask generation from polygon annotations
- **Distributed Training**: Full support for Accelerate framework for multi-GPU training
- **Mixed Precision**: FP16/BF16 support for faster training and reduced memory usage

### 2. Benchmarking Metrics

#### Segmentation Metrics (LISA-compatible)
- **IoU (Intersection over Union)**: Standard segmentation metric
- **gIoU (Generalized IoU)**: Handles non-overlapping predictions better
- **cIoU (Complete IoU)**: Enhanced IoU with additional geometric factors

#### Performance Metrics
- **Iterations per second**: Training speed measurement
- **Average iteration time**: Time per training step
- **Epoch time**: Total time for one complete epoch
- **Peak memory usage**: Maximum GPU memory consumption (GB)

#### Loss Functions
- **Dice Loss**: Optimizes overlap between prediction and ground truth
- **Focal Loss**: Handles class imbalance in segmentation
- **Combined Loss**: Weighted combination of Dice and Focal losses

### 3. Training Configuration

All hyperparameters are centralized in `TrainingConfig`:

```python
@dataclass
class TrainingConfig:
    # Paths
    data_root: str = "/home/bhavya-shah/Projects/EEE598-DLFA"
    
    # Training hyperparameters
    num_epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    
    # Model config
    img_size: int = 224
    max_text_length: int = 77
    
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
```

## Running the Training

### Basic Usage
```bash
accelerate launch scripts/train_new.py
```

### Multi-GPU Training
```bash
# Configure accelerate (first time only)
accelerate config

# Run training
accelerate launch scripts/train_new.py
```

### With Specific GPU
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch scripts/train_new.py
```

## Output Structure

### Checkpoints
Saved in `checkpoints/` directory:
- `best_model_epoch_X.pt`: Best model based on validation IoU
- Contains: model state, optimizer state, metrics

### Metrics
Saved in `outputs/` directory:
- `metrics_epoch_X.json`: Detailed metrics for each epoch
- Includes both training and validation metrics

### Example Metrics File
```json
{
  "epoch": 1,
  "train": {
    "loss": 0.4523,
    "iou": 0.6234,
    "giou": 0.5891,
    "ciou": 0.5891,
    "iter_per_sec": 2.34,
    "avg_iter_time": 0.427,
    "epoch_time": 1234.56,
    "peak_memory_gb": 12.45
  },
  "val": {
    "loss": 0.4123,
    "iou": 0.6534,
    "giou": 0.6212,
    "ciou": 0.6212
  }
}
```

## Benchmark Comparison

The script tracks all metrics needed to compare ModifiedLISA against LISA:

### Memory Efficiency
- **Peak GPU Memory**: Tracked during training
- **Memory per iteration**: Average memory consumption

### Computational Speed
- **Iterations/second**: Higher is better
- **Time per epoch**: Total training time
- **Average iteration time**: Per-step processing time

### Segmentation Quality
- **IoU**: Standard metric used in LISA paper
- **gIoU**: Generalized IoU for better handling of edge cases
- **cIoU**: Complete IoU with geometric considerations

## Dataset Structure

The script expects:
```
/home/bhavya-shah/Projects/EEE598-DLFA/
├── train/
│   ├── image1.jpg
│   ├── image1.json
│   └── ...
├── val/
│   ├── image1.jpg
│   ├── image1.json
│   └── ...
├── test/
│   ├── image1.jpg
│   ├── image1.json
│   └── ...
└── train.json (training metadata)
```

## Customization

### Modify Hyperparameters
Edit the `TrainingConfig` class in `train_new.py`:

```python
config = TrainingConfig()
config.batch_size = 4  # Increase batch size
config.learning_rate = 2e-4  # Adjust learning rate
config.num_epochs = 20  # Train for more epochs
```

### Change Model Configuration
Modify model settings directly:

```python
# Use quantization for reduced memory
config.use_quantization = True
config.quantization = "8bit"

# Freeze certain components
config.freeze_image_encoder = True
config.freeze_text_encoder = False
config.freeze_sam = False
```

### Adjust LoRA Settings
```python
config.lora_r = 32  # Increase LoRA rank
config.lora_alpha = 64
config.lora_dropout = 0.05
```

## Monitoring Training

During training, you'll see output like:
```
Epoch 1/10
Step 0/500 | Loss: 0.4523 | IoU: 0.6234 | gIoU: 0.5891 | Iter/s: 2.34 | Mem: 12.45GB
Step 10/500 | Loss: 0.4401 | IoU: 0.6312 | gIoU: 0.5967 | Iter/s: 2.38 | Mem: 12.48GB
...

Training Metrics:
  Loss: 0.4123
  IoU: 0.6534
  gIoU: 0.6212
  cIoU: 0.6212
  Iterations/sec: 2.35
  Avg iteration time: 0.425s
  Epoch time: 1245.67s
  Peak memory: 12.51GB

Validation Metrics:
  Loss: 0.3987
  IoU: 0.6678
  gIoU: 0.6345
  cIoU: 0.6345

New best validation IoU: 0.6678
Checkpoint saved to checkpoints/best_model_epoch_1.pt
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Enable `use_quantization = True`
- Reduce `img_size` (e.g., from 224 to 192)
- Increase `gradient_accumulation_steps`

### Slow Training
- Increase `batch_size` if memory allows
- Use mixed precision: `use_mixed_precision = True`
- Reduce number of workers in DataLoader
- Use quantization

### Poor Performance
- Increase `num_epochs`
- Adjust `learning_rate`
- Modify LoRA settings (increase `lora_r`)
- Check if data is loading correctly

## Key Differences from LISA

1. **Unified Training Script**: Everything in one file for simplicity
2. **Accelerate Integration**: Better distributed training support
3. **Comprehensive Metrics**: All LISA metrics + performance benchmarks
4. **Flexible Configuration**: Easy to modify via dataclass
5. **Automatic Checkpointing**: Best model saved based on validation IoU

## Next Steps

After training, you can:
1. Analyze metrics from `outputs/metrics_epoch_*.json`
2. Compare with LISA baseline performance
3. Visualize predictions using the saved checkpoints
4. Fine-tune hyperparameters based on results
