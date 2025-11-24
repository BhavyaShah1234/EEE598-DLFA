# ModifiedLISA Training - Quick Start

## Installation

```bash
pip install torch torchvision transformers accelerate peft pillow psutil
```

## Run Training

### Option 1: Using the bash script (Recommended)
```bash
./run_training.sh
```

### Option 2: Direct command
```bash
accelerate launch scripts/train_new.py
```

### Option 3: Single GPU
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_new.py
```

## Key Features

✅ **Complete Training Pipeline** - Dataset loading, training loop, validation  
✅ **LISA Metrics** - IoU, gIoU, cIoU for segmentation quality  
✅ **Benchmark Metrics** - Memory usage, iterations/sec, epoch time  
✅ **Distributed Training** - Multi-GPU support via Accelerate  
✅ **Automatic Checkpointing** - Best model saved based on validation IoU  
✅ **Mixed Precision** - FP16/BF16 for faster training  
✅ **LoRA Support** - Efficient fine-tuning with lower memory  

## Metrics Tracked

### Segmentation Quality (LISA-compatible)
- **IoU**: Intersection over Union
- **gIoU**: Generalized IoU
- **cIoU**: Complete IoU

### Performance Benchmarking
- **Iterations/second**: Training throughput
- **Average iteration time**: Per-step latency
- **Epoch time**: Total time per epoch
- **Peak memory usage**: Maximum GPU memory (GB)

### Loss Components
- Dice Loss
- Focal Loss
- Combined Loss

## Output Structure

```
outputs/
  ├── metrics_epoch_1.json
  ├── metrics_epoch_2.json
  └── ...

checkpoints/
  ├── best_model_epoch_1.pt
  ├── best_model_epoch_3.pt
  └── ...
```

## Configuration

Edit `TrainingConfig` in `scripts/train_new.py`:

```python
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 1e-4
    img_size: int = 224
    lora_r: int = 16
    # ... more options
```

## Monitoring

Watch training progress in real-time:
```
Epoch 1/10
Step 0/500 | Loss: 0.45 | IoU: 0.62 | gIoU: 0.59 | Iter/s: 2.3 | Mem: 12.4GB
Step 10/500 | Loss: 0.44 | IoU: 0.63 | gIoU: 0.60 | Iter/s: 2.4 | Mem: 12.5GB
...
```

## Troubleshooting

**Out of Memory?**
- Reduce `batch_size` (try 1)
- Enable quantization: `use_quantization = True`
- Reduce `img_size` (try 192 or 160)

**Slow Training?**
- Use mixed precision: `use_mixed_precision = True`
- Increase `batch_size` if memory allows
- Enable quantization

**Need More Details?**
See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for comprehensive documentation.

## Quick Test

Test if everything works:
```bash
# This will run a quick sanity check
python scripts/train_new.py --help
```

## Comparison with LISA

This implementation provides all metrics needed to benchmark against LISA:
1. Same segmentation metrics (IoU, gIoU, cIoU)
2. Performance metrics (speed, memory)
3. Compatible loss functions
4. Same model architecture with improvements

## Support

For issues or questions, check:
1. [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Detailed documentation
2. GPU memory: `nvidia-smi`
3. Logs in terminal output
