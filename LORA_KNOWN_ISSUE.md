# LoRA + 4-bit Quantization Known Issue

## Problem

When using `--use_lora` with 4-bit quantization (default), the training encounters OOM (Out of Memory) errors on 8GB GPUs:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 86.00 MiB...
```

## Root Cause

The issue occurs because:
1. **4-bit quantization** stores weights in 4-bit format but **dequantizes to FP16/BF16 during forward/backward passes**
2. **LoRA adapters** add trainable parameters that require gradients
3. During backward pass, gradients must flow through the dequantized 4-bit layers
4. This creates **temporary high-precision tensors** that consume significant memory

Even though LoRA reduces trainable parameters by 90%, the base model still needs to be dequantized for gradient computation through the adapters.

## Solutions

### Solution 1: Use Quantization WITHOUT LoRA (Recommended for 8GB GPUs)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch scripts/train_new2.py \
    --epochs 20 \
    --batch_size 1 \
    --num_workers 0
```

**Memory Usage**: ~5.0 GB peak
**Trainable Params**: Projection layers + new embeddings only
**Status**: ✅ **Working** (tested successfully)

### Solution 2: Use LoRA WITHOUT Quantization (For 12GB+ GPUs)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch scripts/train_new2.py \
    --no_quantization \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --epochs 20 \
    --batch_size 1
```

**Memory Usage**: ~10-12 GB peak (estimated)
**Trainable Params**: ~8M params (0.13% of total)
**Status**: ⚠️ **Requires 12GB+ GPU**

### Solution 3: Reduce LoRA Rank (Experimental)

If you must use both:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch scripts/train_new2.py \
    --use_lora \
    --lora_r 4 \
    --lora_alpha 8 \
    --batch_size 1 \
    --grad_clip 0.5
```

**Memory Usage**: Slightly less than default LoRA
**Status**: ⚠️ **May still OOM on 8GB GPU**

## Recommendations by GPU Memory

### 8GB GPU (RTX 3060, RTX 5060)
```bash
# Use 4-bit quantization WITHOUT LoRA
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch scripts/train_new2.py \
    --batch_size 1 \
    --num_workers 0 \
    --epochs 20
```

### 12GB GPU (RTX 3080, RTX 4070)
```bash
# Use LoRA WITHOUT quantization
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch scripts/train_new2.py \
    --no_quantization \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --batch_size 2 \
    --num_workers 2
```

### 16GB+ GPU (RTX 4080, RTX 4090)
```bash
# Use BOTH LoRA and quantization (or neither)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch scripts/train_new2.py \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --batch_size 4 \
    --num_workers 4
```

## Technical Details

### Why LoRA + Quantization is Memory-Intensive

1. **Forward Pass**:
   ```python
   # 4-bit weights dequantized to FP16/BF16
   dequantized = dequantize_4bit(weight_4bit)  # Temporary tensor!
   output = matmul(input, dequantized)
   ```

2. **LoRA Adaptation**:
   ```python
   # LoRA adds low-rank matrices
   base_output = base_layer(x)
   lora_output = lora_B(lora_A(x))  # Trainable!
   final_output = base_output + lora_output
   ```

3. **Backward Pass**:
   ```python
   # Gradients flow through BOTH paths
   grad_base = backward_through_dequantized_weights()  # Memory spike!
   grad_lora = backward_through_lora_adapters()
   ```

The `backward_through_dequantized_weights()` requires keeping dequantized weights in memory, causing OOM.

### Alternative: QLoRA

True QLoRA (used in the original paper) requires:
- Custom CUDA kernels for gradient computation through quantized weights
- Not fully supported in current transformers/PEFT versions

## Tested Configurations

| Config | Quantization | LoRA | GPU | Status | Peak Memory |
|--------|-------------|------|-----|--------|-------------|
| 1 | ✅ 4-bit | ❌ No | 8GB | ✅ Works | 5.0 GB |
| 2 | ✅ 4-bit | ✅ r=8 | 8GB | ❌ OOM | N/A |
| 3 | ✅ 4-bit | ✅ r=4 | 8GB | ❌ OOM | N/A |
| 4 | ❌ FP16 | ✅ r=8 | 12GB+ | ⚠️ Untested | ~10-12 GB |

## Current Status

✅ **Working**: 4-bit quantization without LoRA
❌ **Not Working**: 4-bit quantization + LoRA (OOM on 8GB)
⚠️ **Untested**: FP16 + LoRA (requires 12GB+)

## Workaround for Development

For testing LoRA functionality without OOM:

```bash
# Test with CPU offloading (VERY slow but works)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch scripts/train_new2.py \
    --use_lora \
    --max_train_samples 2 \
    --max_val_samples 2 \
    --epochs 1
```

Then use accelerate config to enable CPU offloading.

## Conclusion

For production training on 8GB GPUs:
- **Use**: 4-bit quantization (default)
- **Don't use**: `--use_lora` flag
- **Result**: Successfully trains with ~5GB peak memory

The current implementation correctly freezes all pretrained weights and only trains:
- Vision projection layers
- SEG token projection
- New token embeddings
- LM head (output projection)

This is sufficient for fine-tuning LISA on ReasonSeg without LoRA.
