# VLM Integration Guide for ModifiedLISA

## Overview

The `scripts/train_new.py` now supports **Vision-Language Models (VLMs)** as an alternative to using separate image and text encoders. When VLM mode is enabled, a single multimodal model handles both image encoding and text encoding tasks.

## Key Features

✅ **Unified Architecture**: Single VLM replaces separate CLIP + LLM encoders  
✅ **Multi-Model Support**: Works with LLaVA, SMoLVLM, QwenVL, PaliGemma, and more  
✅ **Flexible Configuration**: All VLM settings configurable via CLI  
✅ **Backward Compatible**: Traditional mode still available when `--use_vlm` is not set  
✅ **LoRA Support**: Apply LoRA to VLM for efficient fine-tuning  
✅ **Quantization**: 4-bit/8-bit quantization for memory efficiency

## Supported VLM Models

### LLaVA (Recommended)
```bash
--vlm_model_name llava-hf/llava-1.5-7b-hf
--vlm_model_name llava-hf/llava-1.5-13b-hf
--vlm_model_name llava-hf/llava-v1.6-mistral-7b-hf
```

### SMoLVLM
```bash
--vlm_model_name HuggingFaceTB/SmolVLM-Instruct
--vlm_model_name HuggingFaceTB/SmolVLM-Base
```

### QwenVL
```bash
--vlm_model_name Qwen/Qwen2-VL-2B-Instruct
--vlm_model_name Qwen/Qwen2-VL-7B-Instruct
```

### PaliGemma
```bash
--vlm_model_name google/paligemma-3b-pt-224
--vlm_model_name google/paligemma-3b-mix-224
```

## CLI Arguments

### VLM Mode Control
```bash
--use_vlm                    # Enable VLM mode (overrides separate encoders)
--vlm_model_name MODEL       # HuggingFace VLM model name
```

### VLM Precision Settings
```bash
--vlm_use_mixed_precision    # Enable mixed precision for VLM
--vlm_mixed_precision TYPE   # Choices: 'no', 'fp16', 'bf16'
```

### VLM Quantization
```bash
--vlm_use_quantization       # Enable quantization for VLM
--vlm_quantization TYPE      # Choices: '4bit', '8bit'
```

### VLM Training Modes
```bash
--vlm_freeze                 # Freeze VLM weights (feature extraction only)
--vlm_use_lora               # Apply LoRA adapters to VLM
--vlm_lora_r RANK            # LoRA rank (default: 16)
--vlm_lora_alpha ALPHA       # LoRA alpha (default: 32)
--vlm_lora_dropout DROPOUT   # LoRA dropout (default: 0.1)
```

## Usage Examples

### Example 1: Basic VLM Training with LLaVA
```bash
accelerate launch scripts/train_new.py \
  --use_vlm \
  --vlm_model_name llava-hf/llava-1.5-7b-hf \
  --vlm_use_lora \
  --num_epochs 10 \
  --batch_size 1 \
  --run_test
```

### Example 2: VLM with BF16 Mixed Precision
```bash
accelerate launch scripts/train_new.py \
  --use_vlm \
  --vlm_model_name HuggingFaceTB/SmolVLM-Instruct \
  --vlm_use_mixed_precision \
  --vlm_mixed_precision bf16 \
  --vlm_use_lora \
  --num_epochs 20
```

### Example 3: 8-bit Quantized VLM
```bash
accelerate launch scripts/train_new.py \
  --use_vlm \
  --vlm_model_name Qwen/Qwen2-VL-2B-Instruct \
  --vlm_use_quantization \
  --vlm_quantization 8bit \
  --vlm_use_lora \
  --vlm_lora_r 8 \
  --batch_size 2
```

### Example 4: Frozen VLM (Feature Extraction)
```bash
accelerate launch scripts/train_new.py \
  --use_vlm \
  --vlm_model_name google/paligemma-3b-pt-224 \
  --vlm_freeze \
  --learning_rate 1e-3 \
  --num_epochs 15
```

### Example 5: Traditional Mode (No VLM)
```bash
accelerate launch scripts/train_new.py \
  --use_lora \
  --image_encoder_model_name openai/clip-vit-base-patch16 \
  --text_encoder_model_name Qwen/Qwen3-0.6B-Base \
  --num_epochs 20
```

## Architecture Comparison

### Traditional Mode (--use_vlm NOT set)
```
Input Image → CLIP Encoder → Vision Features ──┐
                                               ├→ Connectors → SAM → Masks
Input Text  → LLM Encoder  → Text Features  ──┘
```

### VLM Mode (--use_vlm set)
```
Input Image ──┐
              ├→ VLM → Vision Features ──┐
Input Text  ──┘        Text Features   ──┘→ Connectors → SAM → Masks
```

## VLM Class Implementation

The `VLM` class provides a unified interface:

### Methods
1. **`encode_image(pixel_values)`**
   - Encodes images to vision features
   - Returns: `(features_seq, spatial_features)`

2. **`encode_text(input_ids, attention_mask)`**
   - Encodes text to text features
   - Returns: `text_features`

3. **`forward(pixel_values, input_ids, attention_mask)`**
   - Full multimodal forward pass
   - Returns: VLM outputs with hidden states

### Key Features
- **Auto-detection**: Automatically detects VLM type (LLaVA, SMoLVLM, etc.)
- **Component Extraction**: Extracts vision_tower and language_model components
- **Unified Processing**: Single processor for both image and text
- **Embedding Dimensions**: Automatically determines vision and text embedding sizes

## Technical Details

### VLM Initialization Flow
1. Parse `--use_vlm` flag
2. If enabled:
   - Load VLM model with specified precision/quantization
   - Extract vision and text components
   - Set embedding dimensions from VLM config
   - Override `image_encoder_model_name` and `text_encoder_model_name`
3. If disabled:
   - Use traditional separate encoders

### Forward Pass Flow (VLM Mode)
1. **Image Encoding**: `vlm.encode_image()` → vision features
2. **Text Encoding**: `vlm.encode_text()` → text features
3. **Projection**: ImageTextConnector projects vision to text space
4. **Fusion**: Concatenate projected vision + text features
5. **Prompt Extraction**: TextSAMConnector generates SAM prompts
6. **SAM Adaptation**: ImageSAMConnector adapts vision features for SAM
7. **Mask Decoding**: SAM decoder generates segmentation masks

### Connector Compatibility
The connector classes remain unchanged and work with both modes:
- **ImageTextConnector**: Projects vision features to LLM space
- **TextSAMConnector**: Extracts prompt embeddings for SAM
- **ImageEncoderSAMConnector**: Adapts vision features for SAM

## Memory & Performance

### Memory Usage Estimates (Batch Size = 1)

| VLM Model | Precision | Quantization | VRAM Usage |
|-----------|-----------|--------------|------------|
| LLaVA-1.5-7B | FP32 | None | ~28 GB |
| LLaVA-1.5-7B | BF16 | None | ~14 GB |
| LLaVA-1.5-7B | - | 8-bit | ~7 GB |
| LLaVA-1.5-7B | - | 4-bit | ~4 GB |
| SmolVLM | BF16 | None | ~6 GB |
| Qwen2-VL-2B | BF16 | None | ~8 GB |
| PaliGemma-3B | BF16 | None | ~6 GB |

### Recommendations
- **8GB GPU**: Use 8-bit quantization with LoRA
- **16GB GPU**: Use BF16 with LoRA
- **24GB+ GPU**: Full precision training possible

## Best Practices

### 1. Start with LoRA
```bash
--vlm_use_lora \
--vlm_lora_r 16 \
--vlm_lora_alpha 32
```
LoRA reduces trainable parameters by ~98% while maintaining performance.

### 2. Use BF16 for Modern GPUs
```bash
--vlm_use_mixed_precision \
--vlm_mixed_precision bf16
```
BF16 provides better numerical stability than FP16 for VLMs.

### 3. Enable Gradient Accumulation
```bash
--batch_size 1 \
--gradient_accumulation_steps 8
```
Simulates larger batch sizes with limited VRAM.

### 4. Monitor Memory
The training script automatically reports peak memory usage.

### 5. Test Before Full Training
```bash
--num_epochs 2 \
--max_train_samples 20 \
--max_val_samples 10
```

## Troubleshooting

### Issue: Out of Memory
**Solutions:**
1. Enable quantization: `--vlm_use_quantization --vlm_quantization 8bit`
2. Reduce batch size: `--batch_size 1`
3. Use gradient checkpointing (if model supports)
4. Reduce image size: `--img_size 224`

### Issue: VLM Model Not Found
**Solutions:**
1. Check model name on HuggingFace
2. Ensure model is publicly available
3. Login to HF if using gated models: `huggingface-cli login`

### Issue: Slow Training
**Solutions:**
1. Enable mixed precision: `--vlm_use_mixed_precision --vlm_mixed_precision bf16`
2. Increase num_workers: `--num_workers 8`
3. Use smaller VLM: Switch to SmolVLM or smaller variant

### Issue: LoRA Not Working with FP16
**Solutions:**
Use BF16 instead: `--vlm_mixed_precision bf16`

## Comparison: VLM vs Traditional Mode

| Aspect | VLM Mode | Traditional Mode |
|--------|----------|------------------|
| **Models** | Single multimodal model | Separate CLIP + LLM |
| **Parameters** | Typically larger (~2-13B) | Typically smaller (~1-2B total) |
| **Training Speed** | Slower (larger model) | Faster (smaller models) |
| **Memory** | Higher (single large model) | Lower (two small models) |
| **Performance** | Better multimodal understanding | Good for simpler tasks |
| **Flexibility** | Less flexible (tied to VLM) | More flexible (mix models) |

## When to Use VLM Mode

### ✅ Use VLM When:
- Need strong multimodal reasoning
- Have sufficient GPU memory (16GB+)
- Working with complex visual-language tasks
- Want state-of-the-art VLM capabilities

### ❌ Use Traditional Mode When:
- Limited GPU memory (<8GB)
- Need fastest training speed
- Want maximum model flexibility
- Simple image-text alignment tasks

## Example Training Outputs

### VLM Mode
```
Using VLM mode
  → VLM quantization: 8-bit
  ✓ VLM loaded: llava-hf/llava-1.5-7b-hf
    Vision dim: 1024, Text dim: 4096
  → VLM mode: LoRA (r=16, alpha=32)
trainable params: 8,388,608 || all params: 7,109,234,688 || trainable%: 0.12%
...
```

### Traditional Mode
```
Using traditional mode (separate encoders)
  → Vision mode: LoRA (r=16, alpha=32)
  ✓ CLIP loaded. Embedding dim: 768
  → LLM mode: LoRA (r=16, alpha=32)
  ✓ LLM loaded. Embedding dim: 2048
trainable params: 27,308,160 || all params: 1,442,657,456 || trainable%: 1.89%
...
```

## Future Enhancements

- [ ] Support for more VLM architectures
- [ ] Flash Attention integration
- [ ] Gradient checkpointing option
- [ ] Multi-GPU training optimization
- [ ] VLM-specific data augmentation

## References

- [LLaVA: Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA)
- [SMoLVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)
- [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224)
