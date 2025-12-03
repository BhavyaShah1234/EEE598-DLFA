#!/usr/bin/env python3
"""Quick test script for Modified LISA model"""

import torch
import transformers as tf
from PIL import Image
from pathlib import Path

# Test if basic components work
print("Testing Modified LISA components...")
print("="*80)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Load a sample image
data_dir = Path('.')
train_dir = data_dir / 'train'
sample_img = list(train_dir.glob('*.jpg'))[0]
print(f"\nLoading sample image: {sample_img.name}")

image = Image.open(sample_img).convert('RGB')
print(f"Image size: {image.size}")

# Test VLM processor
print("\nTesting VLM processor...")
vlm_name = 'llava-hf/llava-1.5-7b-hf'
processor = tf.AutoProcessor.from_pretrained(vlm_name)

conversation = [
    {
        'role': 'user',
        'content': [
            {'type': 'image'},
            {'type': 'text', 'text': 'Test query <SEG>'}
        ]
    }
]
conv_text = processor.apply_chat_template(conversation, add_generation_prompt=False)
print(f"Conversation text length: {len(conv_text)}")

# Test SAM processor
print("\nTesting SAM processor...")
sam_name = 'facebook/sam-vit-base'
sam_processor = tf.AutoProcessor.from_pretrained(sam_name)
sam_inputs = sam_processor(images=[image], return_tensors='pt')
print(f"SAM input shape: {sam_inputs['pixel_values'].shape}")

print("\n" + "="*80)
print("✓ All components loaded successfully!")
print("="*80)
