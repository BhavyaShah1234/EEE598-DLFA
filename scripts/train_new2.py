#!/usr/bin/env python3
"""
Modified LISA Model Training Script
Implements shared image embeddings between VLM and SAM for efficient reasoning segmentation
"""

import argparse
import json
import os
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import transformers as tf
from transformers import BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from PIL import Image
from tqdm import tqdm
import cv2
from accelerate import Accelerator
from accelerate.utils import set_seed
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ===============================
# Dataset Class
# ===============================
class ReasonSegDataset(Dataset):
    """Dataset for ReasonSeg reasoning segmentation task
    
    Creates one sample per text query, so each image can generate multiple training samples.
    """
    
    def __init__(self, data_dir: str, split: str = 'train', max_samples: Optional[int] = None, vlm_processor=None, sam_processor=None):
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.vlm_processor = vlm_processor
        self.sam_processor = sam_processor
        
        # Find all JSON annotation files and create one entry per text query
        self.samples = []
        json_files = list(self.data_dir.glob('*.json'))
        
        for json_file in json_files:
            img_name = json_file.stem
            # Check for corresponding image
            jpg_path = self.data_dir / f"{img_name}.jpg"
            if jpg_path.exists():
                # Load JSON to get text queries
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Get all text queries
                text_list = data.get('text', [])
                if not isinstance(text_list, list):
                    text_list = [text_list]
                
                # Create one sample per text query
                for text_idx, text in enumerate(text_list):
                    self.samples.append({
                        'json_path': json_file,
                        'img_path': jpg_path,
                        'name': img_name,
                        'text_idx': text_idx,
                        'text': text
                    })
        
        if max_samples is not None:
            self.samples = random.choices(self.samples, k=max_samples)
            # self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load annotation
        with open(sample['json_path'], 'r') as f:
            data = json.load(f)
        
        # Load image
        image = Image.open(sample['img_path']).convert('RGB')
        width, height = image.size
        
        # Get the specific text query for this sample
        text = sample['text']
        
        # Add <SEG> token if not present
        if '<SEG>' not in text:
            text = text + ' <SEG>'
        
        # Get polygon points and create mask
        shapes = data.get('shapes', [])
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for shape in shapes:
            if shape['shape_type'] == 'polygon':
                points = np.array(shape['points'], dtype=np.int32)
                mask = cv2.fillPoly(mask, [points], 1)
        
        result = {
            'image': image,
            'text': text,
            'mask': mask,
            'name': f"{sample['name']}_text{sample['text_idx']}"
        }
        
        # Preprocessing for VLM if processor is available
        if self.vlm_processor is not None:
            conversation = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': text}
                    ]
                }
            ]
            conv_text = self.vlm_processor.apply_chat_template(
                conversation,
                add_generation_prompt=False
            )
            result['conversation'] = conv_text
            
            # Store the input text without <SEG> for language modeling labels
            # The model should learn to generate <SEG> token
            text_without_seg = text.replace(' <SEG>', '')
            result['text_input'] = text_without_seg
            result['text_target'] = text  # Target includes <SEG>
        
        return result


# ===============================
# Modified LISA Model
# ===============================
class ModifiedLISA(nn.Module):
    """
    Modified LISA model with shared image embeddings between VLM and SAM
    Key innovation: Single vision encoder forward pass reused for both components
    """
    
    def __init__(
        self,
        vlm_name: str = 'llava-hf/llava-1.5-7b-hf',
        sam_name: str = 'facebook/sam-vit-base',
        use_vlm_vision_encoder: bool = True,  # If True, use VLM's vision encoder; else use SAM's
        dtype: str = 'bf16',
        use_quantization: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # Component-level training control
        train_vlm_vision: bool = False,
        train_vlm_projector: bool = True,
        train_vlm_llm: bool = True,
        train_sam_vision: bool = False,
        train_sam_prompt_encoder: bool = False,
        train_sam_mask_decoder: bool = True,
        train_projection_layers: bool = True
    ):
        super().__init__()
        
        self.vlm_name = vlm_name
        self.sam_name = sam_name
        self.use_vlm_vision_encoder = use_vlm_vision_encoder
        self.use_quantization = use_quantization
        
        # Setup dtype
        if dtype == 'bf16':
            self.dtype = torch.bfloat16
        elif dtype == 'fp16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        # Load VLM components
        print("Loading VLM processor and model...")
        self.vlm_processor = tf.AutoProcessor.from_pretrained(vlm_name)
        
        if use_quantization:
            # Quantization config for 4-bit models
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True
            )
            vlm_model = tf.AutoModelForImageTextToText.from_pretrained(
                vlm_name,
                quantization_config=quantization_config,
                dtype=self.dtype,
                device_map='auto',
                low_cpu_mem_usage=True
            )
        else:
            vlm_model = tf.AutoModelForImageTextToText.from_pretrained(
                vlm_name,
                dtype=self.dtype,
                low_cpu_mem_usage=True
            )
        
        # Extract VLM components
        self.vlm_vision_tower = vlm_model.vision_tower
        self.vlm_projector = vlm_model.multi_modal_projector
        self.vlm_llm = vlm_model.language_model
        self.vlm_lm_head = vlm_model.lm_head
        self.vlm_config = vlm_model.config
        
        # Apply LoRA if requested (BEFORE freezing)
        self.use_lora = use_lora
        if use_lora and train_vlm_llm:
            print("Applying LoRA to VLM language model...")
            # Prepare model for k-bit training if using quantization
            if use_quantization:
                # Enable gradient checkpointing to reduce memory
                self.vlm_llm.gradient_checkpointing_enable()
                self.vlm_llm = prepare_model_for_kbit_training(
                    self.vlm_llm,
                    use_gradient_checkpointing=True
                )
            
            # Default LoRA target modules for LLaMA-based models
            if lora_target_modules is None:
                # Only target attention layers to reduce memory
                lora_target_modules = ["q_proj", "v_proj"]
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                # Use None instead of task_type since we're not using generate()
                task_type=None,
                # Enable more memory optimizations
                modules_to_save=None
            )
            
            # Apply LoRA to language model
            self.vlm_llm = get_peft_model(self.vlm_llm, lora_config)
            print(f"LoRA applied with r={lora_r}, alpha={lora_alpha}")
            print(f"Target modules: {lora_target_modules}")
            self.vlm_llm.print_trainable_parameters()
        else:
            # Freeze or unfreeze VLM LLM based on train_vlm_llm flag
            # Only set requires_grad for non-quantized models
            if not use_quantization:
                for param in self.vlm_llm.parameters():
                    param.requires_grad = train_vlm_llm
        
        # Control VLM vision and projector training
        # Only set requires_grad for non-quantized models
        if not use_quantization:
            for param in self.vlm_vision_tower.parameters():
                param.requires_grad = train_vlm_vision
            for param in self.vlm_projector.parameters():
                param.requires_grad = train_vlm_projector
        
        if train_vlm_vision:
            print("VLM vision encoder: TRAINABLE")
        if train_vlm_projector:
            print("VLM projector: TRAINABLE")
        if train_vlm_llm and not use_lora:
            print("VLM LLM: TRAINABLE (full fine-tuning)")
        
        # Add <SEG> token to vocabulary using transformers' built-in method
        num_added = self.vlm_processor.tokenizer.add_tokens(['<SEG>'])
        print(f"Added {num_added} new token(s) to tokenizer")
        
        # Resize token embeddings - this handles both input embeddings and LM head
        # For VLMs, we need to resize the language_model component
        vlm_model.language_model.resize_token_embeddings(len(self.vlm_processor.tokenizer))
        
        # Get the token ID after adding
        self.seg_token_id = self.vlm_processor.tokenizer.convert_tokens_to_ids('<SEG>')
        print(f"<SEG> token ID: {self.seg_token_id}")
        
        # Store references to embedding layers and LM head (after resizing)
        self.vlm_word_embeddings = vlm_model.get_input_embeddings()
        self.vlm_lm_head = vlm_model.lm_head
        
        # Load SAM components
        print("Loading SAM processor and model...")
        self.sam_processor = tf.AutoProcessor.from_pretrained(sam_name)
        
        if use_quantization:
            sam_model = tf.AutoModel.from_pretrained(
                sam_name,
                quantization_config=quantization_config,
                dtype=self.dtype,
                device_map='auto',
                low_cpu_mem_usage=True
            )
        else:
            sam_model = tf.AutoModel.from_pretrained(
                sam_name,
                dtype=self.dtype,
                low_cpu_mem_usage=True
            )
        
        # Extract SAM components
        self.sam_vision_encoder = sam_model.vision_encoder
        self.sam_prompt_encoder = sam_model.prompt_encoder
        self.sam_mask_decoder = sam_model.mask_decoder
        self.sam_get_image_pe = sam_model.get_image_wide_positional_embeddings
        
        # Control SAM component training
        # Only set requires_grad for non-quantized models
        if not use_quantization:
            for param in self.sam_vision_encoder.parameters():
                param.requires_grad = train_sam_vision
            for param in self.sam_prompt_encoder.parameters():
                param.requires_grad = train_sam_prompt_encoder
            for param in self.sam_mask_decoder.parameters():
                param.requires_grad = train_sam_mask_decoder
        
        if train_sam_vision:
            print("SAM vision encoder: TRAINABLE")
        if train_sam_prompt_encoder:
            print("SAM prompt encoder: TRAINABLE")
        if train_sam_mask_decoder:
            print("SAM mask decoder: TRAINABLE")
        
        # Projection layers to bridge dimension gaps
        if use_vlm_vision_encoder:
            # VLM vision output -> SAM vision encoder output
            # LLaVA: [B, 576, 1024] -> SAM: [B, 64, 64, 256]
            self.vision_to_sam = nn.Sequential(
                nn.Linear(1024, 256),
                nn.GELU(),
                nn.LayerNorm(256)
            )
            # Control projection layer training
            for param in self.vision_to_sam.parameters():
                param.requires_grad = train_projection_layers
        else:
            # SAM vision output -> VLM projector input
            # SAM: [B, 64, 64, 256] -> LLaVA expects: [B, 576, 1024]
            self.vision_to_vlm = nn.Sequential(
                nn.Linear(256, 1024),
                nn.GELU(),
                nn.LayerNorm(1024)
            )
            # Control projection layer training
            for param in self.vision_to_vlm.parameters():
                param.requires_grad = train_projection_layers
        
        # <SEG> token embedding -> SAM mask decoder embedding
        # VLM hidden: 4096 -> SAM sparse embedding: 256
        self.seg_token_to_sam = nn.Sequential(
            nn.Linear(4096, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )
        # Always train this projection
        for param in self.seg_token_to_sam.parameters():
            param.requires_grad = train_projection_layers
        
        if train_projection_layers:
            print("Projection layers: TRAINABLE")
        
        # Print detailed parameter counts for each component
        print("\n" + "="*80)
        print("Architecture Parameter Breakdown")
        print("="*80)
        
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        def count_unique_params(module):
            """Count unique parameters (avoiding double-counting shared params)"""
            seen_params = set()
            total = 0
            for p in module.parameters():
                param_id = id(p)
                if param_id not in seen_params:
                    seen_params.add(param_id)
                    total += p.numel()
            return total
        
        # Count VLM components (note: embeddings are shared between LLM and LM head)
        vlm_vision_params = count_params(self.vlm_vision_tower)
        vlm_projector_params = count_params(self.vlm_projector)
        vlm_llm_params = count_unique_params(self.vlm_llm)
        vlm_lm_head_params = count_params(self.vlm_lm_head)
        
        # Count SAM components
        sam_vision_params = count_params(self.sam_vision_encoder)
        sam_prompt_params = count_params(self.sam_prompt_encoder)
        sam_decoder_params = count_params(self.sam_mask_decoder)
        
        # Count projection layers
        if use_vlm_vision_encoder:
            projection_params = count_params(self.vision_to_sam) + count_params(self.seg_token_to_sam)
        else:
            projection_params = count_params(self.vision_to_vlm) + count_params(self.seg_token_to_sam)
        
        # Total unique parameters in the full architecture
        # Count all unique parameters across the entire model
        total_unique_params = count_unique_params(self)
        
        # For display, show component breakdown
        component_sum = (vlm_vision_params + vlm_projector_params + vlm_llm_params + 
                        vlm_lm_head_params + sam_vision_params + sam_prompt_params + 
                        sam_decoder_params + projection_params)
        
        print(f"VLM Components:")
        print(f"  Vision Tower:      {vlm_vision_params:>15,} ({100*vlm_vision_params/total_unique_params:>6.2f}%)")
        print(f"  Projector:         {vlm_projector_params:>15,} ({100*vlm_projector_params/total_unique_params:>6.2f}%)")
        print(f"  Language Model:    {vlm_llm_params:>15,} ({100*vlm_llm_params/total_unique_params:>6.2f}%)")
        print(f"  LM Head:           {vlm_lm_head_params:>15,} ({100*vlm_lm_head_params/total_unique_params:>6.2f}%)")
        print(f"\nSAM Components:")
        print(f"  Vision Encoder:    {sam_vision_params:>15,} ({100*sam_vision_params/total_unique_params:>6.2f}%)")
        print(f"  Prompt Encoder:    {sam_prompt_params:>15,} ({100*sam_prompt_params/total_unique_params:>6.2f}%)")
        print(f"  Mask Decoder:      {sam_decoder_params:>15,} ({100*sam_decoder_params/total_unique_params:>6.2f}%)")
        print(f"\nProjection Layers:   {projection_params:>15,} ({100*projection_params/total_unique_params:>6.2f}%)")
        print(f"\n{'-'*80}")
        
        if use_quantization:
            # With 4-bit quantization
            # The parameter count represents quantized params, but original architecture is larger
            print(f"Modified LISA Architecture:")
            print(f"  LLaVA-1.5-7B:    ~7,063,427,072 params (7.06B)")
            print(f"  SAM-ViT-Base:      ~93,735,472 params (93.7M)")
            print(f"  Projections:        ~1,312,256 params (1.3M)")
            print(f"  {'-'*78}")
            print(f"  Full Precision:  ~7,158,474,800 params (7.16B)")
            print(f"\nCurrent Configuration (4-bit NF4 quantization):")
            print(f"  Quantized params:  {total_unique_params:>15,} ({total_unique_params/1e9:.2f}B)")
            print(f"  Memory savings:    ~4x less than FP16 (4-bit vs 16-bit)")
            print(f"\nNote: Quantization reduces memory footprint, not parameter count.")
            print(f"      Some parameters may not be quantized (LayerNorm, etc.)")
        else:
            print(f"Modified LISA Architecture (Full Precision):")
            print(f"  Total Parameters:  {total_unique_params:>15,} ({total_unique_params/1e9:.2f}B)")
            print(f"\n  = LLaVA-1.5 (7.06B) + SAM (93.7M) + Projections (1.3M)")
        
        print("="*80)
        
        print("\nModified LISA model initialized successfully!")
    
    def forward(
        self,
        vlm_inputs: Dict[str, torch.Tensor],
        sam_inputs: Dict[str, torch.Tensor],
        return_mask: bool = True
    ):
        """
        Forward pass through modified LISA
        
        Args:
            vlm_inputs: Dictionary with 'pixel_values', 'input_ids', 'attention_mask' from VLM processor
            sam_inputs: Dictionary with 'pixel_values' from SAM processor
            return_mask: Whether to return segmentation mask
        
        Returns:
            Dictionary with masks, IoU predictions, and <SEG> embeddings
        """
        device = next(self.parameters()).device
        batch_size = vlm_inputs['input_ids'].shape[0]
        
        # ===== SHARED VISION ENCODING =====
        if self.use_vlm_vision_encoder:
            # Use VLM's vision encoder
            with torch.no_grad():
                vision_outputs = self.vlm_vision_tower(
                    vlm_inputs['pixel_values'],
                    output_hidden_states=True
                ).hidden_states[self.vlm_config.vision_feature_layer]
                
                # Remove CLS token for VLM
                if self.vlm_config.vision_feature_select_strategy == "default":
                    vision_features_vlm = vision_outputs[:, 1:, :]
                    vision_for_sam = vision_outputs[:, 1:, :]  # Also remove CLS for SAM
                else:
                    vision_features_vlm = vision_outputs
                    vision_for_sam = vision_outputs
                
                # Project to SAM dimensions
                B = vision_for_sam.shape[0]
                N = vision_for_sam.shape[1]  # Number of tokens (e.g., 576 = 24x24)
                
                # Apply projection
                vision_features_sam = self.vision_to_sam(vision_for_sam)  # [B, N, 256]
                
                # Reshape to SAM format [B, 64, 64, 256]
                # We need to interpolate from sqrt(N) x sqrt(N) to 64 x 64
                import math
                h = w = int(math.sqrt(N))  # e.g., 24x24 for CLIP
                vision_features_sam = vision_features_sam.reshape(B, h, w, 256)
                
                # Interpolate to 64x64 if needed
                if h != 64 or w != 64:
                    # Permute to [B, 256, h, w] for interpolation
                    vision_features_sam = vision_features_sam.permute(0, 3, 1, 2)
                    vision_features_sam = F.interpolate(
                        vision_features_sam,
                        size=(64, 64),
                        mode='bilinear',
                        align_corners=False
                    )
                    # Keep in [B, 256, 64, 64] format for SAM
                else:
                    # Permute to [B, 256, 64, 64]
                    vision_features_sam = vision_features_sam.permute(0, 3, 1, 2)
        else:
            # Use SAM's vision encoder
            with torch.no_grad():
                sam_vision_out = self.sam_vision_encoder(
                    sam_inputs['pixel_values']
                )
                vision_features_sam = sam_vision_out.last_hidden_state  # [B, 256, 64, 64]
                
                # Flatten and project to VLM dimensions
                B, C, H, W = vision_features_sam.shape
                # Permute to [B, H, W, C] then flatten
                vision_flat = vision_features_sam.permute(0, 2, 3, 1).reshape(B, H * W, C)
                vision_features_vlm = self.vision_to_vlm(vision_flat)
        
        # ===== VLM FORWARD PASS =====
        # Project vision features
        projector_outputs = self.vlm_projector(vision_features_vlm)
        
        # Get text embeddings
        text_embeddings = self.vlm_word_embeddings(vlm_inputs['input_ids'])
        
        # Clone to avoid in-place operations on leaf variable (needed for LoRA)
        text_embeddings = text_embeddings.clone()
        
        # Merge image and text embeddings using vectorized operations
        image_token_id = self.vlm_config.image_token_index
        mask = vlm_inputs['input_ids'] == image_token_id
        _, y_indices = mask.nonzero(as_tuple=True)
        
        if len(y_indices) > 0:
            # Replace image token embeddings with projected vision features
            # Ensure dtype matches
            y_min = y_indices.min()
            text_embeddings[:, y_indices, :] = projector_outputs[:, y_indices - y_min, :].to(text_embeddings.dtype)
        
        # Forward through LLM
        llm_outputs = self.vlm_llm(
            inputs_embeds=text_embeddings,
            attention_mask=vlm_inputs['attention_mask'],
            output_hidden_states=True
        )
        hidden_states = llm_outputs.last_hidden_state
        
        # Extract <SEG> token embeddings using vectorized operations
        seg_mask = vlm_inputs['input_ids'] == self.seg_token_id
        batch_indices, seg_positions = seg_mask.nonzero(as_tuple=True)
        
        # Create a tensor to store seg embeddings
        seg_embeddings = torch.zeros(batch_size, hidden_states.shape[-1], device=device, dtype=hidden_states.dtype)
        
        if len(batch_indices) > 0:
            # Extract embeddings for samples with <SEG> token
            seg_embeddings[batch_indices] = hidden_states[batch_indices, seg_positions]
            
            # For samples without <SEG> token, use last token
            has_seg = torch.zeros(batch_size, dtype=torch.bool, device=device)
            has_seg[batch_indices] = True
            no_seg_indices = (~has_seg).nonzero(as_tuple=True)[0]
            
            if len(no_seg_indices) > 0:
                # Get sequence lengths for samples without <SEG>
                seq_lengths = vlm_inputs['attention_mask'][no_seg_indices].sum(dim=1) - 1
                seg_embeddings[no_seg_indices] = hidden_states[no_seg_indices, seq_lengths]
        else:
            # If no <SEG> tokens found, use last token for all samples
            seq_lengths = vlm_inputs['attention_mask'].sum(dim=1) - 1
            seg_embeddings = hidden_states[torch.arange(batch_size, device=device), seq_lengths]
        
        if not return_mask:
            return {'seg_embeddings': seg_embeddings}
        
        # ===== SAM MASK DECODING =====
        # Project <SEG> embeddings to SAM sparse embeddings
        seg_sparse_embeddings = self.seg_token_to_sam(seg_embeddings)  # [B, 256]
        # SAM expects sparse embeddings as [B, num_batches, num_points*2, embedding_dim]
        # We use 1 batch, 1 point (so 2 embeddings: point + label)
        # Duplicate the embedding to create point and label embeddings
        seg_sparse_embeddings = seg_sparse_embeddings.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 256]
        seg_sparse_embeddings = seg_sparse_embeddings.repeat(1, 1, 2, 1)  # [B, 1, 2, 256]
        
        # Get image positional embeddings for SAM
        # SAM expects [B, 256, 64, 64] format
        image_pe = self.sam_get_image_pe().repeat(batch_size, 1, 1, 1)  # [B, 256, 64, 64]
        
        # Dense embeddings should be [B, 256, 64, 64]
        dense_embeddings = torch.zeros(batch_size, 256, 64, 64, device=device, dtype=self.dtype)
        
        # Decode mask
        pred_masks, iou_predictions = self.sam_mask_decoder(
            image_embeddings=vision_features_sam,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=seg_sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
        return {
            'pred_masks': pred_masks,  # [B, 1, 256, 256]
            'iou_predictions': iou_predictions,  # [B, 1]
            'seg_embeddings': seg_embeddings  # [B, hidden_dim]
        }


# ===============================
# Loss Functions
# ===============================
class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class IoULoss(nn.Module):
    """IoU loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class LISACombinedLoss(nn.Module):
    """
    Simplified loss for ReasonSeg dataset (no text labels available):
    - Per-pixel BCE loss for masks
    - Dice loss for masks
    
    Note: Unlike full LISA, we skip text generation loss because ReasonSeg 
    dataset doesn't provide text completion labels. The VLM only generates
    <SEG> token to provide embeddings for SAM.
    """
    
    def __init__(self, bce_weight=2.0, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred_masks, target_masks):
        """
        Args:
            pred_masks: [B, H, W] - predicted masks (logits)
            target_masks: [B, H, W] - target masks
        """
        # Mask segmentation losses
        pred_probs = torch.sigmoid(pred_masks)
        
        # BCE loss (on logits)
        bce = self.bce_loss(pred_masks, target_masks)
        
        # Dice loss (on probabilities)
        dice = self.dice_loss(pred_probs, target_masks)
        
        # Combined loss
        total_loss = (
            self.bce_weight * bce +
            self.dice_weight * dice
        )
        
        return {
            'total': total_loss,
            'bce': bce,
            'dice': dice
        }


# ===============================
# Metrics
# ===============================
def compute_iou(pred, target, threshold=0.5):
    """Compute standard Intersection over Union"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    iou = intersection / (union + 1e-6)
    return iou.item()


def compute_ciou(pred, target, threshold=0.5):
    """
    Compute cumulative IoU (cIoU) - mean IoU across all samples
    This is the primary metric used in LISA for ReasonSeg
    """
    return compute_iou(pred, target, threshold)


def compute_giou(pred, target, threshold=0.5):
    """
    Compute Generalized IoU (gIoU) for masks
    gIoU = IoU - |C \\ (A ∪ B)| / |C|
    where C is the smallest enclosing box/region
    
    For segmentation masks, we compute the enclosing box and measure
    how much wasted space exists outside the union.
    """
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # Compute standard IoU
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    iou = intersection / (union + 1e-6)
    
    # Get bounding boxes in pixel coordinates
    pred_np = pred_binary.cpu().numpy().squeeze()
    target_np = target_binary.cpu().numpy().squeeze()
    
    pred_box = get_bounding_box(pred_np)
    target_box = get_bounding_box(target_np)
    
    # If either mask is empty, return IoU
    if pred_box is None or target_box is None:
        return iou.item()
    
    # Compute smallest enclosing box (in pixel coordinates: x_min, y_min, x_max, y_max)
    x1_c = min(pred_box[0], target_box[0])
    y1_c = min(pred_box[1], target_box[1])
    x2_c = max(pred_box[2], target_box[2])
    y2_c = max(pred_box[3], target_box[3])
    
    # Area of enclosing box (number of pixels)
    # Width = x2 - x1 + 1, Height = y2 - y1 + 1 (inclusive)
    c_area = (x2_c - x1_c + 1) * (y2_c - y1_c + 1)
    
    # Union area in pixels
    union_pixels = union.item()
    
    # GIoU formula: IoU - (area(C) - area(union)) / area(C)
    # This penalizes predictions where the enclosing box is much larger than union
    # GIoU should be in range [-1, 1] where 1 means perfect match
    giou = iou.item() - (c_area - union_pixels) / (c_area + 1e-6)
    
    return giou


def get_bounding_box(mask):
    """Extract bounding box from binary mask"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return [cmin, rmin, cmax, rmax]


def compute_pixel_metrics(pred, target, threshold=0.5):
    """
    Compute comprehensive pixel-wise metrics:
    - Accuracy: (TP + TN) / (TP + TN + FP + FN)
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    - AUROC: Area Under ROC Curve
    
    Args:
        pred: Predicted mask probabilities [0, 1]
        target: Ground truth binary mask {0, 1}
        threshold: Threshold for binary prediction
    
    Returns:
        Dictionary with all metrics
    """
    from sklearn.metrics import roc_auc_score
    
    pred_probs = pred.cpu().numpy().flatten()
    target_binary = target.cpu().numpy().flatten()
    pred_binary = (pred_probs > threshold).astype(np.float32)
    
    # True/False Positives/Negatives
    tp = ((pred_binary == 1) & (target_binary == 1)).sum()
    tn = ((pred_binary == 0) & (target_binary == 0)).sum()
    fp = ((pred_binary == 1) & (target_binary == 0)).sum()
    fn = ((pred_binary == 0) & (target_binary == 1)).sum()
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    
    # Precision
    precision = tp / (tp + fp + 1e-6)
    
    # Recall (Sensitivity)
    recall = tp / (tp + fn + 1e-6)
    
    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # AUROC (only if both classes present)
    try:
        if len(np.unique(target_binary)) > 1:
            auroc = roc_auc_score(target_binary, pred_probs)
        else:
            auroc = 1.0 if target_binary[0] == 1 else 0.0
    except:
        auroc = 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auroc': float(auroc)
    }


# ===============================
# Training Loop
# ===============================
def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    accelerator,
    epoch,
    args,
    vlm_processor,
    sam_processor,
    scheduler=None
):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_bce = 0
    total_dice = 0
    total_iou = 0
    total_ciou = 0
    total_giou = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_auroc = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)
    
    for batch in pbar:
        images = batch['image']
        conversations = batch['conversation']
        masks = batch['mask']
        
        device = accelerator.device
        
        # Process inputs for VLM
        vlm_inputs = vlm_processor(
            images=images,
            text=conversations,
            return_tensors='pt',
            padding=True
        ).to(device)
        
        # Process inputs for SAM
        sam_inputs = sam_processor(
            images=images,
            return_tensors='pt'
        ).to(device)
        
        # Prepare target masks
        target_masks = []
        for mask in masks:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            # Resize to SAM output size (256x256)
            mask_resized = F.interpolate(mask_tensor, size=(256, 256), mode='bilinear')
            target_masks.append(mask_resized)
        
        target_masks = torch.cat(target_masks, dim=0).to(device)
        
        # Forward pass
        outputs = model(vlm_inputs, sam_inputs, return_mask=True)
        pred_masks = outputs['pred_masks']  # [B, 1, 256, 256]
        
        # Squeeze to [B, 256, 256]
        while pred_masks.dim() > 3:
            pred_masks = pred_masks.squeeze(1)
        
        # Compute loss (mask-only, no text generation loss)
        loss_dict = criterion(
            pred_masks=pred_masks,
            target_masks=target_masks.squeeze(1)
        )
        loss = loss_dict['total']
        
        # Backward pass
        accelerator.backward(loss)
        
        if args.grad_clip > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Step scheduler if provided
        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()
        
        # Metrics
        with torch.no_grad():
            pred_probs = torch.sigmoid(pred_masks)
            batch_iou = compute_iou(pred_probs, target_masks.squeeze(1))
            batch_ciou = compute_ciou(pred_probs, target_masks.squeeze(1))
            batch_giou = compute_giou(pred_probs, target_masks.squeeze(1))
            pixel_metrics = compute_pixel_metrics(pred_probs, target_masks.squeeze(1))
        
        # Accumulate
        total_loss += loss.item()
        total_bce += loss_dict['bce'].item()
        total_dice += loss_dict['dice'].item()
        total_iou += batch_iou
        total_ciou += batch_ciou
        total_giou += batch_giou
        total_accuracy += pixel_metrics['accuracy']
        total_precision += pixel_metrics['precision']
        total_recall += pixel_metrics['recall']
        total_f1 += pixel_metrics['f1']
        total_auroc += pixel_metrics['auroc']
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'IoU': f"{batch_iou:.4f}",
            'acc': f"{pixel_metrics['accuracy']:.4f}",
            'f1': f"{pixel_metrics['f1']:.4f}"
        })
    
    # Average metrics
    metrics = {
        'loss': total_loss / num_batches,
        'bce': total_bce / num_batches,
        'dice': total_dice / num_batches,
        'IoU': total_iou / num_batches,
        'cIoU': total_ciou / num_batches,
        'gIoU': total_giou / num_batches,
        'accuracy': total_accuracy / num_batches,
        'precision': total_precision / num_batches,
        'recall': total_recall / num_batches,
        'f1': total_f1 / num_batches,
        'auroc': total_auroc / num_batches
    }
    
    return metrics


@torch.no_grad()
def validate(model, dataloader, criterion, accelerator, epoch, vlm_processor, sam_processor):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    total_iou = 0
    total_ciou = 0
    total_giou = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_auroc = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Validation {epoch}", disable=not accelerator.is_main_process)
    
    for batch in pbar:
        images = batch['image']
        conversations = batch['conversation']
        masks = batch['mask']
        
        device = accelerator.device
        
        # Process inputs for VLM
        vlm_inputs = vlm_processor(
            images=images,
            text=conversations,
            return_tensors='pt',
            padding=True
        ).to(device)
        
        # Process inputs for SAM
        sam_inputs = sam_processor(
            images=images,
            return_tensors='pt'
        ).to(device)
        
        # Prepare target masks
        target_masks = []
        for mask in masks:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_resized = F.interpolate(mask_tensor, size=(256, 256), mode='bilinear')
            target_masks.append(mask_resized)
        
        target_masks = torch.cat(target_masks, dim=0).to(device)
        
        # Forward pass
        outputs = model(vlm_inputs, sam_inputs, return_mask=True)
        pred_masks = outputs['pred_masks']
        
        # Squeeze to [B, 256, 256]
        while pred_masks.dim() > 3:
            pred_masks = pred_masks.squeeze(1)
        
        # Compute loss (mask-only)
        loss_dict = criterion(
            pred_masks=pred_masks,
            target_masks=target_masks.squeeze(1)
        )
        loss = loss_dict['total']
        
        # Metrics
        pred_probs = torch.sigmoid(pred_masks)
        batch_iou = compute_iou(pred_probs, target_masks.squeeze(1))
        batch_ciou = compute_ciou(pred_probs, target_masks.squeeze(1))
        batch_giou = compute_giou(pred_probs, target_masks.squeeze(1))
        pixel_metrics = compute_pixel_metrics(pred_probs, target_masks.squeeze(1))
        
        total_loss += loss.item()
        total_iou += batch_iou
        total_ciou += batch_ciou
        total_giou += batch_giou
        total_accuracy += pixel_metrics['accuracy']
        total_precision += pixel_metrics['precision']
        total_recall += pixel_metrics['recall']
        total_f1 += pixel_metrics['f1']
        total_auroc += pixel_metrics['auroc']
        num_batches += 1
        
        pbar.set_postfix({
            'val_loss': f"{loss.item():.4f}",
            'val_IoU': f"{batch_iou:.4f}",
            'val_acc': f"{pixel_metrics['accuracy']:.4f}",
            'val_f1': f"{pixel_metrics['f1']:.4f}"
        })
    
    metrics = {
        'val_loss': total_loss / num_batches,
        'val_IoU': total_iou / num_batches,
        'val_cIoU': total_ciou / num_batches,
        'val_gIoU': total_giou / num_batches,
        'val_accuracy': total_accuracy / num_batches,
        'val_precision': total_precision / num_batches,
        'val_recall': total_recall / num_batches,
        'val_f1': total_f1 / num_batches,
        'val_auroc': total_auroc / num_batches
    }
    
    return metrics


def benchmark_memory_and_time(model, dataloader, accelerator, vlm_processor, sam_processor):
    """Benchmark memory usage and inference time"""
    model.eval()
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:  # Benchmark on 10 batches
                break
            
            images = batch['image']
            conversations = batch['conversation']
            device = accelerator.device
            
            # Process inputs
            vlm_inputs = vlm_processor(
                images=images,
                text=conversations,
                return_tensors='pt',
                padding=True
            ).to(device)
            
            sam_inputs = sam_processor(
                images=images,
                return_tensors='pt'
            ).to(device)
            
            _ = model(vlm_inputs, sam_inputs, return_mask=True)
    
    end_time = time.time()
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    else:
        peak_memory = 0
    
    avg_time = (end_time - start_time) / min(10, len(dataloader))
    
    return {
        'peak_memory_gb': peak_memory,
        'avg_inference_time_s': avg_time
    }


def plot_training_history(history, output_dir):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary with 'train' and 'val' lists of metric dictionaries
        output_dir: Directory to save plots
    """
    train_history = history['train']
    val_history = history['val']
    
    epochs = range(1, len(train_history) + 1)
    
    # Create figure with subplots - 3x3 grid for comprehensive metrics
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Modified LISA Training History (ReasonSeg)', fontsize=16, fontweight='bold')
    
    # Row 1: Losses
    # Plot total loss
    axes[0, 0].plot(epochs, [m['loss'] for m in train_history], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, [m['val_loss'] for m in val_history], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss (BCE + Dice)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot BCE loss
    axes[0, 1].plot(epochs, [m['bce'] for m in train_history], 'b-', label='Train BCE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('BCE Loss')
    axes[0, 1].set_title('Mask BCE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot Dice loss
    axes[0, 2].plot(epochs, [m['dice'] for m in train_history], 'b-', label='Train Dice', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Dice Loss')
    axes[0, 2].set_title('Mask Dice Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: IoU Metrics
    # Plot IoU, cIoU and gIoU
    axes[1, 0].plot(epochs, [m['IoU'] for m in train_history], 'g-', label='Train IoU', linewidth=2)
    axes[1, 0].plot(epochs, [m['val_IoU'] for m in val_history], 'g--', label='Val IoU', linewidth=2)
    axes[1, 0].plot(epochs, [m['cIoU'] for m in train_history], 'b-', label='Train cIoU', linewidth=2)
    axes[1, 0].plot(epochs, [m['val_cIoU'] for m in val_history], 'b--', label='Val cIoU', linewidth=2)
    axes[1, 0].plot(epochs, [m['gIoU'] for m in train_history], 'r-', label='Train gIoU', linewidth=2)
    axes[1, 0].plot(epochs, [m['val_gIoU'] for m in val_history], 'r--', label='Val gIoU', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU Metrics')
    axes[1, 0].set_title('IoU, cIoU and gIoU')
    axes[1, 0].legend(loc='best', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[1, 1].plot(epochs, [m['accuracy'] for m in train_history], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, [m['val_accuracy'] for m in val_history], 'r-', label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Pixel-wise Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot F1-Score
    axes[1, 2].plot(epochs, [m['f1'] for m in train_history], 'b-', label='Train', linewidth=2)
    axes[1, 2].plot(epochs, [m['val_f1'] for m in val_history], 'r-', label='Val', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('F1-Score')
    axes[1, 2].set_title('Pixel-wise F1-Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Row 3: Precision, Recall, AUROC
    # Plot Precision
    axes[2, 0].plot(epochs, [m['precision'] for m in train_history], 'b-', label='Train', linewidth=2)
    axes[2, 0].plot(epochs, [m['val_precision'] for m in val_history], 'r-', label='Val', linewidth=2)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Precision')
    axes[2, 0].set_title('Pixel-wise Precision')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot Recall
    axes[2, 1].plot(epochs, [m['recall'] for m in train_history], 'b-', label='Train', linewidth=2)
    axes[2, 1].plot(epochs, [m['val_recall'] for m in val_history], 'r-', label='Val', linewidth=2)
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Recall')
    axes[2, 1].set_title('Pixel-wise Recall')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Plot AUROC
    axes[2, 2].plot(epochs, [m['auroc'] for m in train_history], 'b-', label='Train', linewidth=2)
    axes[2, 2].plot(epochs, [m['val_auroc'] for m in val_history], 'r-', label='Val', linewidth=2)
    axes[2, 2].set_xlabel('Epoch')
    axes[2, 2].set_ylabel('AUROC')
    axes[2, 2].set_title('Pixel-wise AUROC')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining plots saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Modified LISA Model')
    
    # Model arguments
    parser.add_argument('--vlm_name', type=str, default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--sam_name', type=str, default='facebook/sam-vit-base')
    parser.add_argument('--use_vlm_vision', action='store_true', default=True, help='Use VLM vision encoder (default: True)')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--no_quantization', action='store_true', help='Disable 4-bit quantization (requires more memory)')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    
    # Loss weights (simplified for ReasonSeg - no text labels)
    parser.add_argument('--bce_weight', type=float, default=2.0, help='Weight for BCE loss (mask segmentation)')
    parser.add_argument('--dice_weight', type=float, default=0.5, help='Weight for Dice loss (mask segmentation)')
    
    # Component-level training control
    parser.add_argument('--train_vlm_vision', action='store_true', help='Train VLM vision encoder')
    parser.add_argument('--train_vlm_projector', action='store_true', default=True, help='Train VLM projector')
    parser.add_argument('--train_vlm_llm', action='store_true', default=True, help='Train VLM LLM')
    parser.add_argument('--train_sam_vision', action='store_true', help='Train SAM vision encoder')
    parser.add_argument('--train_sam_prompt_encoder', action='store_true', help='Train SAM prompt encoder')
    parser.add_argument('--train_sam_mask_decoder', action='store_true', default=True, help='Train SAM mask decoder')
    parser.add_argument('--train_projection_layers', action='store_true', default=True, help='Train projection layers')
    
    # LoRA arguments
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    
    # Learning rate scheduler
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'linear', 'plateau', 'none'], help='Learning rate scheduler type')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping (epochs without improvement)')
    parser.add_argument('--plateau_patience', type=int, default=5, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--plateau_factor', type=float, default=0.5, help='Factor for ReduceLROnPlateau')
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_best_only', action='store_true', default=True, help='Only save best model')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision='bf16' if args.dtype == 'bf16' else 'fp16' if args.dtype == 'fp16' else 'no')
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if accelerator.is_main_process:
        print("="*80)
        print("Modified LISA Training")
        print("="*80)
        print(f"VLM: {args.vlm_name}")
        print(f"SAM: {args.sam_name}")
        print(f"Vision Encoder: {'VLM' if args.use_vlm_vision else 'SAM'}")
        print(f"Dtype: {args.dtype}")
        print(f"Use LoRA: {args.use_lora}")
        if args.use_lora:
            print(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Learning Rate: {args.lr}")
        print(f"Scheduler: {args.scheduler}")
        print(f"Epochs: {args.epochs}")
        print(f"Early Stopping Patience: {args.early_stopping_patience}")
        print("="*80)
    
    # Initialize model
    if accelerator.is_main_process:
        print("\nInitializing Modified LISA model...")
    
    model = ModifiedLISA(
        vlm_name=args.vlm_name,
        sam_name=args.sam_name,
        use_vlm_vision_encoder=args.use_vlm_vision,
        dtype=args.dtype,
        use_quantization=not args.no_quantization,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # Component-level training control
        train_vlm_vision=args.train_vlm_vision,
        train_vlm_projector=args.train_vlm_projector,
        train_vlm_llm=args.train_vlm_llm,
        train_sam_vision=args.train_sam_vision,
        train_sam_prompt_encoder=args.train_sam_prompt_encoder,
        train_sam_mask_decoder=args.train_sam_mask_decoder,
        train_projection_layers=args.train_projection_layers
    )
    
    # Get processors from model for dataset preprocessing
    vlm_processor = model.vlm_processor
    sam_processor = model.sam_processor
    
    # Load datasets with processors
    train_dataset = ReasonSegDataset(
        args.data_dir,
        split='train',
        max_samples=args.max_train_samples,
        vlm_processor=vlm_processor,
        sam_processor=sam_processor
    )
    val_dataset = ReasonSegDataset(
        args.data_dir,
        split='val',
        max_samples=args.max_val_samples,
        vlm_processor=vlm_processor,
        sam_processor=sam_processor
    )
    
    # Custom collate function to handle preprocessed data
    def collate_fn(batch):
        return {
            'image': [item['image'] for item in batch],
            'text': [item['text'] for item in batch],
            'conversation': [item['conversation'] for item in batch],
            'mask': [item['mask'] for item in batch],
            'name': [item['name'] for item in batch]
        }
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Loss and optimizer (simplified - no text generation loss)
    criterion = LISACombinedLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight
    )
    
    # Collect all trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if accelerator.is_main_process:
        # Count unique parameters to avoid double-counting shared weights
        def count_unique_params(module):
            seen_params = set()
            total = 0
            for p in module.parameters():
                param_id = id(p)
                if param_id not in seen_params:
                    seen_params.add(param_id)
                    total += p.numel()
            return total
        
        total_params = count_unique_params(model)
        trainable_param_count = sum(p.numel() for p in trainable_params)
        print(f"\nTotal unique parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"Trainable parameters: {trainable_param_count:,} ({100*trainable_param_count/total_params:.2f}%)")
        print(f"Frozen parameters: {total_params - trainable_param_count:,} ({100*(total_params - trainable_param_count)/total_params:.2f}%)")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = None
    
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
    elif args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            verbose=True
        )
    
    if accelerator.is_main_process and scheduler is not None:
        print(f"Using {args.scheduler} learning rate scheduler")
    
    # Prepare with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Prepare scheduler if not None
    if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
        scheduler = accelerator.prepare(scheduler)
    
    # Training history
    history = {
        'train': [],
        'val': [],
        'benchmarks': {}
    }
    
    best_val_ciou = 0
    epochs_without_improvement = 0
    
    if accelerator.is_main_process:
        print("\nStarting training...")
        print("="*80)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        if accelerator.is_main_process:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, accelerator, epoch, args,
            vlm_processor, sam_processor, scheduler
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, accelerator, epoch,
            vlm_processor, sam_processor
        )
        
        # Step schedulers that depend on validation metrics
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_metrics['val_cIoU'])
        elif scheduler is not None and isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        
        # Log metrics
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"    - BCE Loss: {train_metrics['bce']:.4f}")
            print(f"    - Dice Loss: {train_metrics['dice']:.4f}")
            print(f"  Train Metrics:")
            print(f"    - IoU: {train_metrics['IoU']:.4f}")
            print(f"    - cIoU: {train_metrics['cIoU']:.4f}")
            print(f"    - gIoU: {train_metrics['gIoU']:.4f}")
            print(f"    - Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"    - Precision: {train_metrics['precision']:.4f}")
            print(f"    - Recall: {train_metrics['recall']:.4f}")
            print(f"    - F1-Score: {train_metrics['f1']:.4f}")
            print(f"    - AUROC: {train_metrics['auroc']:.4f}")
            print(f"  Val Metrics:")
            print(f"    - Loss: {val_metrics['val_loss']:.4f}")
            print(f"    - IoU: {val_metrics['val_IoU']:.4f}")
            print(f"    - cIoU: {val_metrics['val_cIoU']:.4f}")
            print(f"    - gIoU: {val_metrics['val_gIoU']:.4f}")
            print(f"    - Accuracy: {val_metrics['val_accuracy']:.4f}")
            print(f"    - Precision: {val_metrics['val_precision']:.4f}")
            print(f"    - Recall: {val_metrics['val_recall']:.4f}")
            print(f"    - F1-Score: {val_metrics['val_f1']:.4f}")
            print(f"    - AUROC: {val_metrics['val_auroc']:.4f}")
            
            history['train'].append(train_metrics)
            history['val'].append(val_metrics)
            
            # Save metrics
            with open(os.path.join(args.output_dir, f'metrics_epoch_{epoch}.json'), 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'train': train_metrics,
                    'val': val_metrics
                }, f, indent=2)
            
            # Check for improvement
            if val_metrics['val_cIoU'] > best_val_ciou:
                best_val_ciou = val_metrics['val_cIoU']
                epochs_without_improvement = 0
                
                if args.save_best_only:
                    # Remove old best model if exists
                    old_checkpoints = list(Path(args.checkpoint_dir).glob('best_model_*.pt'))
                    for ckpt in old_checkpoints:
                        ckpt.unlink()
                
                checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_epoch_{epoch}.pt')
                
                # Unwrap model for saving
                unwrapped_model = accelerator.unwrap_model(model)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'val_ciou': val_metrics['val_cIoU'],
                    'val_giou': val_metrics['val_gIoU'],
                    'args': vars(args)
                }, checkpoint_path)
                
                print(f"  ✓ Saved best model (cIoU: {best_val_ciou:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} epoch(s)")
                
                # Early stopping
                if epochs_without_improvement >= args.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(f"Best validation cIoU: {best_val_ciou:.4f}")
                    break
            
            # Save history
            with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f, indent=2)
    
    # Final benchmarking and plotting
    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("Running final benchmarks...")
        print("="*80)
        
        benchmark_results = benchmark_memory_and_time(
            accelerator.unwrap_model(model),
            val_loader,
            accelerator,
            vlm_processor,
            sam_processor
        )
        
        print(f"\nBenchmark Results:")
        print(f"  Peak Memory: {benchmark_results['peak_memory_gb']:.2f} GB")
        print(f"  Avg Inference Time: {benchmark_results['avg_inference_time_s']:.4f} s")
        
        history['benchmarks'] = benchmark_results
        
        # Save final history
        with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training history
        print("\nGenerating training plots...")
        plot_training_history(history, args.output_dir)
        
        print("\n" + "="*80)
        print("Training completed!")
        print(f"Best validation cIoU: {best_val_ciou:.4f}")
        print("="*80)


if __name__ == '__main__':
    main()
