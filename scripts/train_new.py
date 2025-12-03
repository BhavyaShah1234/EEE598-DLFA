import os
import json
import time
import argparse
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from dataclasses import dataclass
import peft as p
import torch as t
import typing as ty
import transformers as tf
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import torchvision.transforms as T
from accelerate import Accelerator
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ImageEncoder(t.nn.Module):
    """
    Wraps a pretrained vision encoder (CLIP, DINOv2, etc.) to output both sequential and spatial features from a single forward pass.
    """
    def __init__(
            self,
            device: str,
            dtype: t.dtype,
            model_name: str,
            use_quantization: bool,
            quantization: str,
            freeze: bool,
            use_lora: bool,
            lora_r: int,
            lora_alpha: int,
            lora_dropout: float,
            lora_target_modules: list,
        ):
        super(ImageEncoder, self).__init__()
        # Determine dtype based on mixed precision settings
        self.device = device
        self.dtype = dtype
        # 1. (Quantization OR torch dtype) → 2. LoRA / Freeze
        quantization_config = None
        if use_quantization:
            if quantization == "4bit":
                quantization_config = tf.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print(f"  → Vision quantization: 4-bit (compute={dtype})")
            elif quantization == "8bit":
                quantization_config = tf.BitsAndBytesConfig(load_in_8bit=True)
                print("  → Vision quantization: 8-bit")
            else:
                print(f"  ! Unknown vision quantization '{quantization}' ignored")
        load_kwargs = {"device_map": "auto"}
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            # Only set dtype when NOT quantized
            load_kwargs["dtype"] = dtype
        self.vision_model = tf.CLIPVisionModel.from_pretrained(model_name, **load_kwargs)
        # self.vision_model = self.vision_model.to(device)
        self.config = self.vision_model.config
        self.embed_dim = self.config.hidden_size
        self.patch_size = self.config.patch_size
        self.image_size = self.config.image_size
        # Apply freeze or LoRA
        if freeze:
            print("  → Vision mode: Frozen")
            self.vision_model.requires_grad_(False)
        elif use_lora:
            print(f"  → Vision mode: LoRA (r={lora_r}, alpha={lora_alpha})")
            # Omit task_type to avoid PEFT mislabeling inputs for vision backbone
            lora_config = p.LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.vision_model = p.get_peft_model(self.vision_model, lora_config)
            self.vision_model.print_trainable_parameters()

    def forward(self, pixel_values: t.Tensor) -> ty.Tuple[t.Tensor, t.Tensor]:
        """
        Args:
            pixel_values: [B, 3, H, W] - preprocessed images
        Returns:
            features_seq: [B, num_patches+1, embed_dim] - includes CLS token
            spatial_features: [B, embed_dim, H_feat, W_feat] - spatial grid
        """
        outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=True)
        features_seq = outputs.last_hidden_state
        patch_features = features_seq[:, 1:, :]#.unsqueeze(1) if features_seq.dim() == 3 and features_seq.size(1) == 1 else features_seq[:, 1:, :]
        batch_size = patch_features.shape[0]
        num_patches = patch_features.shape[1]
        H_feat = W_feat = int(num_patches ** 0.5)
        spatial_features = patch_features.transpose(1, 2).reshape(batch_size, self.embed_dim, H_feat, W_feat)
        return features_seq, spatial_features

class TextEncoder(t.nn.Module):
    """
    Wraps a pretrained LLM (LLaMA, Vicuna, etc.) for text encoding.
    """
    def __init__(
            self,
            device: str,
            dtype: t.dtype,
            model_name: str,
            use_quantization: bool,
            quantization: str,
            freeze: bool,
            use_lora: bool,
            lora_r: int,
            lora_alpha,
            lora_dropout,
            lora_target_modules: list,
        ):
        super(TextEncoder, self).__init__()
        self.device = device
        self.dtype = dtype
        quantization_config = None
        if use_quantization:
            if quantization == "4bit":
                quantization_config = tf.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print(f"  → LLM quantization: 4-bit (compute={dtype})")
            elif quantization == "8bit":
                quantization_config = tf.BitsAndBytesConfig(load_in_8bit=True)
                print("  → LLM quantization: 8-bit")
            else:
                print(f"  ! Unknown LLM quantization '{quantization}' ignored")
        load_kwargs = {"device_map": "auto"}
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["dtype"] = dtype
        self.llm = tf.AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embed_dim = self.llm.config.hidden_size
        # Apply freeze or LoRA
        if freeze:
            print("  → LLM mode: Frozen")
            self.llm.requires_grad_(False)
        elif use_lora:
            print(f"  → LLM mode: LoRA (r={lora_r}, alpha={lora_alpha})")
            lora_config = p.LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=p.TaskType.CAUSAL_LM,
            )
            self.llm = p.get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()

    def forward(self, input_ids: t.Tensor, attention_mask: ty.Optional[t.Tensor] = None) -> t.Tensor:
        """
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
        Returns:
            text_features: [B, seq_len, embed_dim]
        """
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # CausalLM models (e.g., GPT2LMHeadModel) return CausalLMOutput* without last_hidden_state.
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if outputs.hidden_states is None:
            raise RuntimeError("LLM forward did not return hidden_states; ensure output_hidden_states=True.")
        return outputs.hidden_states[-1]

class VLM(t.nn.Module):
    """
    Vision-Language Model wrapper supporting models like LLaVA, SMoLVLM, QwenVL, etc.
    
    Provides unified interface for:
    - Image encoding
    - Text encoding
    - Multimodal fusion
    """
    def __init__(
            self,
            device: str,
            dtype: t.dtype,
            model_name: str,
            use_quantization: bool,
            quantization: str,
            freeze: bool,
            use_lora: bool,
            lora_r: int,
            lora_alpha: int,
            lora_dropout: float,
            lora_target_modules: list,
        ):
        super(VLM, self).__init__()
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        
        # Determine quantization config
        quantization_config = None
        if use_quantization:
            if quantization == "4bit":
                quantization_config = tf.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print(f"  → VLM quantization: 4-bit (compute={dtype})")
            elif quantization == "8bit":
                quantization_config = tf.BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                print("  → VLM quantization: 8-bit (with CPU offload)")
            else:
                print(f"  ! Unknown VLM quantization '{quantization}' ignored")

        # Load VLM based on model type
        load_kwargs = {"device_map": "auto"}
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["dtype"] = dtype
        
        # Detect VLM type and load appropriate model
        model_name_lower = model_name.lower()

        if "llava" in model_name_lower:
            # LLaVA models
            self.vlm = tf.LlavaForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
            self.processor = tf.AutoProcessor.from_pretrained(model_name)
            self.vision_tower = self.vlm.vision_tower
            self.language_model = self.vlm.language_model
            self.multi_modal_projector = self.vlm.multi_modal_projector
            self.vision_embed_dim = self.vlm.config.vision_config.hidden_size
            self.text_embed_dim = self.vlm.config.text_config.hidden_size

        elif "smol" in model_name_lower or "smolvlm" in model_name_lower:
            # SMoLVLM models
            self.vlm = tf.AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)
            self.processor = tf.AutoProcessor.from_pretrained(model_name)
            # Get vision and text components
            if hasattr(self.vlm, 'vision_model'):
                self.vision_tower = self.vlm.vision_model
            elif hasattr(self.vlm, 'vision_encoder'):
                self.vision_tower = self.vlm.vision_encoder
            else:
                self.vision_tower = None
                
            if hasattr(self.vlm, 'language_model'):
                self.language_model = self.vlm.language_model
            elif hasattr(self.vlm, 'text_decoder'):
                self.language_model = self.vlm.text_decoder
            else:
                self.language_model = self.vlm

            # Get embedding dimensions
            self.vision_embed_dim = getattr(self.vlm.config, 'vision_config', self.vlm.config).hidden_size
            self.text_embed_dim = getattr(self.vlm.config, 'text_config', self.vlm.config).hidden_size
            
        elif "qwen" in model_name_lower and ("vl" in model_name_lower or "vision" in model_name_lower):
            # QwenVL models
            self.vlm = tf.Qwen2VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
            self.processor = tf.AutoProcessor.from_pretrained(model_name)
            self.vision_tower = self.vlm.visual if hasattr(self.vlm, 'visual') else None
            self.language_model = self.vlm.model if hasattr(self.vlm, 'model') else self.vlm
            self.vision_embed_dim = self.vlm.config.vision_config.hidden_size if hasattr(self.vlm.config, 'vision_config') else 1024
            self.text_embed_dim = self.vlm.config.hidden_size
        elif "paligemma" in model_name_lower or "gemma-2" in model_name_lower:
            # PaliGemma / Gemma vision models
            self.vlm = tf.PaliGemmaForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
            self.processor = tf.AutoProcessor.from_pretrained(model_name)
            self.vision_tower = self.vlm.vision_tower
            self.language_model = self.vlm.language_model
            self.multi_modal_projector = self.vlm.multi_modal_projector
            self.vision_embed_dim = self.vlm.config.vision_config.hidden_size
            self.text_embed_dim = self.vlm.config.text_config.hidden_size
        elif "gemma-3" in model_name_lower:
            self.vlm = tf.Gemma3ForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
            self.processor = tf.AutoProcessor.from_pretrained(model_name)
            self.vision_tower = self.vlm.vision_tower
            self.language_model = self.vlm.language_model
            self.multi_modal_projector = self.vlm.multi_modal_projector
            self.vision_embed_dim = self.vlm.config.vision_config.hidden_size
            self.text_embed_dim = self.vlm.config.text_config.hidden_size
        else:
            # Generic VLM - try AutoModel
            try:
                self.vlm = tf.AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)
                self.processor = tf.AutoProcessor.from_pretrained(model_name)
            except:
                self.vlm = tf.AutoModel.from_pretrained(model_name, **load_kwargs)
                self.processor = tf.AutoProcessor.from_pretrained(model_name)
            
            # Try to find vision and text components
            self.vision_tower = getattr(self.vlm, 'vision_model', getattr(self.vlm, 'vision_tower', None))
            self.language_model = getattr(self.vlm, 'language_model', getattr(self.vlm, 'text_model', self.vlm))
            
            # Get embedding dimensions
            config = self.vlm.config
            self.vision_embed_dim = getattr(getattr(config, 'vision_config', config), 'hidden_size', 768)
            self.text_embed_dim = getattr(getattr(config, 'text_config', config), 'hidden_size', 2048)
        
        print(f"  ✓ VLM loaded: {model_name}")
        print(f"    Vision dim: {self.vision_embed_dim}, Text dim: {self.text_embed_dim}")
        
        # Store tokenizer for compatibility
        if hasattr(self.processor, 'tokenizer'):
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

        # Enable gradient checkpointing
        if hasattr(self.vlm, 'gradient_checkpointing_enable'):
            self.vlm.gradient_checkpointing_enable()
            print("  → VLM gradient checkpointing enabled")

        # Apply freeze or LoRA
        if freeze:
            print("  → VLM mode: Frozen")
            self.vlm.requires_grad_(False)
        elif use_lora:
            print(f"  → VLM mode: LoRA (r={lora_r}, alpha={lora_alpha})")
            lora_config = p.LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.vlm = p.get_peft_model(self.vlm, lora_config)
            self.vlm.print_trainable_parameters()

    def encode_image(self, pixel_values: t.Tensor) -> ty.Tuple[t.Tensor, t.Tensor]:
        """
        Encode images to get both sequential and spatial features
        
        Args:
            pixel_values: [B, 3, H, W] or [B, num_images, 3, H, W] (for Idefics3)
        Returns:
            features_seq: [B, num_patches, vision_embed_dim]
            spatial_features: [B, vision_embed_dim, H_feat, W_feat]
        """
        if self.vision_tower is not None:
            # Use vision tower if available (LLaVA, PaliGemma)
            vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
            features_seq = vision_outputs.last_hidden_state
        else:
            # For models without separate vision tower (Idefics3/SmolVLM)
            # Extract vision features from the model's vision encoder
            if hasattr(self.vlm, 'model') and hasattr(self.vlm.model, 'vision_model'):
                # Idefics3 architecture - handle different pixel_values shapes
                # The processor gives us [B, num_images, C, H, W] (5D)
                # But vision_model embeddings expects [B, C, H, W] (4D)
                # We need to squeeze out the num_images dimension
                if pixel_values.dim() == 5:
                    # [B, num_images, C, H, W] -> [B*num_images, C, H, W]
                    batch_size, num_images, c, h, w = pixel_values.shape
                    pixel_values = pixel_values.reshape(batch_size * num_images, c, h, w)
                    vision_outputs = self.vlm.model.vision_model(pixel_values)
                    features_seq = vision_outputs.last_hidden_state
                    # Reshape back: [B*num_images, num_patches, dim] -> [B, num_images, num_patches, dim]
                    num_patches = features_seq.shape[1]
                    dim = features_seq.shape[2]
                    features_seq = features_seq.reshape(batch_size, num_images, num_patches, dim)
                elif pixel_values.dim() == 4:
                    # [B, C, H, W] - single image per batch
                    vision_outputs = self.vlm.model.vision_model(pixel_values)
                    features_seq = vision_outputs.last_hidden_state
                else:
                    raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")
            elif hasattr(self.vlm, 'vision_model'):
                vision_outputs = self.vlm.vision_model(pixel_values)
                features_seq = vision_outputs.last_hidden_state
            else:
                # Last resort: create dummy features with correct dimensions
                batch_size = pixel_values.shape[0]
                num_patches = 196  # Standard ViT patch count
                features_seq = t.zeros(batch_size, num_patches, self.vision_embed_dim, device=pixel_values.device, dtype=pixel_values.dtype)

        # Handle Idefics3's 5D output [B, num_images, num_patches, dim]
        if features_seq.dim() == 4:
            # Reshape [B, num_images, num_patches, dim] -> [B, num_images*num_patches, dim]
            batch_size, num_images, num_patches, dim = features_seq.shape
            features_seq = features_seq.reshape(batch_size, num_images * num_patches, dim)
        
        # Convert to spatial features
        # Remove CLS token if present
        if features_seq.shape[1] > 1:
            patch_features = features_seq[:, 1:, :] if features_seq.shape[1] > 196 else features_seq
        else:
            patch_features = features_seq
            
        batch_size = patch_features.shape[0]
        num_patches = patch_features.shape[1]
        H_feat = W_feat = int(num_patches ** 0.5)
        
        if H_feat * W_feat != num_patches:
            # Handle non-square patch grids
            H_feat = int(num_patches ** 0.5)
            W_feat = num_patches // H_feat
            patch_features = patch_features[:, :H_feat*W_feat, :]
        
        spatial_features = patch_features.transpose(1, 2).reshape(batch_size, self.vision_embed_dim, H_feat, W_feat)
        
        return features_seq, spatial_features
    
    def encode_text(self, input_ids: t.Tensor, attention_mask: ty.Optional[t.Tensor] = None) -> t.Tensor:
        """
        Encode text to get text embeddings
        
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
        Returns:
            text_features: [B, seq_len, text_embed_dim]
        """
        if self.language_model is not None:
            outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        else:
            # Use full VLM
            outputs = self.vlm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if outputs.hidden_states is None:
            raise RuntimeError("VLM text encoding did not return hidden_states")
        return outputs.hidden_states[-1]
    
    def forward(self, pixel_values: t.Tensor, input_ids: t.Tensor, attention_mask: ty.Optional[t.Tensor] = None, return_dict: bool = True):
        """
        Full VLM forward pass for multimodal generation
        
        Args:
            pixel_values: [B, 3, H, W] or [B, num_images, 3, H, W]
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            return_dict: Whether to return ModelOutput dict
        Returns:
            outputs: VLM outputs with logits, hidden_states, and image_hidden_states
        """
        # Full multimodal forward pass - VLM handles image-text fusion internally
        return self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict
        )

class SegmentAnythingModel(t.nn.Module):
    """
    Wraps SAM's pretrained mask decoder to use shared image embeddings.
    """
    def __init__(
            self,
            device: str,
            dtype: t.dtype,
            sam_model_name: str,
            use_quantization: bool,
            quantization: str,
            freeze: bool,
            use_lora: bool,
            lora_r: int,
            lora_alpha: int,
            lora_dropout: float,
            lora_target_modules: list,
        ):
        super(SegmentAnythingModel, self).__init__()
        self.device = device
        self.dtype = dtype
        quantization_config = None
        if use_quantization:
            if quantization == "4bit":
                quantization_config = tf.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print(f"  → SAM quantization: 4-bit (compute={dtype})")
            elif quantization == "8bit":
                quantization_config = tf.BitsAndBytesConfig(load_in_8bit=True)
                print("  → SAM quantization: 8-bit")
            else:
                print(f"  ! Unknown SAM quantization '{quantization}' ignored")
        print(f"  → Loading pretrained SAM: {sam_model_name}")
        load_kwargs = {"device_map": "auto"}
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["dtype"] = dtype
        self.sam = tf.SamModel.from_pretrained(sam_model_name, **load_kwargs)
        # Apply freeze or LoRA
        if freeze:
            print("  → SAM mode: Frozen")
            self.sam.requires_grad_(False)
        elif use_lora:
            print(f"  → SAM mode: LoRA (r={lora_r}, alpha={lora_alpha})")
            # Omit task_type for SAM to prevent incorrect input mapping
            lora_config = p.LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.sam = p.get_peft_model(self.sam, lora_config)
            self.sam.print_trainable_parameters()
        self.prompt_encoder = self.sam.prompt_encoder
        self.mask_decoder = self.sam.mask_decoder
        # Determine image embedding channel dimension for adapter usage.
        if self.prompt_encoder.no_mask_embed.weight.ndim == 2:
            self.sam_embed_dim = self.prompt_encoder.no_mask_embed.weight.shape[1]
        else:
            self.sam_embed_dim = getattr(getattr(self.sam.config, 'vision_config', self.sam.config), 'hidden_size', 256)

    def forward(self, image_embeddings: t.Tensor, prompt_embeddings: t.Tensor, multimask_output: bool = True) -> ty.Tuple[t.Tensor, t.Tensor]:
        """
        Args:
            image_embeddings: [B, sam_embed_dim, 64, 64]
            prompt_embeddings: [B, num_prompts, 256]
        Returns:
            masks: [B, num_masks, 256, 256]
            iou_predictions: [B, num_masks]
        """
        # target_dtype = self.param_dtype
        # target_device = self.runtime_device
        image_embeddings = image_embeddings.to(device=self.device, dtype=self.dtype)
        prompt_embeddings = prompt_embeddings.to(device=self.device, dtype=self.dtype)
        batch_size = image_embeddings.shape[0]
        dense_embeddings = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(batch_size, -1, image_embeddings.shape[-2], image_embeddings.shape[-1]).to(device=self.device, dtype=self.dtype)
        image_positional_embeddings = self.sam.get_image_wide_positional_embeddings().repeat(batch_size, 1, 1, 1).to(device=self.device, dtype=self.dtype)
        sparse_prompt_embeddings = prompt_embeddings.unsqueeze(2).to(device=self.device, dtype=self.dtype)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        # SAM returns masks shape: [B, point_batch, num_masks, H, W]
        batch_size, P, M, height, width = low_res_masks.shape
        # Flatten point and mask dimensions for downstream simplicity
        flat_masks = low_res_masks.reshape(batch_size, P * M, height, width)
        # Typically base model outputs 256x256 already; interpolate only if different
        # Flatten IoU predictions similarly: [B, point_batch, num_masks] -> [B, point_batch*num_masks]
        iou_predictions = iou_predictions.reshape(iou_predictions.shape[0], -1)
        return flat_masks, iou_predictions

class ImageEncoderTextEncoderConnector(t.nn.Module):
    """
    Projects vision features to LLM dimension for multimodal fusion. Adds optional dtype casting so inputs and weights match under mixed precision.
    """
    def __init__(
            self,
            device: str,
            dtype: t.dtype,
            image_encoder_dim: int,
            hidden_dim: int,
            text_encoder_dim: int,
            num_layers: int,
        ):
        super(ImageEncoderTextEncoderConnector, self).__init__()
        self.device = device
        self.dtype = dtype
        modules = []
        if num_layers > 1:
            for i in range(num_layers):
                if i == 0:
                    modules.append(t.nn.Linear(image_encoder_dim, hidden_dim, dtype=dtype, device=device))
                else:
                    modules.append(t.nn.Linear(hidden_dim, text_encoder_dim, dtype=dtype, device=device))
                if i < num_layers - 1:
                    modules.append(t.nn.GELU())
        else:
            modules = [t.nn.Linear(image_encoder_dim, text_encoder_dim, dtype=dtype, device=device)]
        self.projector = t.nn.Sequential(*modules)

    def forward(self, vision_features: t.Tensor) -> t.Tensor:
        vision_features = vision_features.to(device=self.device, dtype=self.dtype)
        return self.projector(vision_features)

class TextEncoderSAMConnector(t.nn.Module):
    """
    Extracts task-specific prompt embeddings from fused VLM features. Supports mixed precision by casting internal parameters to requested dtype.
    """
    def __init__(
            self,
            device: str,
            dtype: t.dtype,
            text_encoder_dim: int,
            hidden_dim: int,
            sam_dim: int,
            num_prompt_tokens: int,
        ):
        super(TextEncoderSAMConnector, self).__init__()
        self.device = device
        self.dtype = dtype
        self.prompt_queries = t.nn.Parameter(t.randn(1, num_prompt_tokens, text_encoder_dim, dtype=dtype, device=device) * 0.02)
        self.cross_attn = t.nn.MultiheadAttention(text_encoder_dim, num_heads=8, batch_first=True, dtype=dtype, device=device)
        self.norm = t.nn.LayerNorm(text_encoder_dim, dtype=dtype, device=device)
        self.projector = t.nn.Sequential(
            t.nn.Linear(text_encoder_dim, hidden_dim, dtype=dtype, device=device),
            t.nn.GELU(),
            t.nn.Linear(hidden_dim, sam_dim, dtype=dtype, device=device),
        )#.to(device=device, dtype=dtype)

    def forward(self, vlm_features: t.Tensor) -> t.Tensor:
        vlm_features = vlm_features.to(device=self.device, dtype=self.dtype)
        batch_size = vlm_features.shape[0]
        queries = self.prompt_queries.expand(batch_size, -1, -1)
        prompts, _ = self.cross_attn(queries, vlm_features, vlm_features)
        prompts = self.norm(prompts + queries)
        prompt_embeddings = self.projector(prompts)
        return prompt_embeddings

class ImageEncoderSAMConnector(t.nn.Module):
    """
    Adapts the shared vision encoder features to SAM's expected format.
    """
    def __init__(
            self,
            device: str,
            dtype: t.dtype,
            image_encoder_dim: int,
            sam_dim: int,
            target_spatial_size: ty.Tuple[int, int],
        ):
        super(ImageEncoderSAMConnector, self).__init__()
        self.device = device
        self.dtype = dtype
        self.target_spatial_size = target_spatial_size
        # Choose a valid number of groups for GroupNorm: must divide sam_embed_dim.
        max_groups = 32 if sam_dim >= 32 else sam_dim
        groups = max_groups
        while groups > 1 and sam_dim % groups != 0:
            groups -= 1
        self.projection = t.nn.Sequential(
            t.nn.Conv2d(image_encoder_dim, sam_dim, kernel_size=1, dtype=dtype, device=device),
            t.nn.GroupNorm(groups, sam_dim, dtype=dtype, device=device),
            t.nn.Conv2d(sam_dim, sam_dim, kernel_size=3, padding=1, dtype=dtype, device=device),
            t.nn.GroupNorm(groups, sam_dim, dtype=dtype, device=device),
        )

    def forward(self, vision_features: t.Tensor) -> t.Tensor:
        """
        Args:
            vision_features: [B, input_dim, H, W]
        Returns:
            adapted_features: [B, sam_embed_dim, target_H, target_W]
        """
        vision_features = vision_features.to(device=self.device, dtype=self.dtype)
        features = self.projection(vision_features)
        features = t.nn.functional.interpolate(features, size=self.target_spatial_size, mode='bilinear', align_corners=False)
        return features

class ModifiedLISA(t.nn.Module):
    """
    Modified LISA using ALL pretrained HuggingFace models including SAM.
    
    Key Features:
    - Single image encoding (no redundancy)
    - Configurable quantization (4-bit, 8-bit)
    - Mixed precision support (bf16, fp16)
    - Flexible training modes (freeze, LoRA, full training)
    """
    def __init__(
            self,
            device: str,
            use_vlm: bool = False,
            vlm_model_name: str = None,
            vlm_use_mixed_precision: bool = False,
            vlm_mixed_precision: str = 'no',
            vlm_use_quantization: bool = False,
            vlm_quantization: str = '8bit',
            vlm_freeze: bool = False,
            vlm_use_lora: bool = False,
            vlm_lora_r: int = 16,
            vlm_lora_alpha: int = 32,
            vlm_lora_dropout: float = 0.1,
            vlm_lora_target_modules: list = None,
            image_encoder_model_name: str = 'openai/clip-vit-base-patch16',
            image_encoder_use_mixed_precision: bool = False,
            image_encoder_mixed_precision: str = 'no',
            image_encoder_use_quantization: bool = False,
            image_encoder_quantization: str = '8bit',
            image_encoder_freeze: bool = False,
            image_encoder_use_lora: bool = False,
            image_encoder_lora_r: int = 16,
            image_encoder_lora_target_modules: list = None,
            image_encoder_lora_alpha: int = 32,
            image_encoder_lora_dropout: float = 0.1,
            text_encoder_model_name: str = 'meta-llama/Llama-3.2-1B',
            text_encoder_use_mixed_precision: bool = False,
            text_encoder_mixed_precision: str = 'no',
            text_encoder_use_quantization: bool = False,
            text_encoder_quantization: str = '8bit',
            text_encoder_freeze: bool = False,
            text_encoder_use_lora: bool = False,
            text_encoder_lora_r: int = 16,
            text_encoder_lora_alpha: int = 32,
            text_encoder_lora_dropout: float = 0.1,
            text_encoder_lora_target_modules: list = None,
            sam_model_name: str = 'facebook/sam-vit-base',
            sam_use_mixed_precision: bool = False,
            sam_mixed_precision: str = 'no',
            sam_use_quantization: bool = False,
            sam_quantization: str = '8bit',
            sam_freeze: bool = False,
            sam_use_lora: bool = False,
            sam_lora_r: int = 16,
            sam_lora_alpha: int = 32,
            sam_lora_target_modules: list = None,
            sam_lora_dropout: float = 0.1,
            image_text_connector_use_mixed_precision: bool = False,
            image_text_connector_mixed_precision: str = 'no',
            image_text_connector_num_layers: int = 2,
            text_sam_connector_use_mixed_precision: bool = False,
            text_sam_connector_mixed_precision: str = 'no',
            text_sam_connector_tokens: int = 8,
            image_sam_connector_use_mixed_precision: bool = False,
            image_sam_connector_mixed_precision: str = 'no',
        ):
        super(ModifiedLISA, self).__init__()
        self.device = device
        self.use_vlm = use_vlm
        
        # Set default VLM target modules if not provided
        if vlm_lora_target_modules is None:
            vlm_lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        
        if use_vlm:
            # VLM Mode - use a single vision-language model
            print("Using VLM mode")
            vlm_dtype = t.float32
            if vlm_use_mixed_precision:
                if vlm_mixed_precision == 'bf16':
                    vlm_dtype = t.bfloat16
                elif vlm_mixed_precision == 'fp16':
                    vlm_dtype = t.float16
            
            self.vlm = VLM(
                device=device,
                dtype=vlm_dtype,
                model_name=vlm_model_name,
                use_quantization=vlm_use_quantization,
                quantization=vlm_quantization,
                freeze=vlm_freeze,
                use_lora=vlm_use_lora,
                lora_r=vlm_lora_r,
                lora_alpha=vlm_lora_alpha,
                lora_dropout=vlm_lora_dropout,
                lora_target_modules=vlm_lora_target_modules,
            )
            
            # Warn if VLM is fully trainable without quantization (high memory usage)
            if not vlm_freeze and not vlm_use_quantization and not vlm_use_lora:
                print("⚠️  WARNING: VLM is fully trainable without quantization!")
                print("   This requires significant GPU memory (>20GB for 2B+ models)")
                print("   Consider using --vlm_use_quantization or --vlm_freeze or --vlm_use_lora")
            
            # Use VLM's embedding dimensions
            image_encoder_dim = self.vlm.vision_embed_dim
            text_encoder_dim = self.vlm.text_embed_dim
            self.vision_model_name = vlm_model_name
            self.llm_model_name = vlm_model_name
            
            # Store references for compatibility
            self.image_encoder = None
            self.text_encoder = None
            
        else:
            # Traditional Mode - separate image and text encoders
            print("Using traditional mode (separate encoders)")
            self.vlm = None
            self.vision_model_name = image_encoder_model_name
            self.llm_model_name = text_encoder_model_name

            image_encoder_dtype = t.float32
            if image_encoder_use_mixed_precision:
                if image_encoder_mixed_precision == 'bf16':
                    image_encoder_dtype = t.bfloat16
                elif image_encoder_mixed_precision == 'fp16':
                    image_encoder_dtype = t.float16
            self.image_encoder_dtype = image_encoder_dtype
            # Shared Image Encoder (CLIP)
            self.image_encoder = ImageEncoder(
                device=device,
                dtype=image_encoder_dtype,
                model_name=image_encoder_model_name,
                use_quantization=image_encoder_use_quantization,
                quantization=image_encoder_quantization,
                freeze=image_encoder_freeze,
                use_lora=image_encoder_use_lora,
                lora_r=image_encoder_lora_r,
                lora_alpha=image_encoder_lora_alpha,
                lora_dropout=image_encoder_lora_dropout,
                lora_target_modules=image_encoder_lora_target_modules,
            )
            image_encoder_dim = self.image_encoder.embed_dim
            print(f"  ✓ CLIP loaded. Embedding dim: {image_encoder_dim}")

            # Text Encoder (LLM)
            text_encoder_dtype = t.float32
            if text_encoder_use_mixed_precision:
                if text_encoder_mixed_precision == 'bf16':
                    text_encoder_dtype = t.bfloat16
                elif text_encoder_mixed_precision == 'fp16':
                    text_encoder_dtype = t.float16
            self.text_encoder_dtype = text_encoder_dtype
            self.text_encoder = TextEncoder(
                device=device,
                dtype=text_encoder_dtype,
                model_name=text_encoder_model_name,
                use_quantization=text_encoder_use_quantization,
                quantization=text_encoder_quantization,
                freeze=text_encoder_freeze,
                use_lora=text_encoder_use_lora,
                lora_r=text_encoder_lora_r,
                lora_alpha=text_encoder_lora_alpha,
                lora_dropout=text_encoder_lora_dropout,
                lora_target_modules=text_encoder_lora_target_modules,
            )
            text_encoder_dim = self.text_encoder.embed_dim
            print(f"  ✓ LLM loaded. Embedding dim: {text_encoder_dim}")

        # SAM
        sam_dtype = t.float32
        if sam_use_mixed_precision:
            if sam_mixed_precision == 'bf16':
                sam_dtype = t.bfloat16
            elif sam_mixed_precision == 'fp16':
                sam_dtype = t.float16
        self.sam_decoder = SegmentAnythingModel(
            device=device,
            dtype=sam_dtype,
            sam_model_name=sam_model_name,
            use_quantization=sam_use_quantization,
            quantization=sam_quantization,
            freeze=sam_freeze,
            use_lora=sam_use_lora,
            lora_r=sam_lora_r,
            lora_alpha=sam_lora_alpha,
            lora_dropout=sam_lora_dropout,
            lora_target_modules=sam_lora_target_modules,
        )
        sam_dim = self.sam_decoder.sam_embed_dim
        print(f"  ✓ SAM created. Embedding dim: {sam_dim}")

        # Projector
        image_text_connector_dtype = t.float32
        if image_text_connector_use_mixed_precision:
            if image_text_connector_mixed_precision == 'bf16':
                image_text_connector_dtype = t.bfloat16
            elif image_text_connector_mixed_precision == 'fp16':
                image_text_connector_dtype = t.float16
        self.image_text_connector = ImageEncoderTextEncoderConnector(
            device=device,
            dtype=image_text_connector_dtype,
            image_encoder_dim=image_encoder_dim,
            hidden_dim=(image_encoder_dim + text_encoder_dim) // 2,
            text_encoder_dim=text_encoder_dim,
            num_layers=image_text_connector_num_layers,
        )
        print("  ✓ Projector created")

        text_sam_connector_dtype = t.float32
        if text_sam_connector_use_mixed_precision:
            if text_sam_connector_mixed_precision == 'bf16':
                text_sam_connector_dtype = t.bfloat16
            elif text_sam_connector_mixed_precision == 'fp16':
                text_sam_connector_dtype = t.float16
        self.text_sam_connector = TextEncoderSAMConnector(
            device=device,
            dtype=text_sam_connector_dtype,
            text_encoder_dim=text_encoder_dim,
            hidden_dim=text_encoder_dim // 2,
            sam_dim=sam_dim,
            num_prompt_tokens=text_sam_connector_tokens,
        )
        print("  ✓ Prompt extractor created")

        image_sam_connector_dtype = t.float32
        if image_sam_connector_use_mixed_precision:
            if image_sam_connector_mixed_precision == 'bf16':
                image_sam_connector_dtype = t.bfloat16
            elif image_sam_connector_mixed_precision == 'fp16':
                image_sam_connector_dtype = t.float16
        
        # For VLM mode, vision features are projected to text_embed_dim
        # For traditional mode, they remain at image_encoder_dim
        vision_spatial_dim = text_encoder_dim if use_vlm else image_encoder_dim
        
        self.image_sam_connector = ImageEncoderSAMConnector(
            device=device,
            dtype=image_sam_connector_dtype,
            image_encoder_dim=vision_spatial_dim,
            sam_dim=sam_dim,
            target_spatial_size=(64, 64),
        )
        print("  ✓ SAM adapter created")

    def forward(self, pixel_values: t.Tensor, input_ids: t.Tensor, attention_mask: ty.Optional[t.Tensor] = None, multimask_output: bool = True) -> ty.Tuple[t.Tensor, t.Tensor]:
        """
        Args:
            pixel_values: [B, 3, H, W] or [B, num_images, 3, H, W]
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            multimask_output: whether to output multiple masks
        Returns:
            masks: [B, num_masks, 256, 256]
            iou_predictions: [B, num_masks]
        """
        if self.use_vlm:
            # VLM Mode: Use full multimodal forward pass (critical for proper fusion)
            # The VLM internally fuses image and text through its architecture
            vlm_outputs = self.vlm.vlm(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract features from VLM's multimodal processing
            if hasattr(vlm_outputs, 'image_hidden_states') and vlm_outputs.image_hidden_states is not None:
                # Use projected image features from VLM (already aligned with text space)
                # This is the output from multi_modal_projector
                vision_features_seq = vlm_outputs.image_hidden_states  # [B, num_image_tokens, text_embed_dim]
                
                # Convert to spatial format for SAM
                # Ensure we have batch dimension
                if vision_features_seq.dim() == 2:
                    vision_features_seq = vision_features_seq.unsqueeze(0)
                
                batch_size = vision_features_seq.shape[0]
                num_patches = vision_features_seq.shape[1]
                embed_dim = vision_features_seq.shape[2]
                H_feat = W_feat = int(num_patches ** 0.5)
                
                if H_feat * W_feat != num_patches:
                    H_feat = int(num_patches ** 0.5)
                    W_feat = num_patches // H_feat
                    vision_features_seq_trimmed = vision_features_seq[:, :H_feat*W_feat, :]
                else:
                    vision_features_seq_trimmed = vision_features_seq
                
                # Reshape to spatial: [B, num_patches, dim] -> [B, dim, H, W]
                vision_features_spatial = vision_features_seq_trimmed.permute(0, 2, 1).reshape(
                    batch_size, embed_dim, H_feat, W_feat
                )
                
                # Get fused multimodal features from last hidden state
                # This contains both image and text information after VLM processing
                if hasattr(vlm_outputs, 'hidden_states') and vlm_outputs.hidden_states:
                    fused_features = vlm_outputs.hidden_states[-1]  # [B, total_seq_len, text_embed_dim]
                else:
                    # Fallback: manually concatenate
                    text_embeddings = self.vlm.language_model.embed_tokens(input_ids)
                    fused_features = t.cat([vision_features_seq, text_embeddings], dim=1)
                    
            else:
                # Fallback for VLMs without image_hidden_states
                # This path is less ideal as it breaks multimodal fusion
                vision_features_seq, vision_features_spatial = self.vlm.encode_image(pixel_values)
                text_features = self.vlm.encode_text(input_ids, attention_mask)
                
                # Project vision to text space and concatenate
                vision_features_projected = self.image_text_connector(vision_features_seq)
                fused_features = t.cat([vision_features_projected, text_features], dim=1)
        else:
            # Traditional Mode: Use separate image and text encoders
            vision_features_seq, vision_features_spatial = self.image_encoder(pixel_values)
            text_features = self.text_encoder(input_ids, attention_mask)

            # Project vision features to LLM space
            vision_features_projected = self.image_text_connector(vision_features_seq)
            
            # Fuse vision and language
            fused_features = t.cat([vision_features_projected, text_features], dim=1)

        # Step 5: Extract prompt embeddings
        prompt_embeddings = self.text_sam_connector(fused_features)

        # Step 6a: Adapt vision features to SAM's format
        sam_image_embeddings = self.image_sam_connector(vision_features_spatial)

        # Step 6b: Decode masks using pretrained SAM
        masks, iou_predictions = self.sam_decoder(sam_image_embeddings, prompt_embeddings, multimask_output=multimask_output)

        return masks, iou_predictions

    def prepare_inputs(self, images: list, prompts: list):
        """
        Helper function to prepare inputs
        
        Args:
            images: List of PIL Images
            prompts: List of text prompts
        Returns:
            pixel_values, input_ids, attention_mask
        """
        if self.use_vlm:
            # Use VLM's processor
            processor = self.vlm.processor
            # Some VLM processors handle both image and text
            try:
                inputs = processor(images=images, text=prompts, padding=True, truncation=True, return_tensors="pt")
                pixel_values = inputs.get("pixel_values", inputs.get("images"))
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask")
            except:
                # Fallback: process separately
                pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
                tokenizer = self.vlm.tokenizer
                text_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
                input_ids = text_inputs["input_ids"]
                attention_mask = text_inputs.get("attention_mask")
        else:
            # Traditional mode
            processor = tf.CLIPImageProcessor.from_pretrained(self.vision_model_name)
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
            tokenizer = self.text_encoder.tokenizer
            text_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
            input_ids = text_inputs["input_ids"]
            attention_mask = text_inputs.get("attention_mask")
            
        return pixel_values.to(self.device), input_ids.to(self.device), attention_mask.to(self.device) if attention_mask is not None else None

    def print_trainable_parameters(self):
        """Print the number of trainable parameters in the model"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"\nTrainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.2f}%")

class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve"""
    def __init__(self, patience=7, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like IoU (higher is better), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
            return False
            
        if self.mode == 'max':
            improvement = metric - self.best_score
        else:
            improvement = self.best_score - metric
            
        if improvement > self.min_delta:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop

@dataclass
class TrainingConfig:
    # Paths
    data_root: str = "/home/bhavya-shah/Projects/EEE598-DLFA"
    train_json: str = "train.json"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"

    # Training hyperparameters (optimized for 8GB GPU)
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0

    # Model config
    img_size: int = 256
    max_text_length: int = 128
    max_train_samples: int = None  # Limit for faster testing
    max_val_samples: int = None  # Limit for faster testing

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Model names
    image_encoder_model_name: str = 'openai/clip-vit-base-patch16'
    text_encoder_model_name: str = 'meta-llama/Llama-3.2-1B'
    sam_model_name: str = 'facebook/sam-vit-base'

    # Mixed precision settings
    use_mixed_precision: bool = False
    mixed_precision: str = "no"

    # Quantization settings
    use_quantization: bool = False
    quantization: str = "8bit"

    # Training modes
    freeze_image_encoder: bool = False
    freeze_text_encoder: bool = False
    freeze_sam: bool = False
    use_lora: bool = True

class ReasonSegDataset(Dataset):
    """Dataset for ReasonSeg"""
    def __init__(self, data_root, json_file=None, image_dir=None, img_size=224, max_samples=None):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.samples = []
        
        if json_file:
            # Load from train.json
            json_path = self.data_root / json_file
            with open(json_path, 'r') as f:
                data = json.load(f)
            for item in data:
                img_path = self.data_root / image_dir / item['image']
                json_path = self.data_root / image_dir / item['json']
                if img_path.exists() and json_path.exists():
                    self.samples.append({
                        'image_path': img_path,
                        'json_path': json_path,
                        'query': item['query']
                    })
        elif image_dir:
            # Load from directory (val/test)
            img_dir = self.data_root / image_dir
            for json_file in img_dir.glob("*.json"):
                img_file = json_file.with_suffix('.jpg')
                if img_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    query = data['text'][0] if 'text' in data else ""
                    self.samples.append({
                        'image_path': img_file,
                        'json_path': json_file,
                        'query': query
                    })
        
        # Limit samples if specified
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Load mask from JSON
        with open(sample['json_path'], 'r') as f:
            data = json.load(f)
        
        # Create mask from polygon points
        orig_w, orig_h = image.size
        mask_pil = Image.new('L', (orig_w, orig_h), 0)
        draw = ImageDraw.Draw(mask_pil)
        
        if 'shapes' in data and len(data['shapes']) > 0:
            for shape in data['shapes']:
                if 'points' in shape:
                    points = [(float(x), float(y)) for x, y in shape['points']]
                    draw.polygon(points, fill=255)
        
        # Resize mask to match img_size
        mask_pil = mask_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
        mask_tensor = t.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        
        return {
            'image': image,
            'query': sample['query'],
            'mask': mask_tensor,
            'image_path': str(sample['image_path'])
        }

def collate_fn(batch):
    """Custom collate function for batching"""
    images = [item['image'] for item in batch]  # Keep as PIL images
    queries = [item['query'] for item in batch]
    masks = t.stack([item['mask'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'queries': queries,
        'masks': masks,
        'image_paths': image_paths
    }

def compute_iou(pred_mask, gt_mask, threshold=0.5):
    """Compute Intersection over Union"""
    pred_mask = (pred_mask > threshold).float()
    gt_mask = (gt_mask > threshold).float()
    
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()

def intersectionAndUnion(pred_mask, gt_mask, num_classes=2, threshold=0.5):
    """
    Compute intersection and union for multi-class segmentation.
    Following LISA's implementation for 2-class (background=0, foreground=1).
    
    Args:
        pred_mask: Predicted mask (H, W) with continuous values
        gt_mask: Ground truth mask (H, W) with binary values
        num_classes: Number of classes (default=2 for binary segmentation)
        threshold: Threshold for converting predictions to binary
    
    Returns:
        intersection: Intersection area for each class [num_classes]
        union: Union area for each class [num_classes]
    """
    pred_mask = (pred_mask > threshold).long().view(-1)
    gt_mask = (gt_mask > threshold).long().view(-1)
    
    # Compute intersection: count pixels where pred == gt for each class
    intersection = pred_mask[pred_mask == gt_mask]
    area_intersection = t.histc(intersection.float(), bins=num_classes, min=0, max=num_classes - 1)
    
    # Compute areas
    area_pred = t.histc(pred_mask.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_gt = t.histc(gt_mask.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_union = area_pred + area_gt - area_intersection
    
    return area_intersection, area_union

def dice_loss(pred_masks, gt_masks, smooth=1.0):
    """Dice loss for segmentation - LISA L_dice"""
    pred_flat = pred_masks.view(pred_masks.size(0), -1)
    gt_flat = gt_masks.view(gt_masks.size(0), -1)
    
    # Apply sigmoid for dice calculation
    pred_sigmoid = t.sigmoid(pred_flat)
    
    intersection = (pred_sigmoid * gt_flat).sum(dim=1)
    union = pred_sigmoid.sum(dim=1) + gt_flat.sum(dim=1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def bce_loss(pred_masks, gt_masks):
    """Binary Cross-Entropy loss for segmentation - LISA L_bce"""
    return t.nn.functional.binary_cross_entropy_with_logits(pred_masks, gt_masks, reduction='mean')

def text_generation_loss(text_outputs, target_ids, attention_mask=None):
    """Text generation loss - LISA L_txt (auto-regressive cross-entropy)"""
    if text_outputs is None:
        return t.tensor(0.0, device=target_ids.device)
    
    # Shift for autoregressive prediction
    shift_logits = text_outputs[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()
    
    # Flatten for cross-entropy
    loss_fct = t.nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    return loss

def lisa_loss(pred_masks, gt_masks, text_outputs=None, target_ids=None, 
              lambda_txt=1.0, lambda_mask=1.0, lambda_bce=2.0, lambda_dice=0.5):
    """
    LISA training loss: L_total = λ_txt * L_txt + λ_mask * L_mask
    where L_mask = λ_bce * L_bce + λ_dice * L_dice
    
    Args:
        pred_masks: Predicted segmentation masks [B, 1, H, W]
        gt_masks: Ground truth masks [B, 1, H, W]
        text_outputs: Text generation logits [B, seq_len, vocab_size] (optional)
        target_ids: Target text token IDs [B, seq_len] (optional)
        lambda_txt: Weight for text generation loss (default: 1.0)
        lambda_mask: Weight for mask loss (default: 1.0)
        lambda_bce: Weight for BCE loss within mask loss (default: 2.0)
        lambda_dice: Weight for Dice loss within mask loss (default: 0.5)
    
    Returns:
        total_loss: Weighted combination of text and mask losses
        loss_dict: Dictionary with individual loss components
    """
    # Resize predicted masks to match ground truth
    if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
        pred_masks = t.nn.functional.interpolate(
            pred_masks, 
            size=gt_masks.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )

    # If multiple masks predicted, take the first one
    if pred_masks.dim() == 4 and pred_masks.size(1) > 1:
        pred_masks = pred_masks[:, 0:1, :, :]
    
    # L_txt: Text generation loss (auto-regressive cross-entropy)
    L_txt = text_generation_loss(text_outputs, target_ids) if text_outputs is not None else t.tensor(0.0, device=pred_masks.device)
    
    # L_mask: Mask loss (BCE + Dice)
    L_bce = bce_loss(pred_masks, gt_masks)
    L_dice = dice_loss(pred_masks, gt_masks)
    L_mask = lambda_bce * L_bce + lambda_dice * L_dice
    
    # L_total: Combined loss
    L_total = lambda_txt * L_txt + lambda_mask * L_mask
    
    loss_dict = {
        'total': L_total.item(),
        'text': L_txt.item() if isinstance(L_txt, t.Tensor) else 0.0,
        'mask': L_mask.item(),
        'bce': L_bce.item(),
        'dice': L_dice.item()
    }
    
    return L_total, loss_dict

def focal_loss(pred_masks, gt_masks, alpha=0.25, gamma=2.0):
    """Focal loss for segmentation (legacy, not used in LISA)"""
    # Use BCE with logits (safe for autocast)
    bce = t.nn.functional.binary_cross_entropy_with_logits(pred_masks, gt_masks, reduction='none')
    
    # Get probabilities for focal term
    pred_probs = t.sigmoid(pred_masks)
    pt = t.where(gt_masks == 1, pred_probs, 1 - pred_probs)
    focal_term = (1 - pt) ** gamma
    loss = alpha * focal_term * bce
    return loss.mean()

def combined_loss(pred_masks, gt_masks, iou_preds=None, 
                 lambda_txt=0.0, lambda_mask=1.0, lambda_bce=2.0, lambda_dice=0.5):
    """
    Combined loss wrapper for LISA-style training
    Kept for backwards compatibility - calls lisa_loss internally
    """
    loss, loss_dict = lisa_loss(pred_masks, gt_masks, text_outputs=None, target_ids=None,
                                lambda_txt=lambda_txt, lambda_mask=lambda_mask,
                                lambda_bce=lambda_bce, lambda_dice=lambda_dice)
    return loss

class BenchmarkMetrics:
    """Track benchmarking metrics following LISA's approach"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.iou_scores = []  # Instance-level IoU scores
        self.intersection_sum = t.zeros(2)  # [bg, fg] cumulative intersection
        self.union_sum = t.zeros(2)  # [bg, fg] cumulative union
        self.acc_iou_sum = 0.0  # Accumulator for per-instance foreground IoU (gIoU)
        self.num_samples = 0
        self.losses = []
        self.iteration_times = []
        self.memory_usage = []
        self.start_time = None
        self.epoch_start_time = None
    
    def update(self, loss, pred_masks, gt_masks, iter_time):
        self.losses.append(loss)
        self.iteration_times.append(iter_time)
        
        # Resize predictions to match ground truth
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            pred_masks = t.nn.functional.interpolate(
                pred_masks, 
                size=gt_masks.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Take first mask if multiple
        if pred_masks.dim() == 4 and pred_masks.size(1) > 1:
            pred_masks = pred_masks[:, 0, :, :]
        else:
            pred_masks = pred_masks.squeeze(1)
        
        pred_masks = t.sigmoid(pred_masks)
        
        # Compute metrics for each sample in batch (LISA-style)
        batch_size = pred_masks.size(0)
        for i in range(batch_size):
            # Instance-level IoU for reporting
            iou = compute_iou(pred_masks[i], gt_masks[i, 0])
            self.iou_scores.append(iou)
            
            # Class-wise intersection/union for cIoU (LISA approach)
            intersection, union = intersectionAndUnion(pred_masks[i], gt_masks[i, 0], num_classes=2)
            self.intersection_sum += intersection.cpu()
            self.union_sum += union.cpu()
            
            # Per-instance foreground IoU for gIoU
            if union[1] > 0:
                instance_fg_iou = (intersection[1] / union[1]).item()
                self.acc_iou_sum += instance_fg_iou
            else:
                # No object in ground truth - count as perfect if prediction is also empty
                self.acc_iou_sum += 1.0 if intersection[1] == 0 else 0.0
            
            self.num_samples += 1
        
        # Track memory
        if t.cuda.is_available():
            mem = t.cuda.max_memory_allocated() / 1024**3  # GB
            self.memory_usage.append(mem)
    
    def get_metrics(self):
        # cIoU: Class IoU for foreground = total_intersection[fg] / total_union[fg]
        ciou = (self.intersection_sum[1] / (self.union_sum[1] + 1e-10)).item() if self.num_samples > 0 else 0.0
        
        # gIoU: Average per-instance foreground IoU  
        giou = (self.acc_iou_sum / self.num_samples) if self.num_samples > 0 else 0.0
        
        return {
            'loss': np.mean(self.losses) if self.losses else 0.0,
            'iou': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'giou': giou,
            'ciou': ciou,
            'iter_per_sec': 1.0 / np.mean(self.iteration_times) if self.iteration_times else 0.0,
            'avg_iter_time': np.mean(self.iteration_times) if self.iteration_times else 0.0,
            'peak_memory_gb': max(self.memory_usage) if self.memory_usage else 0.0,
        }

def prepare_vlm_inputs(processor, queries, images, max_length, device):
    """Prepare inputs for VLM based on processor type"""
    # Check if processor requires <image> tokens (Idefics3/SmolVLM)
    processor_name = processor.__class__.__name__.lower()
    
    if 'llava' in processor_name:
        # LLaVA: requires <image> token in text
        formatted_queries = [f"<image>\n{q}" for q in queries]
        inputs = processor(
            text=formatted_queries,
            images=images,
            padding=True,
            return_tensors="pt"
        )
    elif 'idefics' in processor_name:
        # Idefics3/SmolVLM: requires <image> token in text
        # Don't truncate to preserve image tokens
        formatted_queries = [f"<image>{q}" for q in queries]
        inputs = processor(
            text=formatted_queries,
            images=images,
            padding=True,
            truncation=False,  # Disable truncation to preserve image tokens
            return_tensors="pt"
        )
    elif 'paligemma' in processor_name:
        # PaliGemma: requires <image> token prefix for each image
        formatted_queries = [f"<image>{q}" for q in queries]
        inputs = processor(
            text=formatted_queries,
            images=images,
            padding=True,
            return_tensors="pt"
        )
    elif 'gemma' in processor_name:
        # Gemma3 (non-PaliGemma): use chat template for proper image token formatting
        # Expects images as list of lists (one list per text, supporting multiple images per text)
        import torch as t
        
        # Format all queries with chat template
        all_formatted_texts = []
        for query in queries:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
            formatted_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            all_formatted_texts.append(formatted_text)
        
        # Process all at once - wrap each image in a list for Gemma3's format
        images_as_lists = [[img] for img in images]
        inputs = processor(
            text=all_formatted_texts,
            images=images_as_lists,  # List of lists: [[img1], [img2], ...]
            padding=True,
            truncation=False,  # Disable truncation to preserve image tokens
            return_tensors="pt"
        )
    else:
        # Qwen: standard format
        inputs = processor(
            text=queries,
            images=images,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    return {
        "pixel_values": inputs["pixel_values"].to(device),
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device)
    }

def train_epoch(model, dataloader, optimizer, accelerator, metrics, args):
    """Train for one epoch"""
    model.train()
    metrics.reset()
    metrics.epoch_start_time = time.time()

    # Get processor and tokenizer based on mode
    if hasattr(model, 'module'):
        unwrapped_model = model.module
    else:
        unwrapped_model = model

    if unwrapped_model.use_vlm:
        # VLM mode: use VLM's processor
        processor = unwrapped_model.vlm.processor
    else:
        # Traditional mode: use CLIP processor and text encoder tokenizer
        image_processor = tf.CLIPImageProcessor.from_pretrained(args.image_encoder_model_name)
        tokenizer = unwrapped_model.text_encoder.tokenizer

    for step, batch in enumerate(dataloader):
        iter_start = time.time()
        # Prepare inputs
        images = batch['images']  # Already PIL images
        queries = batch['queries']
        gt_masks = batch['masks']

        if unwrapped_model.use_vlm:
            # VLM mode: process images and text together
            vlm_inputs = prepare_vlm_inputs(processor, queries, images, args.max_text_length, accelerator.device)
            pixel_values = vlm_inputs["pixel_values"]
            input_ids = vlm_inputs["input_ids"]
            attention_mask = vlm_inputs["attention_mask"]
        else:
            # Traditional mode: process separately
            pixel_values = image_processor(images=images, return_tensors="pt")["pixel_values"]
            pixel_values = pixel_values.to(accelerator.device)
            
            text_inputs = tokenizer(
                queries, 
                padding=True, 
                truncation=True, 
                max_length=args.max_text_length,
                return_tensors="pt"
            )
            input_ids = text_inputs["input_ids"].to(accelerator.device)
            attention_mask = text_inputs["attention_mask"].to(accelerator.device)
        
        gt_masks = gt_masks.to(accelerator.device)

        # Forward pass
        with accelerator.autocast():
            pred_masks, iou_preds = model(pixel_values, input_ids, attention_mask, multimask_output=False)
            loss = combined_loss(pred_masks, gt_masks, iou_preds)
        
        # Backward pass
        accelerator.backward(loss)
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Skip gradient clipping with FP16 due to accelerate limitations
            # if accelerator.sync_gradients and config.max_grad_norm > 0:
            #     accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        iter_time = time.time() - iter_start
        
        # Update metrics
        metrics.update(loss.item(), pred_masks.detach(), gt_masks, iter_time)
        
        # if step % 20 == 0:
        #     current_metrics = metrics.get_metrics()
        #     accelerator.print(
        #         f"Step {step}/{len(dataloader)} | "
        #         f"Loss: {current_metrics['loss']:.4f} | "
        #         f"IoU: {current_metrics['iou']:.4f} | "
        #         f"gIoU: {current_metrics['giou']:.4f} | "
        #         f"cIoU: {current_metrics['ciou']:.4f} | "
        #         f"Iter/s: {current_metrics['iter_per_sec']:.2f} | "
        #         f"Mem: {current_metrics['peak_memory_gb']:.2f}GB"
        #     )

    epoch_time = time.time() - metrics.epoch_start_time
    epoch_metrics = metrics.get_metrics()
    epoch_metrics['epoch_time'] = epoch_time
    
    return epoch_metrics

def validate(model, dataloader, accelerator, args):
    """Validate the model"""
    model.eval()
    metrics = BenchmarkMetrics()
    
    # Get processor and tokenizer based on mode
    if hasattr(model, 'module'):
        unwrapped_model = model.module
    else:
        unwrapped_model = model
    
    if unwrapped_model.use_vlm:
        # VLM mode: use VLM's processor
        processor = unwrapped_model.vlm.processor
    else:
        # Traditional mode: use CLIP processor and text encoder tokenizer
        image_processor = tf.CLIPImageProcessor.from_pretrained(args.image_encoder_model_name)
        tokenizer = unwrapped_model.text_encoder.tokenizer
    
    with t.no_grad():
        for batch in dataloader:
            iter_start = time.time()
            
            images = batch['images']  # Already PIL images
            queries = batch['queries']
            gt_masks = batch['masks']
            
            if unwrapped_model.use_vlm:
                # VLM mode: process images and text together
                vlm_inputs = prepare_vlm_inputs(processor, queries, images, args.max_text_length, accelerator.device)
                pixel_values = vlm_inputs["pixel_values"]
                input_ids = vlm_inputs["input_ids"]
                attention_mask = vlm_inputs["attention_mask"]
            else:
                # Traditional mode: process separately
                pixel_values = image_processor(images=images, return_tensors="pt")["pixel_values"]
                pixel_values = pixel_values.to(accelerator.device)
                text_inputs = tokenizer(
                    queries, 
                    padding=True, 
                    truncation=True,
                    max_length=args.max_text_length,
                    return_tensors="pt"
                )
                input_ids = text_inputs["input_ids"].to(accelerator.device)
                attention_mask = text_inputs["attention_mask"].to(accelerator.device)
            
            gt_masks = gt_masks.to(accelerator.device)            # Forward pass
            with accelerator.autocast():
                pred_masks, iou_preds = model(pixel_values, input_ids, attention_mask, multimask_output=False)
                loss = combined_loss(pred_masks, gt_masks, iou_preds)
            
            iter_time = time.time() - iter_start
            metrics.update(loss.item(), pred_masks, gt_masks, iter_time)
    
    return metrics.get_metrics()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ModifiedLISA on ReasonSeg dataset')

    # Data paths
    parser.add_argument('--data_root', type=str, default='/home/bhavya-shah/Projects/EEE598-DLFA', help='Root directory containing train/val/test folders')
    parser.add_argument('--train_json', type=str, default='train.json', help='Training JSON file name')
    parser.add_argument('--train_dir', type=str, default='train', help='Training images directory')
    parser.add_argument('--val_dir', type=str, default='val', help='Validation images directory')
    parser.add_argument('--test_dir', type=str, default='test', help='Test images directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for metrics')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--plot_metrics', action='store_true', default=True, help='Plot training metrics')
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=50, help='Warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    # Model config
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--max_text_length', type=int, default=128, help='Maximum text length')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum training samples (None for all)')
    parser.add_argument('--max_val_samples', type=int, default=None, help='Maximum validation samples (None for all)')
    parser.add_argument('--max_test_samples', type=int, default=None, help='Maximum test samples (None for all)')
    parser.add_argument('--num_workers', type=int, default=8, help='CPU cores to load data')
    # LoRA config
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    # Model names
    parser.add_argument('--image_encoder_model_name', type=str, default='openai/clip-vit-base-patch16', help='Image encoder model name')
    parser.add_argument('--text_encoder_model_name', type=str, default='Qwen/Qwen3-0.6B-Base', help='Text encoder model name')
    parser.add_argument('--sam_model_name', type=str, default='facebook/sam-vit-base', help='SAM model name')
    parser.add_argument('--use_vlm', action='store_true', help='Use Vision-Language Model (overrides separate image/text encoders)')
    parser.add_argument('--vlm_model_name', type=str, default='llava-hf/llava-1.5-7b-hf', help='VLM model name (PaliGemma for 8GB GPU, use llava-hf/llava-1.5-7b-hf for LISA comparison with 16GB+ GPU)')

    # Mixed precision settings
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='Mixed precision mode')
    # Quantization settings
    parser.add_argument('--use_quantization', action='store_true', help='Use quantization')
    parser.add_argument('--quantization', type=str, default='8bit', choices=['4bit', '8bit'], help='Quantization mode')
    # Training modes
    parser.add_argument('--freeze_image_encoder', action='store_true', help='Freeze image encoder')
    parser.add_argument('--freeze_text_encoder', action='store_true', help='Freeze text encoder')
    parser.add_argument('--freeze_sam', action='store_true', help='Freeze SAM')
    parser.add_argument('--freeze_vlm', action='store_true', help='Freeze SAM')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA adapters')
    # LISA loss weights
    parser.add_argument('--lambda_txt', type=float, default=1.0, help='LISA text generation loss weight (default: 0.0, set to 1.0 to enable)')
    parser.add_argument('--lambda_mask', type=float, default=1.0, help='LISA mask loss weight (default: 1.0)')
    parser.add_argument('--lambda_bce', type=float, default=2.0, help='LISA BCE loss weight within mask loss (default: 2.0)')
    parser.add_argument('--lambda_dice', type=float, default=0.5, help='LISA Dice loss weight within mask loss (default: 0.5)')
    # Early stopping & LR scheduler
    parser.add_argument('--early_stopping_patience', type=int, default=8, help='Early stopping patience (epochs)')
    parser.add_argument('--lr_scheduler_patience', type=int, default=4, help='LR scheduler patience (epochs)')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5, help='LR reduction factor')
    # Test inference
    parser.add_argument('--run_test', action='store_true', help='Run test inference after training')
    parser.add_argument('--test_only', action='store_true', help='Only run test inference (requires checkpoint)')
    parser.add_argument('--test_checkpoint', type=str, default=None, help='Checkpoint path for test inference')
    return parser.parse_args()

def plot_training_metrics(history, output_dir):
    """
    Plot training and validation metrics over epochs
    
    Args:
        history: Dict with keys 'train' and 'val', each containing lists of metrics per epoch
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = range(1, len(history['train']['loss']) + 1)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics Over Epochs', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train']['loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0, 0].plot(epochs, history['val']['loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: IoU
    axes[0, 1].plot(epochs, history['train']['iou'], 'b-o', label='Train IoU', linewidth=2, markersize=6)
    axes[0, 1].plot(epochs, history['val']['iou'], 'r-s', label='Val IoU', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('IoU', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('IoU vs Epoch', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 3: gIoU
    axes[1, 0].plot(epochs, history['train']['giou'], 'b-o', label='Train gIoU', linewidth=2, markersize=6)
    axes[1, 0].plot(epochs, history['val']['giou'], 'r-s', label='Val gIoU', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('gIoU', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Generalized IoU vs Epoch', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 4: cIoU
    axes[1, 1].plot(epochs, history['train']['ciou'], 'b-o', label='Train cIoU', linewidth=2, markersize=6)
    axes[1, 1].plot(epochs, history['val']['ciou'], 'r-s', label='Val cIoU', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('cIoU', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Complete IoU vs Epoch', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training plots saved to {plot_path}")
    plt.close()
    
    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, 'training_history.json')
    with open(metrics_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📊 Training history saved to {metrics_path}")

def test_inference(model, accelerator, args):
    """Run inference on test dataset"""
    accelerator.print("\n" + "="*60)
    accelerator.print("Running Test Inference")
    accelerator.print("="*60)

    # Load test dataset
    test_dataset = ReasonSegDataset(
        args.data_root,
        image_dir=args.test_dir,
        img_size=args.img_size,
        max_samples=args.max_test_samples
    )

    accelerator.print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=2 * args.num_workers,
    )

    # Prepare for inference
    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()
    
    # Metrics tracking
    metrics = BenchmarkMetrics()
    metrics.reset()
    
    # Get processor and tokenizer based on mode
    if hasattr(model, 'module'):
        unwrapped_model = model.module
    else:
        unwrapped_model = model
    
    if unwrapped_model.use_vlm:
        # VLM mode: use VLM's processor
        processor = unwrapped_model.vlm.processor
    else:
        # Traditional mode: use CLIP processor and text encoder tokenizer
        image_processor = tf.CLIPImageProcessor.from_pretrained(args.image_encoder_model_name)
        tokenizer = unwrapped_model.text_encoder.tokenizer
    
    accelerator.print("\nRunning inference...")
    test_start_time = time.time()
    
    with t.no_grad():
        for step, batch in enumerate(test_loader):
            iter_start = time.time()
            
            images = batch['images']
            queries = batch['queries']
            gt_masks = batch['masks']
            
            if unwrapped_model.use_vlm:
                # VLM mode: process images and text together
                vlm_inputs = prepare_vlm_inputs(processor, queries, images, args.max_text_length, accelerator.device)
                pixel_values = vlm_inputs["pixel_values"]
                input_ids = vlm_inputs["input_ids"]
                attention_mask = vlm_inputs["attention_mask"]
            else:
                # Traditional mode: process separately
                pixel_values = image_processor(images=images, return_tensors="pt")["pixel_values"]
                pixel_values = pixel_values.to(accelerator.device)
                
                text_inputs = tokenizer(
                    queries,
                    padding=True,
                    truncation=True,
                    max_length=args.max_text_length,
                    return_tensors="pt"
                )
                input_ids = text_inputs["input_ids"].to(accelerator.device)
                attention_mask = text_inputs["attention_mask"].to(accelerator.device)
            
            gt_masks = gt_masks.to(accelerator.device)
            
            # Forward pass
            with accelerator.autocast():
                pred_masks, iou_preds = model(pixel_values, input_ids, attention_mask, multimask_output=False)
                loss = combined_loss(pred_masks, gt_masks, iou_preds)
            
            iter_time = time.time() - iter_start
            metrics.update(loss.item(), pred_masks, gt_masks, iter_time)
            
            if step % 10 == 0:
                current_metrics = metrics.get_metrics()
                accelerator.print(
                    f"Step {step}/{len(test_loader)} | "
                    f"IoU: {current_metrics['iou']:.4f} | "
                    f"gIoU: {current_metrics['giou']:.4f} | "
                    f"cIoU: {current_metrics['ciou']:.4f}"
                )
    
    test_time = time.time() - test_start_time
    test_metrics = metrics.get_metrics()
    test_metrics['total_time'] = test_time
    
    accelerator.print(f"\nTest Results:")
    accelerator.print(f"  Loss: {test_metrics['loss']:.4f}")
    accelerator.print(f"  IoU: {test_metrics['iou']:.4f}")
    accelerator.print(f"  gIoU: {test_metrics['giou']:.4f}")
    accelerator.print(f"  cIoU: {test_metrics['ciou']:.4f}")
    accelerator.print(f"  Total time: {test_time:.2f}s")
    accelerator.print(f"  Avg iteration time: {test_metrics['avg_iter_time']:.3f}s")
    accelerator.print(f"  Peak memory: {test_metrics['peak_memory_gb']:.2f}GB")
    
    # Save test results
    if accelerator.is_main_process:
        test_results_path = os.path.join(args.output_dir, "test_results.json")
        with open(test_results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        accelerator.print(f"\nTest results saved to {test_results_path}")
    
    return test_metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision if args.use_mixed_precision else "no",
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Test-only mode
    if args.test_only:
        if args.test_checkpoint is None:
            raise ValueError("--test_checkpoint must be provided when using --test_only")
        
        accelerator.print(f"Loading checkpoint from {args.test_checkpoint}")

        # Initialize model
        model = ModifiedLISA(
            device=str(accelerator.device),
            use_vlm=args.use_vlm,
            vlm_model_name=args.vlm_model_name if args.use_vlm else None,
            vlm_use_mixed_precision=args.use_mixed_precision,
            vlm_mixed_precision=args.mixed_precision,
            vlm_use_quantization=args.use_quantization,
            vlm_quantization=args.quantization,
            vlm_freeze=args.freeze_vlm,
            vlm_use_lora=args.use_lora,
            vlm_lora_r=args.lora_r,
            vlm_lora_alpha=args.lora_alpha,
            vlm_lora_dropout=args.lora_dropout,
            vlm_lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            image_encoder_model_name=args.image_encoder_model_name,
            image_encoder_use_mixed_precision=args.use_mixed_precision,
            image_encoder_mixed_precision=args.mixed_precision,
            image_encoder_use_quantization=args.use_quantization,
            image_encoder_quantization=args.quantization,
            image_encoder_freeze=args.freeze_image_encoder,
            image_encoder_use_lora=args.use_lora,
            image_encoder_lora_r=args.lora_r,
            image_encoder_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
            image_encoder_lora_alpha=args.lora_alpha,
            image_encoder_lora_dropout=args.lora_dropout,
            text_encoder_model_name=args.text_encoder_model_name,
            text_encoder_use_mixed_precision=args.use_mixed_precision,
            text_encoder_mixed_precision=args.mixed_precision,
            text_encoder_use_quantization=args.use_quantization,
            text_encoder_quantization=args.quantization,
            text_encoder_freeze=args.freeze_text_encoder,
            text_encoder_use_lora=args.use_lora,
            text_encoder_lora_r=args.lora_r,
            text_encoder_lora_alpha=args.lora_alpha,
            text_encoder_lora_dropout=args.lora_dropout,
            text_encoder_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
            sam_model_name=args.sam_model_name,
            sam_use_mixed_precision=args.use_mixed_precision,
            sam_mixed_precision=args.mixed_precision,
            sam_use_quantization=args.use_quantization,
            sam_quantization=args.quantization,
            sam_freeze=args.freeze_sam,
            sam_use_lora=args.use_lora,
            sam_lora_r=args.lora_r,
            sam_lora_alpha=args.lora_alpha,
            sam_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
            sam_lora_dropout=args.lora_dropout,
            image_text_connector_use_mixed_precision=args.use_mixed_precision,
            image_text_connector_mixed_precision=args.mixed_precision,
            image_text_connector_num_layers=2,
            text_sam_connector_use_mixed_precision=args.use_mixed_precision,
            text_sam_connector_mixed_precision=args.mixed_precision,
            text_sam_connector_tokens=8,
            image_sam_connector_use_mixed_precision=args.use_mixed_precision,
            image_sam_connector_mixed_precision=args.mixed_precision,
        )

        # Load checkpoint
        checkpoint = t.load(args.test_checkpoint, map_location=accelerator.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Run test inference
        test_inference(model, accelerator, args)
        return

    # Initialize model
    accelerator.print("Initializing ModifiedLISA model...")
    model = ModifiedLISA(
        device=str(accelerator.device),
        use_vlm=args.use_vlm,
        vlm_model_name=args.vlm_model_name if args.use_vlm else None,
        vlm_use_mixed_precision=args.use_mixed_precision if args.use_vlm and args.use_mixed_precision else False,
        vlm_mixed_precision=args.mixed_precision if args.use_vlm else 'no',
        vlm_use_quantization=args.use_quantization if args.use_vlm and args.use_quantization else False,
        vlm_quantization=args.quantization if args.use_vlm and args.use_quantization else '8bit',
        vlm_freeze=args.freeze_vlm if args.use_vlm and args.freeze_vlm else False,
        vlm_use_lora=args.use_lora if args.use_vlm and args.use_lora else False,
        vlm_lora_r=args.lora_r if args.use_vlm and args.use_lora else 16,
        vlm_lora_alpha=args.lora_alpha if args.use_vlm and args.use_lora else 32,
        vlm_lora_dropout=args.lora_dropout if args.use_vlm and args.use_lora else 0.0,
        vlm_lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        image_encoder_model_name=args.image_encoder_model_name,
        image_encoder_use_mixed_precision=args.use_mixed_precision,
        image_encoder_mixed_precision=args.mixed_precision,
        image_encoder_use_quantization=args.use_quantization,
        image_encoder_quantization=args.quantization,
        image_encoder_freeze=args.freeze_image_encoder,
        image_encoder_use_lora=args.use_lora,
        image_encoder_lora_r=args.lora_r,
        image_encoder_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
        image_encoder_lora_alpha=args.lora_alpha,
        image_encoder_lora_dropout=args.lora_dropout,
        text_encoder_model_name=args.text_encoder_model_name,
        text_encoder_use_mixed_precision=args.use_mixed_precision,
        text_encoder_mixed_precision=args.mixed_precision,
        text_encoder_use_quantization=args.use_quantization,
        text_encoder_quantization=args.quantization,
        text_encoder_freeze=args.freeze_text_encoder,
        text_encoder_use_lora=args.use_lora,
        text_encoder_lora_r=args.lora_r,
        text_encoder_lora_alpha=args.lora_alpha,
        text_encoder_lora_dropout=args.lora_dropout,
        text_encoder_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
        sam_model_name=args.sam_model_name,
        sam_use_mixed_precision=args.use_mixed_precision,
        sam_mixed_precision=args.mixed_precision,
        sam_use_quantization=args.use_quantization,
        sam_quantization=args.quantization,
        sam_freeze=args.freeze_sam,
        sam_use_lora=args.use_lora,
        sam_lora_r=args.lora_r,
        sam_lora_alpha=args.lora_alpha,
        sam_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
        sam_lora_dropout=args.lora_dropout,
        image_text_connector_use_mixed_precision=args.use_mixed_precision,
        image_text_connector_mixed_precision=args.mixed_precision,
        image_text_connector_num_layers=2,
        text_sam_connector_use_mixed_precision=args.use_mixed_precision,
        text_sam_connector_mixed_precision=args.mixed_precision,
        text_sam_connector_tokens=8,
        image_sam_connector_use_mixed_precision=args.use_mixed_precision,
        image_sam_connector_mixed_precision=args.mixed_precision,
    )
    model.print_trainable_parameters()

    # Create datasets
    accelerator.print("\nLoading datasets...")
    train_dataset = ReasonSegDataset(
        args.data_root, 
        args.train_json, 
        args.train_dir, 
        args.img_size,
        max_samples=args.max_train_samples
    )
    val_dataset = ReasonSegDataset(
        args.data_root, 
        image_dir=args.val_dir, 
        img_size=args.img_size,
        max_samples=args.max_val_samples
    )
    
    accelerator.print(f"Train samples: {len(train_dataset)}")
    accelerator.print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=2 * args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=2 * args.num_workers,
    )

    # Initialize optimizer
    optimizer = t.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # Initialize LR scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        mode='max'
    )

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    
    # Training loop
    accelerator.print("\nStarting training...")
    accelerator.print(f"Early stopping patience: {args.early_stopping_patience} epochs")
    accelerator.print(f"LR scheduler patience: {args.lr_scheduler_patience} epochs")
    best_val_iou = 0.0
    best_checkpoint_path = None
    
    # Initialize history tracking
    history = {
        'train': {'loss': [], 'iou': [], 'giou': [], 'ciou': []},
        'val': {'loss': [], 'iou': [], 'giou': [], 'ciou': []}
    }

    for epoch in range(args.num_epochs):
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"Epoch {epoch + 1}/{args.num_epochs}")
        accelerator.print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        accelerator.print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, accelerator, BenchmarkMetrics(), args)
        
        accelerator.print(f"\nTraining Metrics:")
        accelerator.print(f"  Loss: {train_metrics['loss']:.4f} | IoU: {train_metrics['iou']:.4f} | gIoU: {train_metrics['giou']:.4f} | cIoU: {train_metrics['ciou']:.4f} | Iterations/sec: {train_metrics['iter_per_sec']:.2f} | Avg iteration time: {train_metrics['avg_iter_time']:.3f}s | Epoch time: {train_metrics['epoch_time']:.2f}s | Peak memory: {train_metrics['peak_memory_gb']:.2f}GB")
        
        # Validate
        val_metrics = validate(model, val_loader, accelerator, args)
        
        accelerator.print(f"\nValidation Metrics:")
        accelerator.print(f"  Loss: {val_metrics['loss']:.4f} | IoU: {val_metrics['iou']:.4f} | gIoU: {val_metrics['giou']:.4f} | cIoU: {val_metrics['ciou']:.4f}")
        
        # Track metrics history
        history['train']['loss'].append(train_metrics['loss'])
        history['train']['iou'].append(train_metrics['iou'])
        history['train']['giou'].append(train_metrics['giou'])
        history['train']['ciou'].append(train_metrics['ciou'])
        history['val']['loss'].append(val_metrics['loss'])
        history['val']['iou'].append(val_metrics['iou'])
        history['val']['giou'].append(val_metrics['giou'])
        history['val']['ciou'].append(val_metrics['ciou'])
        
        # Update learning rate scheduler
        lr_scheduler.step(val_metrics['iou'])
        
        # Save checkpoint if best
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            accelerator.print(f"\n🎯 New best validation IoU: {best_val_iou:.4f}")
            
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                best_checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model_epoch_{epoch+1}.pt")
                t.save({
                    'epoch': epoch + 1,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_iou': best_val_iou,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'args': vars(args),
                }, best_checkpoint_path)
                accelerator.print(f"Checkpoint saved to {best_checkpoint_path}")
        
        # Check early stopping
        if early_stopping(val_metrics['iou']):
            accelerator.print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
            accelerator.print(f"Best validation IoU: {best_val_iou:.4f}")
            break
        
        if early_stopping.counter > 0:
            accelerator.print(f"Early stopping counter: {early_stopping.counter}/{args.early_stopping_patience}")
        
        # Save metrics to file
        if accelerator.is_main_process:
            metrics_file = os.path.join(args.output_dir, f"metrics_epoch_{epoch+1}.json")
            with open(metrics_file, 'w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'train': train_metrics,
                    'val': val_metrics,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }, f, indent=2)
    
    accelerator.print("\nTraining completed!")
    accelerator.print(f"Best validation IoU: {best_val_iou:.4f}")
    
    # Plot training metrics
    if accelerator.is_main_process and args.plot_metrics:
        accelerator.print("\n" + "="*60)
        accelerator.print("Generating training plots...")
        accelerator.print("="*60)
        plot_training_metrics(history, args.output_dir)
    
    # Run test inference if requested
    if args.run_test and best_checkpoint_path:
        accelerator.print("\n" + "="*60)
        accelerator.print("Running test inference with best checkpoint...")
        accelerator.print("="*60)
        
        # Load best model
        checkpoint = t.load(best_checkpoint_path, map_location=accelerator.device)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Run test inference
        test_inference(model, accelerator, args)

if __name__ == "__main__":
    main()
