import os
import json
import time
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
            image_encoder_model_name: str,
            image_encoder_use_mixed_precision: bool,
            image_encoder_mixed_precision: str,
            image_encoder_use_quantization: bool,
            image_encoder_quantization: str,
            image_encoder_freeze: bool,
            image_encoder_use_lora: bool,
            image_encoder_lora_r: int,
            image_encoder_lora_target_modules: list,
            image_encoder_lora_alpha: int,
            image_encoder_lora_dropout: float,
            text_encoder_model_name: str,
            text_encoder_use_mixed_precision: bool,
            text_encoder_mixed_precision: str,
            text_encoder_use_quantization: bool,
            text_encoder_quantization: str,
            text_encoder_freeze: bool,
            text_encoder_use_lora: bool,
            text_encoder_lora_r: int,
            text_encoder_lora_alpha: int,
            text_encoder_lora_dropout: float,
            text_encoder_lora_target_modules: list,
            sam_model_name: str,
            sam_use_mixed_precision: bool,
            sam_mixed_precision: str,
            sam_use_quantization: bool,
            sam_quantization: str,
            sam_freeze: bool,
            sam_use_lora: bool,
            sam_lora_r: int,
            sam_lora_alpha: int,
            sam_lora_target_modules: list,
            sam_lora_dropout: float,
            image_text_connector_use_mixed_precision: str,
            image_text_connector_mixed_precision: str,
            image_text_connector_num_layers: int,
            text_sam_connector_use_mixed_precision: bool,
            text_sam_connector_mixed_precision: str,
            text_sam_connector_tokens: int,
            image_sam_connector_use_mixed_precision: bool,
            image_sam_connector_mixed_precision: str,
        ):
        super(ModifiedLISA, self).__init__()
        self.device = device
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
        self.image_sam_connector = ImageEncoderSAMConnector(
            device=device,
            dtype=image_sam_connector_dtype,
            image_encoder_dim=image_encoder_dim,
            sam_dim=sam_dim,
            target_spatial_size=(64, 64),
        )
        print("  ✓ SAM adapter created")

    def forward(self, pixel_values: t.Tensor, input_ids: t.Tensor, attention_mask: ty.Optional[t.Tensor] = None, multimask_output: bool = True) -> ty.Tuple[t.Tensor, t.Tensor]:
        """
        Args:
            pixel_values: [B, 3, H, W]
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            multimask_output: whether to output multiple masks
        Returns:
            masks: [B, num_masks, 256, 256]
            iou_predictions: [B, num_masks]
        """
        # Step 1: Encode image ONCE
        vision_features_seq, vision_features_spatial = self.image_encoder(pixel_values)

        # Step 2: Process text with LLM
        text_features = self.text_encoder(input_ids, attention_mask)

        # Step 3: Project vision features to LLM space
        vision_features_projected = self.image_text_connector(vision_features_seq)

        # Step 4: Fuse vision and language
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
        processor = tf.CLIPImageProcessor.from_pretrained(self.vision_model_name)
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        tokenizer = self.text_encoder.tokenizer
        text_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        return pixel_values.to(self.device), text_inputs["input_ids"].to(self.device), text_inputs["attention_mask"].to(self.device)

    def print_trainable_parameters(self):
        """Print the number of trainable parameters in the model"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"\nTrainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.2f}%")

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

def compute_giou(pred_mask, gt_mask, threshold=0.5):
    """Compute Generalized IoU"""
    pred_mask = (pred_mask > threshold).float()
    gt_mask = (gt_mask > threshold).float()
    
    # IoU
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    
    # Find smallest enclosing box
    pred_coords = t.nonzero(pred_mask, as_tuple=False)
    gt_coords = t.nonzero(gt_mask, as_tuple=False)
    
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return iou.item()
    
    all_coords = t.cat([pred_coords, gt_coords], dim=0)
    c_min = all_coords.min(dim=0)[0]
    c_max = all_coords.max(dim=0)[0]
    
    c_area = (c_max - c_min + 1).prod()
    
    giou = iou - (c_area - union) / c_area
    return giou.item()

def compute_ciou(pred_mask, gt_mask, threshold=0.5):
    """Compute Complete IoU for masks"""
    pred_mask = (pred_mask > threshold).float()
    gt_mask = (gt_mask > threshold).float()
    
    # IoU
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    
    # Get coordinates
    pred_coords = t.nonzero(pred_mask, as_tuple=False)
    gt_coords = t.nonzero(gt_mask, as_tuple=False)
    
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return iou.item()
    
    # Enclosing box (for GIoU part)
    all_coords = t.cat([pred_coords, gt_coords], dim=0)
    c_min = all_coords.min(dim=0)[0]
    c_max = all_coords.max(dim=0)[0]
    c_area = (c_max - c_min + 1).prod()
    
    # GIoU term
    giou = iou - (c_area - union) / c_area
    
    # Centroid distance penalty for CIoU
    pred_center = pred_coords.float().mean(dim=0)
    gt_center = gt_coords.float().mean(dim=0)
    center_distance = ((pred_center - gt_center) ** 2).sum()
    diagonal_distance = ((c_max - c_min) ** 2).sum() + 1e-7
    
    # Aspect ratio penalty
    pred_h = pred_coords[:, 0].max() - pred_coords[:, 0].min() + 1
    pred_w = pred_coords[:, 1].max() - pred_coords[:, 1].min() + 1
    gt_h = gt_coords[:, 0].max() - gt_coords[:, 0].min() + 1
    gt_w = gt_coords[:, 1].max() - gt_coords[:, 1].min() + 1
    
    v = (4 / (3.14159 ** 2)) * ((t.atan(gt_w / (gt_h + 1e-7)) - t.atan(pred_w / (pred_h + 1e-7))) ** 2)
    alpha = v / (1 - iou + v + 1e-7)
    
    # Complete IoU
    ciou = giou - (center_distance / diagonal_distance) - alpha * v
    
    return ciou.item()

def dice_loss(pred_masks, gt_masks, smooth=1.0):
    """Dice loss for segmentation"""
    # Don't sigmoid here - apply to logits
    pred_flat = pred_masks.view(pred_masks.size(0), -1)
    gt_flat = gt_masks.view(gt_masks.size(0), -1)
    
    # Apply sigmoid for dice calculation
    pred_sigmoid = t.sigmoid(pred_flat)
    
    intersection = (pred_sigmoid * gt_flat).sum(dim=1)
    union = pred_sigmoid.sum(dim=1) + gt_flat.sum(dim=1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def focal_loss(pred_masks, gt_masks, alpha=0.25, gamma=2.0):
    """Focal loss for segmentation"""
    # Use BCE with logits (safe for autocast)
    bce = t.nn.functional.binary_cross_entropy_with_logits(pred_masks, gt_masks, reduction='none')
    
    # Get probabilities for focal term
    pred_probs = t.sigmoid(pred_masks)
    pt = t.where(gt_masks == 1, pred_probs, 1 - pred_probs)
    focal_term = (1 - pt) ** gamma
    
    loss = alpha * focal_term * bce
    return loss.mean()

def combined_loss(pred_masks, gt_masks, iou_preds=None):
    """Combined loss: Dice + Focal"""
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
    
    loss = dice_loss(pred_masks, gt_masks) + focal_loss(pred_masks, gt_masks)
    return loss

class BenchmarkMetrics:
    """Track benchmarking metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.iou_scores = []
        self.giou_scores = []
        self.ciou_scores = []
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
        
        # Compute metrics for each sample in batch
        batch_size = pred_masks.size(0)
        for i in range(batch_size):
            iou = compute_iou(pred_masks[i], gt_masks[i, 0])
            giou = compute_giou(pred_masks[i], gt_masks[i, 0])
            ciou = compute_ciou(pred_masks[i], gt_masks[i, 0])
            
            self.iou_scores.append(iou)
            self.giou_scores.append(giou)
            self.ciou_scores.append(ciou)
        
        # Track memory
        if t.cuda.is_available():
            mem = t.cuda.max_memory_allocated() / 1024**3  # GB
            self.memory_usage.append(mem)
    
    def get_metrics(self):
        return {
            'loss': np.mean(self.losses) if self.losses else 0.0,
            'iou': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'giou': np.mean(self.giou_scores) if self.giou_scores else 0.0,
            'ciou': np.mean(self.ciou_scores) if self.ciou_scores else 0.0,
            'iter_per_sec': 1.0 / np.mean(self.iteration_times) if self.iteration_times else 0.0,
            'avg_iter_time': np.mean(self.iteration_times) if self.iteration_times else 0.0,
            'peak_memory_gb': max(self.memory_usage) if self.memory_usage else 0.0,
        }

def train_epoch(model, dataloader, optimizer, accelerator, metrics, config):
    """Train for one epoch"""
    model.train()
    metrics.reset()
    metrics.epoch_start_time = time.time()
    
    processor = tf.CLIPImageProcessor.from_pretrained(config.image_encoder_model_name)
    # Get tokenizer once before the loop
    if hasattr(model, 'module'):
        tokenizer = model.module.text_encoder.tokenizer
    else:
        tokenizer = model.text_encoder.tokenizer
    
    for step, batch in enumerate(dataloader):
        iter_start = time.time()
        
        # Prepare inputs
        images = batch['images']  # Already PIL images
        queries = batch['queries']
        gt_masks = batch['masks']
        
        # Process images for CLIP
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(accelerator.device)
        
        # Tokenize text
        text_inputs = tokenizer(
            queries, 
            padding=True, 
            truncation=True, 
            max_length=config.max_text_length,
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
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            # Skip gradient clipping with FP16 due to accelerate limitations
            # if accelerator.sync_gradients and config.max_grad_norm > 0:
            #     accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        iter_time = time.time() - iter_start
        
        # Update metrics
        metrics.update(loss.item(), pred_masks.detach(), gt_masks, iter_time)
        
        if step % 5 == 0:
            current_metrics = metrics.get_metrics()
            accelerator.print(
                f"Step {step}/{len(dataloader)} | "
                f"Loss: {current_metrics['loss']:.4f} | "
                f"IoU: {current_metrics['iou']:.4f} | "
                f"gIoU: {current_metrics['giou']:.4f} | "
                f"cIoU: {current_metrics['ciou']:.4f} | "
                f"Iter/s: {current_metrics['iter_per_sec']:.2f} | "
                f"Mem: {current_metrics['peak_memory_gb']:.2f}GB"
            )
    
    epoch_time = time.time() - metrics.epoch_start_time
    epoch_metrics = metrics.get_metrics()
    epoch_metrics['epoch_time'] = epoch_time
    
    return epoch_metrics

def validate(model, dataloader, accelerator, config):
    """Validate the model"""
    model.eval()
    metrics = BenchmarkMetrics()
    
    processor = tf.CLIPImageProcessor.from_pretrained(config.image_encoder_model_name)
    # Get tokenizer once before the loop
    if hasattr(model, 'module'):
        tokenizer = model.module.text_encoder.tokenizer
    else:
        tokenizer = model.text_encoder.tokenizer
    
    with t.no_grad():
        for batch in dataloader:
            iter_start = time.time()
            
            images = batch['images']  # Already PIL images
            queries = batch['queries']
            gt_masks = batch['masks']
            
            # Process inputs
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
            pixel_values = pixel_values.to(accelerator.device)
            
            text_inputs = tokenizer(
                queries, 
                padding=True, 
                truncation=True, 
                max_length=config.max_text_length,
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
    
    return metrics.get_metrics()

def main():
    # Initialize configuration
    config = TrainingConfig()
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision if config.use_mixed_precision else "no",
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize model
    accelerator.print("Initializing ModifiedLISA model...")
    model = ModifiedLISA(
        device=str(accelerator.device),
        image_encoder_model_name=config.image_encoder_model_name,
        image_encoder_use_mixed_precision=config.use_mixed_precision,
        image_encoder_mixed_precision=config.mixed_precision,
        image_encoder_use_quantization=config.use_quantization,
        image_encoder_quantization=config.quantization,
        image_encoder_freeze=config.freeze_image_encoder,
        image_encoder_use_lora=config.use_lora,
        image_encoder_lora_r=config.lora_r,
        image_encoder_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
        image_encoder_lora_alpha=config.lora_alpha,
        image_encoder_lora_dropout=config.lora_dropout,
        text_encoder_model_name=config.text_encoder_model_name,
        text_encoder_use_mixed_precision=config.use_mixed_precision,
        text_encoder_mixed_precision=config.mixed_precision,
        text_encoder_use_quantization=config.use_quantization,
        text_encoder_quantization=config.quantization,
        text_encoder_freeze=config.freeze_text_encoder,
        text_encoder_use_lora=config.use_lora,
        text_encoder_lora_r=config.lora_r,
        text_encoder_lora_alpha=config.lora_alpha,
        text_encoder_lora_dropout=config.lora_dropout,
        text_encoder_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
        sam_model_name=config.sam_model_name,
        sam_use_mixed_precision=config.use_mixed_precision,
        sam_mixed_precision=config.mixed_precision,
        sam_use_quantization=config.use_quantization,
        sam_quantization=config.quantization,
        sam_freeze=config.freeze_sam,
        sam_use_lora=config.use_lora,
        sam_lora_r=config.lora_r,
        sam_lora_alpha=config.lora_alpha,
        sam_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
        sam_lora_dropout=config.lora_dropout,
        image_text_connector_use_mixed_precision=config.use_mixed_precision,
        image_text_connector_mixed_precision=config.mixed_precision,
        image_text_connector_num_layers=2,
        text_sam_connector_use_mixed_precision=config.use_mixed_precision,
        text_sam_connector_mixed_precision=config.mixed_precision,
        text_sam_connector_tokens=8,
        image_sam_connector_use_mixed_precision=config.use_mixed_precision,
        image_sam_connector_mixed_precision=config.mixed_precision,
    )

    model.print_trainable_parameters()
    
    # Create datasets
    accelerator.print("\nLoading datasets...")
    train_dataset = ReasonSegDataset(
        config.data_root, 
        config.train_json, 
        config.train_dir, 
        config.img_size,
        max_samples=config.max_train_samples
    )
    val_dataset = ReasonSegDataset(
        config.data_root, 
        image_dir=config.val_dir, 
        img_size=config.img_size,
        max_samples=config.max_val_samples
    )
    
    accelerator.print(f"Train samples: {len(train_dataset)}")
    accelerator.print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )

    # Initialize optimizer
    optimizer = t.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    
    # Training loop
    accelerator.print("\nStarting training...")
    best_val_iou = 0.0
    
    for epoch in range(config.num_epochs):
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"Epoch {epoch + 1}/{config.num_epochs}")
        accelerator.print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, accelerator, BenchmarkMetrics(), config)
        
        accelerator.print(f"\nTraining Metrics:")
        accelerator.print(f"  Loss: {train_metrics['loss']:.4f}")
        accelerator.print(f"  IoU: {train_metrics['iou']:.4f}")
        accelerator.print(f"  gIoU: {train_metrics['giou']:.4f}")
        accelerator.print(f"  cIoU: {train_metrics['ciou']:.4f}")
        accelerator.print(f"  Iterations/sec: {train_metrics['iter_per_sec']:.2f}")
        accelerator.print(f"  Avg iteration time: {train_metrics['avg_iter_time']:.3f}s")
        accelerator.print(f"  Epoch time: {train_metrics['epoch_time']:.2f}s")
        accelerator.print(f"  Peak memory: {train_metrics['peak_memory_gb']:.2f}GB")
        
        # Validate
        val_metrics = validate(model, val_loader, accelerator, config)
        
        accelerator.print(f"\nValidation Metrics:")
        accelerator.print(f"  Loss: {val_metrics['loss']:.4f}")
        accelerator.print(f"  IoU: {val_metrics['iou']:.4f}")
        accelerator.print(f"  gIoU: {val_metrics['giou']:.4f}")
        accelerator.print(f"  cIoU: {val_metrics['ciou']:.4f}")
        
        # Save checkpoint
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            accelerator.print(f"\nNew best validation IoU: {best_val_iou:.4f}")
            
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint_path = os.path.join(config.checkpoint_dir, f"best_model_epoch_{epoch+1}.pt")
                t.save({
                    'epoch': epoch + 1,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_iou': best_val_iou,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                }, checkpoint_path)
                accelerator.print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save metrics to file
        if accelerator.is_main_process:
            metrics_file = os.path.join(config.output_dir, f"metrics_epoch_{epoch+1}.json")
            with open(metrics_file, 'w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'train': train_metrics,
                    'val': val_metrics
                }, f, indent=2)
    
    accelerator.print("\nTraining completed!")
    accelerator.print(f"Best validation IoU: {best_val_iou:.4f}")

if __name__ == "__main__":
    main()
