import peft as p
import torch as t
import typing as ty
import transformers as tf

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
        )

    def forward(self, vlm_features: t.Tensor) -> t.Tensor:
        # vlm_features = vlm_features.to(device=self.device, dtype=self.dtype)
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
        # if self.dtype != t.float32:
        #     text_features = text_features.to(self.dtype)

        # Step 3: Project vision features to LLM space
        vision_features_projected = self.image_text_connector(vision_features_seq)
        # if self.dtype != t.float32 and vision_features_projected.dtype != self.dtype:
        #     vision_features_projected = vision_features_projected.to(self.dtype)

        # Step 4: Fuse vision and language
        fused_features = t.cat([vision_features_projected, text_features], dim=1)
        # if fused_features.dtype != self.dtype and self.dtype != t.float32:
        #     fused_features = fused_features.to(self.dtype)

        # Step 5: Extract prompt embeddings
        prompt_embeddings = self.text_sam_connector(fused_features)
        # if self.dtype != t.float32 and prompt_embeddings.dtype != self.dtype:
        #     prompt_embeddings = prompt_embeddings.to(self.dtype)

        # Step 6a: Adapt vision features to SAM's format
        sam_image_embeddings = self.image_sam_connector(vision_features_spatial)
        # if self.dtype != t.float32 and sam_image_embeddings.dtype != self.dtype:
        #     sam_image_embeddings = sam_image_embeddings.to(self.dtype)

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

# Example usage
if __name__ == "__main__":
    vocabulary_size = 32000
    max_length = 50
    img_h, img_w = 224, 224
    device = "cuda" if t.cuda.is_available() else "cpu"
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    image_encoder_model_name = 'openai/clip-vit-base-patch16'
    image_encoder_use_mixed_precision = True
    image_encoder_mixed_precision = "fp16"
    image_encoder_use_quantization = False
    image_encoder_quantization = "8bit"
    image_encoder_freeze = False
    image_encoder_use_lora = False
    image_encoder_lora_target_modules = ['q_proj', 'k_proj', 'v_proj']
    text_encoder_model_name = 'meta-llama/Llama-3.2-1B' # 'openai-community/gpt2'
    text_encoder_use_mixed_precision = True
    text_encoder_mixed_precision = "fp16"
    text_encoder_use_quantization = False
    text_encoder_quantization = "8bit"
    text_encoder_freeze = False
    text_encoder_use_lora = False
    text_encoder_lora_target_modules = ['q_proj', 'k_proj', 'v_proj'] # ['c_proj', 'c_attn']
    sam_model_name = 'facebook/sam-vit-base'
    sam_use_mixed_precision = True
    sam_mixed_precision = "fp16"
    sam_use_quantization = False
    sam_quantization = "8bit"
    sam_freeze = False
    sam_use_lora = False
    sam_lora_target_modules = ['q_proj', 'k_proj', 'v_proj']
    image_text_connector_use_mixed_precision=False
    image_text_connector_mixed_precision="bf16"
    image_text_connector_num_layers=2
    text_sam_connector_use_mixed_precision=False
    text_sam_connector_mixed_precision="bf16"
    text_sam_connector_tokens=8
    image_sam_connector_use_mixed_precision=False
    image_sam_connector_mixed_precision="bf16"
    print(f"\nUsing device: {device}")

    model = ModifiedLISA(
        device=device,
        image_encoder_model_name=image_encoder_model_name,
        image_encoder_use_mixed_precision=image_encoder_use_mixed_precision,
        image_encoder_mixed_precision=image_encoder_mixed_precision,
        image_encoder_use_quantization=image_encoder_use_quantization,
        image_encoder_quantization=image_encoder_quantization,
        image_encoder_freeze=image_encoder_freeze,
        image_encoder_use_lora=image_encoder_use_lora,
        image_encoder_lora_r=lora_r,
        image_encoder_lora_target_modules=image_encoder_lora_target_modules,
        image_encoder_lora_alpha=lora_alpha,
        image_encoder_lora_dropout=lora_dropout,
        text_encoder_model_name=text_encoder_model_name,
        text_encoder_use_mixed_precision=text_encoder_use_mixed_precision,
        text_encoder_mixed_precision=text_encoder_mixed_precision,
        text_encoder_use_quantization=text_encoder_use_quantization,
        text_encoder_quantization=text_encoder_quantization,
        text_encoder_freeze=text_encoder_freeze,
        text_encoder_use_lora=text_encoder_use_lora,
        text_encoder_lora_r=lora_r,
        text_encoder_lora_alpha=lora_alpha,
        text_encoder_lora_dropout=lora_dropout,
        text_encoder_lora_target_modules=text_encoder_lora_target_modules,
        sam_model_name=sam_model_name,
        sam_use_mixed_precision=sam_use_mixed_precision,
        sam_mixed_precision=sam_mixed_precision,
        sam_use_quantization=sam_use_quantization,
        sam_quantization=sam_quantization,
        sam_freeze=sam_freeze,
        sam_use_lora=sam_use_lora,
        sam_lora_r=lora_r,
        sam_lora_alpha=lora_alpha,
        sam_lora_target_modules=sam_lora_target_modules,
        sam_lora_dropout=lora_dropout,
        image_text_connector_use_mixed_precision=image_text_connector_use_mixed_precision,
        image_text_connector_mixed_precision=image_text_connector_mixed_precision,
        image_text_connector_num_layers=image_text_connector_num_layers,
        text_sam_connector_use_mixed_precision=text_sam_connector_use_mixed_precision,
        text_sam_connector_mixed_precision=text_sam_connector_mixed_precision,
        text_sam_connector_tokens=text_sam_connector_tokens,
        image_sam_connector_use_mixed_precision=image_sam_connector_use_mixed_precision,
        image_sam_connector_mixed_precision=image_sam_connector_mixed_precision,
    )
    model.print_trainable_parameters()

    batch_size = 2
    pixel_values = t.randn(batch_size, 3, img_h, img_w, dtype=t.float32).to(device)
    input_ids = t.randint(0, vocabulary_size, (batch_size, max_length), dtype=t.int32).to(device)
    attention_mask = t.ones(batch_size, max_length, dtype=t.int32).to(device)

    print("\nRunning forward pass...")
    with t.no_grad():
        masks, iou_predictions = model(pixel_values, input_ids, attention_mask, multimask_output=True)
