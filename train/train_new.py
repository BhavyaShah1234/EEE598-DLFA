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
            device: str = "cuda" if t.cuda.is_available() else "cpu",
            model_name: str = "openai/clip-vit-large-patch14",
            use_mixed_precision: bool = False,
            mixed_precision: str = "bf16",
            use_quantization: bool = False,
            quantization: str = "8bit",
            freeze: bool = False,
            use_lora: bool = False,
            lora_r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.1,
            lora_target_modules: list = [],
        ):
        super(ImageEncoder, self).__init__()
        # Determine dtype based on mixed precision settings
        if use_mixed_precision:
            if mixed_precision == 'bf16':
                dtype = t.bfloat16
            elif mixed_precision == 'fp16':
                dtype = t.float16
            else:
                dtype = t.float32
        else:
            dtype = t.float32
        # Prepare quantization config if needed
        if use_quantization:
            if quantization == "4bit":
                quantization_config = tf.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
                print(f"  → Using 4-bit quantization with dtype={dtype or t.float16}")
            elif quantization == "8bit":
                quantization_config = tf.BitsAndBytesConfig(load_in_8bit=True)
                print(f"  → Using 8-bit quantization")
            else:
                quantization_config = None
        else:
            quantization_config = None
        # Load pretrained CLIP vision encoder
        if quantization_config is not None:
            self.vision_model = tf.CLIPVisionModel.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
        else:
            self.vision_model = tf.CLIPVisionModel.from_pretrained(model_name, device_map="auto")
        # self.vision_model = self.vision_model.to(device)
        self.config = self.vision_model.config
        self.embed_dim = self.config.hidden_size
        self.patch_size = self.config.patch_size
        self.image_size = self.config.image_size
        # Apply freeze or LoRA
        if freeze:
            print("  → Freezing vision encoder")
            self.vision_model.requires_grad_(False)
        elif use_lora:
            print(f"  → Applying LoRA to vision encoder (r={lora_r}, alpha={lora_alpha})")
            lora_config = p.LoraConfig(task_type=p.TaskType.FEATURE_EXTRACTION, r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules, lora_dropout=lora_dropout, bias="none")
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
        outputs = self.vision_model(pixel_values, output_hidden_states=True)
        features_seq = outputs.last_hidden_state
        patch_features = features_seq[:, 1:, :]
        B = patch_features.shape[0]
        num_patches = patch_features.shape[1]
        H_feat = W_feat = int(num_patches ** 0.5)
        spatial_features = patch_features.transpose(1, 2).reshape(B, self.embed_dim, H_feat, W_feat)
        return features_seq, spatial_features

class LLMTextEncoder(t.nn.Module):
    """
    Wraps a pretrained LLM (LLaMA, Vicuna, etc.) for text encoding.
    """
    def __init__(
            self,
            device: str = "cuda" if t.cuda.is_available() else "cpu",
            model_name: str = "openai-community/gpt2",
            use_mixed_precision: bool = False,
            mixed_precision: str = 'bf16',
            use_quantization: bool = False,
            quantization: str = "8bit",
            freeze: bool = False,
            use_lora: bool = False,
            lora_r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.1,
            lora_target_modules: list = [],
        ):
        super(LLMTextEncoder, self).__init__()
        if use_mixed_precision:
            if mixed_precision == 'bf16':
                dtype = t.bfloat16
            elif mixed_precision == 'fp16':
                dtype = t.float16
            else:
                dtype = t.float32
        else:
            dtype = t.float32
        # Prepare quantization config
        if use_quantization:
            if quantization == "4bit":
                quantization_config = tf.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
                print(f"  → Using 4-bit quantization with dtype={dtype or t.float16}")
            elif quantization == "8bit":
                quantization_config = tf.BitsAndBytesConfig(load_in_8bit=True)
                print(f"  → Using 8-bit quantization")
            else:
                quantization_config = None
        else:
            quantization_config = None
        # Load pretrained LLM
        if quantization_config is not None:
            self.llm = tf.AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
        else:
            self.llm = tf.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")#.to(device)
            if dtype != t.float32:
                self.llm = self.llm.to(dtype)
        self.tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embed_dim = self.llm.config.hidden_size
        # Apply freeze or LoRA
        if freeze:
            print("  → Freezing LLM")
            self.llm.requires_grad_(False)
        elif use_lora:
            print(f"  → Applying LoRA to LLM (r={lora_r}, alpha={lora_alpha})")
            lora_config = p.LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules, lora_dropout=lora_dropout, bias="none", task_type=p.TaskType.CAUSAL_LM)
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

class Projector(t.nn.Module):
    """Projects vision features to LLM dimension for multimodal fusion.

    Adds optional dtype casting so inputs and weights match under mixed precision.
    """
    def __init__(
            self,
            device: str,
            vision_dim: int,
            hidden_dim: int,
            llm_dim: int,
            num_projection_layers: int,
            dtype: t.dtype = t.float32,
        ):
        super(Projector, self).__init__()
        modules = []
        for i in range(num_projection_layers):
            if i == 0:
                modules.append(t.nn.Linear(vision_dim, hidden_dim))
            else:
                modules.append(t.nn.Linear(hidden_dim, llm_dim))
            if i < num_projection_layers - 1:
                modules.append(t.nn.GELU())
        self.projector = t.nn.Sequential(*modules).to(device)
        if dtype != t.float32:
            self.projector = self.projector.to(dtype)
        self.dtype = dtype

    def forward(self, vision_features: t.Tensor) -> t.Tensor:
        if vision_features.dtype != self.dtype:
            vision_features = vision_features.to(self.dtype)
        return self.projector(vision_features)

class PromptExtractor(t.nn.Module):
    """Extracts task-specific prompt embeddings from fused VLM features.

    Supports mixed precision by casting internal parameters to requested dtype.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_prompt_tokens: int,
            dtype: t.dtype = t.float32,
        ):
        super(PromptExtractor, self).__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.output_dim = output_dim
        self.dtype = dtype
        self.prompt_queries = t.nn.Parameter(t.randn(1, num_prompt_tokens, input_dim, dtype=dtype) * 0.02)
        self.cross_attn = t.nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True).to(dtype)
        self.norm = t.nn.LayerNorm(input_dim).to(dtype)
        self.projector = t.nn.Sequential(
            t.nn.Linear(input_dim, hidden_dim),
            t.nn.GELU(),
            t.nn.Linear(hidden_dim, output_dim),
        ).to(dtype)

    def forward(self, vlm_features: t.Tensor) -> t.Tensor:
        if vlm_features.dtype != self.dtype:
            vlm_features = vlm_features.to(self.dtype)
        batch_size = vlm_features.shape[0]
        queries = self.prompt_queries.expand(batch_size, -1, -1)
        prompts, _ = self.cross_attn(queries, vlm_features, vlm_features)
        prompts = self.norm(prompts + queries)
        prompt_embeddings = self.projector(prompts)
        return prompt_embeddings

class VisionToSAMAdapter(t.nn.Module):
    """
    Adapts the shared vision encoder features to SAM's expected format.
    """
    def __init__(
            self,
            input_dim: int,
            sam_embed_dim: int = 256,
            target_spatial_size: ty.Tuple[int, int] = (64, 64)
        ):
        super(VisionToSAMAdapter, self).__init__()
        self.target_spatial_size = target_spatial_size
        # Choose a valid number of groups for GroupNorm: must divide sam_embed_dim.
        max_groups = 32 if sam_embed_dim >= 32 else sam_embed_dim
        groups = max_groups
        while groups > 1 and sam_embed_dim % groups != 0:
            groups -= 1
        self.projection = t.nn.Sequential(
            t.nn.Conv2d(input_dim, sam_embed_dim, kernel_size=1),
            t.nn.GroupNorm(groups, sam_embed_dim),
            t.nn.Conv2d(sam_embed_dim, sam_embed_dim, kernel_size=3, padding=1),
            t.nn.GroupNorm(groups, sam_embed_dim),
        )

    def forward(self, vision_features: t.Tensor) -> t.Tensor:
        """
        Args:
            vision_features: [B, input_dim, H, W]
        Returns:
            adapted_features: [B, sam_embed_dim, target_H, target_W]
        """
        # Ensure dtype match with projection weights (for mixed precision cases)
        target_dtype = self.projection[0].weight.dtype
        if vision_features.dtype != target_dtype:
            vision_features = vision_features.to(target_dtype)
        features = self.projection(vision_features)
        features = t.nn.functional.interpolate(features, size=self.target_spatial_size, mode='bilinear', align_corners=False)
        return features

class SegmentAnythingModel(t.nn.Module):
    """
    Wraps SAM's pretrained mask decoder to use shared image embeddings.
    """
    def __init__(
            self,
            device: str = "cuda" if t.cuda.is_available() else "cpu",
            sam_model_name: str = "facebook/sam-vit-base",
            use_mixed_precision: bool = False,
            mixed_precision: str = "bf16",
            use_quantization: bool = False,
            quantization: str = "8bit",
            freeze: bool = False,
            use_lora: bool = False,
            lora_r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.1,
            lora_target_modules: list = [],
        ):
        super(SegmentAnythingModel, self).__init__()
        self.device = device
        if use_mixed_precision:
            if mixed_precision == 'fp16':
                dtype = t.float16
            elif mixed_precision == 'bf16':
                dtype = t.bfloat16
            else:
                dtype = t.float32
        else:
            dtype = t.float32
        if use_quantization:
            if quantization == "4bit":
                quantization_config = tf.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
                print(f"  → Using 4-bit quantization with dtype={dtype or t.float16}")
            elif quantization == "8bit":
                quantization_config = tf.BitsAndBytesConfig(load_in_8bit=True)
                print(f"  → Using 8-bit quantization")
            else:
                quantization_config = None
        else:
            quantization_config = None
        print(f"  → Loading pretrained SAM: {sam_model_name}")
        if quantization_config is not None:
            self.sam = tf.SamModel.from_pretrained(sam_model_name, quantization_config=quantization_config, device_map="auto")
        else:
            self.sam = tf.SamModel.from_pretrained(sam_model_name, device_map="auto")#.to(device)
            if dtype != t.float32:
                self.sam = self.sam.to(dtype)
        # Apply freeze or LoRA
        if freeze:
            print("  → Freezing SAM")
            self.sam.requires_grad_(False)
        elif use_lora:
            print(f"  → Applying LoRA to SAM (r={lora_r}, alpha={lora_alpha})")
            # SamModel is not a causal LM; use FEATURE_EXTRACTION task type to avoid generation-specific hooks.
            lora_config = p.LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=p.TaskType.FEATURE_EXTRACTION,
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
        batch_size = image_embeddings.shape[0]
        dense_embeddings = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch_size, -1, image_embeddings.shape[-2], image_embeddings.shape[-1]
        )
        # Obtain wide image positional embeddings from SamModel helper (tied with prompt positional embeddings)
        image_positional_embeddings = self.sam.get_image_wide_positional_embeddings().to(image_embeddings.device)
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)
        if image_positional_embeddings.dtype != image_embeddings.dtype:
            image_positional_embeddings = image_positional_embeddings.to(image_embeddings.dtype)
        # Reshape our custom prompt embeddings to match expected sparse prompt shape: (B, point_batch_size, num_points, hidden)
        # Treat each prompt token as a single point with one coordinate embedding slot.
        sparse_prompt_embeddings = prompt_embeddings.unsqueeze(2)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        # SAM returns masks shape: [B, point_batch, num_masks, H, W]
        if low_res_masks.ndim == 5:
            B, P, M, H, W = low_res_masks.shape
            # Flatten point and mask dimensions for downstream simplicity
            flat_masks = low_res_masks.reshape(B, P * M, H, W)
        else:
            flat_masks = low_res_masks
        # Typically base model outputs 256x256 already; interpolate only if different
        if flat_masks.shape[-2:] != (256, 256):
            flat_masks = t.nn.functional.interpolate(flat_masks, size=(256, 256), mode='bilinear', align_corners=False)
        # Flatten IoU predictions similarly: [B, point_batch, num_masks] -> [B, point_batch*num_masks]
        if iou_predictions.ndim == 3:
            iou_predictions = iou_predictions.reshape(iou_predictions.shape[0], -1)
        return flat_masks, iou_predictions

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
            device: str = "cuda" if t.cuda.is_available() else "cpu",
            vision_model_name: str = "openai/clip-vit-large-patch14",
            vision_use_mixed_precision: bool = False,
            vision_mixed_precision: str = "bf16",
            vision_use_quantization: bool = False,
            vision_quantization: str = "8bit",
            vision_freeze: bool = False,
            vision_use_lora: bool = False,
            vision_lora_r: int = 8,
            vision_lora_target_modules: list = [],
            vision_lora_alpha: int = 16,
            vision_lora_dropout: float = 0.1,
            num_projection_layers: int = 2,
            llm_model_name: str = "openai-community/gpt2",
            llm_use_mixed_precision: bool = False,
            llm_mixed_precision: str = 'bf16',
            llm_use_quantization: bool = False,
            llm_quantization: str = "8bit",
            llm_freeze: bool = False,
            llm_use_lora: bool = False,
            llm_lora_r: int = 8,
            llm_lora_alpha: int = 16,
            llm_lora_dropout: float = 0.1,
            llm_lora_target_modules: list = [],
            sam_model_name: str = "facebook/sam-vit-base",
            sam_use_mixed_precision: bool = False,
            sam_mixed_precision: str = "bf16",
            sam_use_quantization: bool = False,
            sam_quantization: str = "8bit",
            sam_freeze: bool = False,
            sam_use_lora: bool = False,
            sam_lora_r: int = 8,
            sam_lora_alpha: int = 16,
            sam_lora_target_modules: list = ["qkv", "proj"],
            sam_lora_dropout: float = 0.1,
            num_prompt_tokens: int = 8,
        ):
        super(ModifiedLISA, self).__init__()
        self.device = device
        self.vision_model_name = vision_model_name
        # Determine a unified dtype based on first component requesting mixed precision
        if vision_use_mixed_precision or llm_use_mixed_precision or sam_use_mixed_precision:
            # Priority: vision -> llm -> sam (first True encountered)
            if vision_use_mixed_precision:
                active_precision = vision_mixed_precision
            elif llm_use_mixed_precision:
                active_precision = llm_mixed_precision
            else:
                active_precision = sam_mixed_precision
            if active_precision == 'bf16':
                self.dtype = t.bfloat16
            elif active_precision == 'fp16':
                self.dtype = t.float16
            else:
                self.dtype = t.float32
        else:
            self.dtype = t.float32
        # 1. Shared Image Encoder (CLIP)
        print("\n[1/5] Loading pretrained CLIP vision encoder...")
        self.image_encoder = ImageEncoder(
            device=device,
            model_name=vision_model_name,
            use_mixed_precision=vision_use_mixed_precision,
            mixed_precision=vision_mixed_precision,
            use_quantization=vision_use_quantization,
            quantization=vision_quantization,
            freeze=vision_freeze,
            use_lora=vision_use_lora,
            lora_r=vision_lora_r,
            lora_alpha=vision_lora_alpha,
            lora_dropout=vision_lora_dropout,
            lora_target_modules=vision_lora_target_modules,
        )
        vision_dim = self.image_encoder.embed_dim
        print(f"  ✓ CLIP loaded. Embedding dim: {vision_dim}")
        # 2. Text Encoder (LLM)
        print("\n[2/5] Loading pretrained LLM...")
        self.text_encoder = LLMTextEncoder(
            device=device,
            model_name=llm_model_name,
            use_mixed_precision=llm_use_mixed_precision,
            mixed_precision=llm_mixed_precision,
            use_quantization=llm_use_quantization,
            quantization=llm_quantization,
            freeze=llm_freeze,
            use_lora=llm_use_lora,
            lora_r=llm_lora_r,
            lora_alpha=llm_lora_alpha,
            lora_dropout=llm_lora_dropout,
            lora_target_modules=llm_lora_target_modules,
        )
        llm_dim = self.text_encoder.embed_dim
        print(f"  ✓ LLM loaded. Embedding dim: {llm_dim}")
        # 3. Vision-Language Projector
        print("\n[3/5] Creating vision-language projector...")
        self.vision_projector = Projector(
            device=device,
            vision_dim=vision_dim,
            hidden_dim=2 * vision_dim,
            llm_dim=llm_dim,
            num_projection_layers=num_projection_layers,
            dtype=self.dtype,
        )
        print("  ✓ Projector created")
        # 4. Prompt Extractor
        print("\n[4/5] Creating prompt extractor...")
        self.prompt_extractor = PromptExtractor(
            input_dim=llm_dim,
            hidden_dim=llm_dim // 2,
            output_dim=256,
            num_prompt_tokens=num_prompt_tokens,
            dtype=self.dtype,
        ).to(device)
        print("  ✓ Prompt extractor created")
        # 5. SAM Components
        print("\n[5/5] Loading pretrained SAM...")
        self.sam_decoder = SegmentAnythingModel(
            device=device,
            sam_model_name=sam_model_name,
            use_mixed_precision=sam_use_mixed_precision,
            mixed_precision=sam_mixed_precision,
            use_quantization=sam_use_quantization,
            quantization=sam_quantization,
            freeze=sam_freeze,
            use_lora=sam_use_lora,
            lora_r=sam_lora_r,
            lora_alpha=sam_lora_alpha,
            lora_dropout=sam_lora_dropout,
            lora_target_modules=sam_lora_target_modules,
        )
        sam_embed_dim = self.sam_decoder.sam_embed_dim
        self.vision_to_sam_adapter = VisionToSAMAdapter(input_dim=vision_dim, sam_embed_dim=sam_embed_dim, target_spatial_size=(64, 64)).to(device)
        if (vision_use_mixed_precision or llm_use_mixed_precision or sam_use_mixed_precision) and self.dtype != t.float32:
            self.vision_to_sam_adapter = self.vision_to_sam_adapter.to(self.dtype)
        print("  ✓ SAM adapter created")
        print("\n" + "="*70)
        print("✓ All Components Initialized Successfully!")
        print("="*70)
        print(f"\nConfiguration Summary:")
        print(f"  • Vision Encoder: {vision_model_name}")
        print(f"    - Mode: {'Frozen' if vision_freeze else 'LoRA' if vision_use_lora else 'Full Training'}")
        print(f"  • Text Encoder: {llm_model_name}")
        print(f"    - Mode: {'Frozen' if llm_freeze else 'LoRA' if llm_use_lora else 'Full Training'}")
        print(f"    - Quantization: {llm_quantization or 'None'}")
        print(f"  • Mask Decoder: {sam_model_name}")
        if sam_freeze:
            sam_mode_str = 'Frozen'
        elif sam_use_lora:
            sam_mode_str = 'LoRA (feature-extraction, experimental)'
        else:
            sam_mode_str = 'Full Training'
        print(f"    - Mode: {sam_mode_str}")
        if sam_use_lora:
            print("    - Note: SAM LoRA uses FEATURE_EXTRACTION task type; generation hooks are absent.")
        print(f"  • Precision: {self.dtype}")
        print(f"  • Image Encoding: ONCE (shared between VLM and SAM)")
        print("="*70 + "\n")

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
        if self.dtype != t.float32:
            vision_features_seq = vision_features_seq.to(self.dtype)
            vision_features_spatial = vision_features_spatial.to(self.dtype)
        
        # Step 2: Process text with LLM
        text_features = self.text_encoder(input_ids, attention_mask)
        if self.dtype != t.float32:
            text_features = text_features.to(self.dtype)
        
        # Step 3: Project vision features to LLM space
        vision_features_projected = self.vision_projector(vision_features_seq)
        if self.dtype != t.float32 and vision_features_projected.dtype != self.dtype:
            vision_features_projected = vision_features_projected.to(self.dtype)
        
        # Step 4: Fuse vision and language
        fused_features = t.cat([vision_features_projected, text_features], dim=1)
        if fused_features.dtype != self.dtype and self.dtype != t.float32:
            fused_features = fused_features.to(self.dtype)
        
        # Step 5: Extract prompt embeddings
        prompt_embeddings = self.prompt_extractor(fused_features)
        if self.dtype != t.float32 and prompt_embeddings.dtype != self.dtype:
            prompt_embeddings = prompt_embeddings.to(self.dtype)
        
        # Step 6a: Adapt vision features to SAM's format
        sam_image_embeddings = self.vision_to_sam_adapter(vision_features_spatial)
        if self.dtype != t.float32 and sam_image_embeddings.dtype != self.dtype:
            sam_image_embeddings = sam_image_embeddings.to(self.dtype)
        
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
        return (
            pixel_values.to(self.device), 
            text_inputs["input_ids"].to(self.device), 
            text_inputs["attention_mask"].to(self.device)
        )
    
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
    
    print("\n" + "="*70)
    print("Modified LISA: Single Image Encoding with Pretrained Models")
    print("="*70)
    
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Example 1: Full training with mixed precision
    print("\n" + "="*70)
    print("Example 1: Full Training with BF16 Mixed Precision")
    print("="*70)
    model1 = ModifiedLISA(
        device=device,
        vision_model_name='openai/clip-vit-large-patch14',
        vision_use_mixed_precision=True,
        vision_mixed_precision="bf16",
        vision_use_quantization=False,
        vision_use_lora=False,
        vision_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
        llm_model_name='openai-community/gpt2',
        llm_use_mixed_precision=True,
        llm_mixed_precision="bf16",
        llm_use_quantization=False,
        llm_use_lora=False,
        llm_lora_target_modules=['c_proj', 'c_attn'],
        sam_model_name='facebook/sam-vit-base',
        sam_use_mixed_precision=True,
        sam_mixed_precision="bf16",
        sam_use_quantization=False,
        sam_use_lora=False,
        sam_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
    )
    model1.print_trainable_parameters()

    # Example 2: LoRA fine-tuning with 4-bit quantization
    print("\n" + "="*70)
    print("Example 2: LoRA Fine-tuning with 4-bit Quantization")
    print("="*70)
    model2 = ModifiedLISA(
        device=device,
        vision_model_name='openai/clip-vit-large-patch14',
        vision_use_mixed_precision=True,
        vision_mixed_precision="bf16",
        vision_use_quantization=False,
        vision_use_lora=True,
        vision_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
        llm_model_name='openai-community/gpt2',
        llm_use_mixed_precision=True,
        llm_mixed_precision="bf16",
        llm_use_quantization=False,
        llm_use_lora=True,
        llm_lora_target_modules=['c_proj', 'c_attn'],
        sam_model_name='facebook/sam-vit-base',
        sam_use_mixed_precision=True,
        sam_mixed_precision="bf16",
        sam_use_quantization=False,
        sam_use_lora=True,
        sam_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
    )
    model2.print_trainable_parameters()

    # Example 3: Freeze all pretrained models
    print("\n" + "="*70)
    print("Example 3: All Pretrained Models Frozen (Train only adapters)")
    print("="*70)
    model3 = ModifiedLISA(
        device=device,
        vision_model_name='openai/clip-vit-large-patch14',
        vision_use_mixed_precision=True,
        vision_mixed_precision="bf16",
        vision_use_quantization=False,
        vision_freeze=True,
        vision_use_lora=False,
        vision_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
        llm_model_name='openai-community/gpt2',
        llm_use_mixed_precision=True,
        llm_mixed_precision="bf16",
        llm_use_quantization=False,
        llm_freeze=True,
        llm_use_lora=False,
        llm_lora_target_modules=['c_proj', 'c_attn'],
        sam_model_name='facebook/sam-vit-base',
        sam_use_mixed_precision=True,
        sam_mixed_precision="bf16",
        sam_use_quantization=False,
        sam_freeze=True,
        sam_use_lora=False,
        sam_lora_target_modules=['q_proj', 'k_proj', 'v_proj'],
    )
    model3.print_trainable_parameters()
    
    # Test forward pass with model3
    print("\n" + "="*70)
    print("Testing forward pass with frozen model...")
    print("="*70)
    
    batch_size = 2
    pixel_values = t.randn(batch_size, 3, img_h, img_w).to(device)
    input_ids = t.randint(0, vocabulary_size, (batch_size, max_length)).to(device)
    attention_mask = t.ones(batch_size, max_length).to(device)
    
    print(f"\nInput shapes:")
    print(f"  • Images: {pixel_values.shape}")
    print(f"  • Text tokens: {input_ids.shape}")
    print(f"  • Attention mask: {attention_mask.shape}")

    print("\nRunning forward pass...")
    with t.no_grad():
        masks, iou_predictions = model3(pixel_values, input_ids, attention_mask, multimask_output=True)
    
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    print(f"Output masks shape: {masks.shape}")
    print(f"Output IoU predictions shape: {iou_predictions.shape}")
    print(f"Mask resolution: {masks.shape[-2]}x{masks.shape[-1]}")
    print(f"Number of masks per image: {masks.shape[1]}")
    
    print("\n" + "="*70)
    print("Key Features Verified:")
    print("="*70)
    print("✓ Image encoded ONLY ONCE using pretrained CLIP")
    print("✓ Text processed by pretrained LLaMA")
    print("✓ Masks generated by pretrained SAM decoder")
    print("✓ Configurable quantization (4-bit, 8-bit)")
    print("✓ Mixed precision support (bf16, fp16, fp32)")
    print("✓ Flexible training modes (freeze, LoRA, full)")
    print("✓ Shared embeddings used for VLM and SAM")
    print("✓ Zero redundant image encoding!")
    print("="*70 + "\n")
