# import os
# import torch
# import json as j
# import peft as p
# import typing as u
# import argparse as g
# import accelerate as a
# import transformers as t
# from PIL import Image

# class SegmentationDataset(torch.utils.data.Dataset):
#     """Dataset for image segmentation with text prompts."""
#     def __init__(self, data_path: str, image_dir: str, processor, tokenizer, sam_processor, seg_token_id: int):
#         super(SegmentationDataset, self).__init__()
#         self.data = self.load_data(data_path)
#         self.image_dir = image_dir
#         self.processor = processor
#         self.tokenizer = tokenizer
#         self.sam_processor = sam_processor
#         self.seg_token_id = seg_token_id

#     def load_data(self, data_path: str) -> u.List[u.Dict]:
#         with open(data_path, 'r') as f:
#             return j.load(f)

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, idx: int) -> u.Dict:
#         item = self.data[idx]
#         image_path = os.path.join(self.image_dir, item['image'])
#         image = Image.open(image_path).convert('RGB')
#         text = item['text']
#         target = item['target']
#         mask = item.get('mask', None)
#         if hasattr(self.processor, 'image_processor'):
#             pixel_values = self.processor.image_processor(image, return_tensors='pt')['pixel_values'][0]
#         else:
#             pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values'][0]
#         # pixel_values: [channels, height, width]
#         sam_inputs = self.sam_processor(images=image, return_tensors='pt')
#         sam_pixel_values = sam_inputs['pixel_values'][0]
#         # sam_pixel_values: [channels, height, width]
#         input_ids = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=512, truncation=True)['input_ids'][0]
#         # input_ids: [sequence_length]
#         target_ids = self.tokenizer(target, return_tensors='pt', padding='max_length', max_length=512, truncation=True)['input_ids'][0]
#         # target_ids: [sequence_length]
#         return {
#             'pixel_values': pixel_values,
#             'sam_pixel_values': sam_pixel_values,
#             'input_ids': input_ids,
#             'target_ids': target_ids,
#             'mask': torch.tensor(mask) if mask is not None else None,  # [height, width]
#         }

# class VisionEncoder(torch.nn.Module):
#     """Modular vision encoder supporting CLIP, ViT, etc."""
#     def __init__(self, encoder_name: str, projection_dim: int):
#         super(VisionEncoder, self).__init__()
#         self.encoder_name = encoder_name
#         if 'clip' in encoder_name.lower():
#             self.vision_model = t.CLIPVisionModel.from_pretrained(encoder_name)
#             self.processor = t.CLIPImageProcessor.from_pretrained(encoder_name)
#         else:
#             self.vision_model = t.AutoModel.from_pretrained(encoder_name)
#             self.processor = t.AutoProcessor.from_pretrained(encoder_name)
#         vision_hidden_size = self.vision_model.config.hidden_size
#         self.projection = torch.nn.Linear(vision_hidden_size, projection_dim)

#     def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
#         # pixel_values: [batch_size, channels, height, width]
#         vision_outputs = self.vision_model(pixel_values=pixel_values)
#         if hasattr(vision_outputs, 'last_hidden_state'):
#             vision_features = vision_outputs.last_hidden_state
#             # vision_features: [batch_size, num_patches, hidden_size]
#         else:
#             vision_features = vision_outputs.pooler_output.unsqueeze(1)
#             # vision_features: [batch_size, 1, hidden_size]
#         projected_features = self.projection(vision_features)
#         # projected_features: [batch_size, num_patches, projection_dim]
#         return projected_features

# class VLMWithSAM(torch.nn.Module):
#     """Combined Vision Language Model with SAM segmentation head."""
#     def __init__(self, vlm_name: u.Optional[str], llm_name: u.Optional[str], vision_encoder_name: u.Optional[str], sam_model_name: str, seg_token: str = '<SEG>', use_vlm: bool = True, sam_trainable: bool = True):
#         super(VLMWithSAM, self).__init__()
#         self.seg_token = seg_token
#         self.use_vlm = use_vlm
#         self.sam_trainable = sam_trainable
#         if use_vlm:
#             self.vlm = t.LlavaForConditionalGeneration.from_pretrained(vlm_name, dtype=torch.float32)
#             self.processor = t.LlavaProcessor.from_pretrained(vlm_name)
#             self.tokenizer = self.processor.tokenizer
#             llm_hidden_size = self.vlm.config.text_config.hidden_size
#         else:
#             self.llm = t.AutoModelForCausalLM.from_pretrained(llm_name, dtype=torch.float32)
#             self.tokenizer = t.AutoTokenizer.from_pretrained(llm_name)
#             self.vision_encoder = VisionEncoder(vision_encoder_name, self.llm.config.hidden_size)
#             llm_hidden_size = self.llm.config.hidden_size
#         self.tokenizer.add_tokens([seg_token])
#         self.seg_token_id = self.tokenizer.convert_tokens_to_ids(seg_token)
#         if use_vlm:
#             self.vlm.resize_token_embeddings(len(self.tokenizer))
#             self.language_model = self.vlm.language_model
#         else:
#             self.llm.resize_token_embeddings(len(self.tokenizer))
#             self.language_model = self.llm
#         self.sam = t.SamModel.from_pretrained(sam_model_name)
#         self.sam_processor = t.SamProcessor.from_pretrained(sam_model_name)
#         if not sam_trainable:
#             for param in self.sam.parameters():
#                 param.requires_grad = False
#         sam_embed_dim = self.sam.config.vision_config.output_channels
#         self.seg_projection = torch.nn.Linear(llm_hidden_size, sam_embed_dim)

#     def forward(self, pixel_values: torch.Tensor, sam_pixel_values: torch.Tensor, input_ids: torch.Tensor, target_ids: u.Optional[torch.Tensor] = None, masks: u.Optional[torch.Tensor] = None) -> u.Dict[str, torch.Tensor]:
#         # pixel_values: [batch_size, channels, height, width]
#         # sam_pixel_values: [batch_size, channels, height, width]
#         # input_ids: [batch_size, sequence_length]
#         # target_ids: [batch_size, sequence_length] or None
#         # masks: [batch_size, height, width] or None
#         batch_size = input_ids.shape[0]
#         if self.use_vlm:
#             outputs = self.vlm(pixel_values=pixel_values, input_ids=input_ids, labels=target_ids, output_hidden_states=True, return_dict=True)
#             hidden_states = outputs.hidden_states[-1]
#             # hidden_states: [batch_size, sequence_length, hidden_size]
#             lm_loss = outputs.loss if target_ids is not None else None
#         else:
#             vision_features = self.vision_encoder(pixel_values)
#             # vision_features: [batch_size, num_patches, hidden_size]
#             inputs_embeds = self.llm.get_input_embeddings()(input_ids)
#             # inputs_embeds: [batch_size, sequence_length, hidden_size]
#             inputs_embeds = torch.cat([vision_features, inputs_embeds], dim=1)
#             # inputs_embeds: [batch_size, num_patches + sequence_length, hidden_size]
#             if target_ids is not None:
#                 vision_labels = torch.full((batch_size, vision_features.shape[1]), -100, dtype=torch.long, device=target_ids.device)
#                 # vision_labels: [batch_size, num_patches]
#                 labels = torch.cat([vision_labels, target_ids], dim=1)
#                 # labels: [batch_size, num_patches + sequence_length]
#             else:
#                 labels = None
#             outputs = self.llm(inputs_embeds=inputs_embeds, labels=labels, output_hidden_states=True, return_dict=True)
#             hidden_states = outputs.hidden_states[-1]
#             # hidden_states: [batch_size, num_patches + sequence_length, hidden_size]
#             lm_loss = outputs.loss if labels is not None else None
#         seg_mask = (input_ids == self.seg_token_id) | (target_ids == self.seg_token_id if target_ids is not None else False)
#         # seg_mask: [batch_size, sequence_length]
#         seg_loss = None
#         if seg_mask.any() and masks is not None:
#             seg_token_hidden = hidden_states[seg_mask]
#             # seg_token_hidden: [num_seg_tokens, hidden_size]
#             seg_embeddings = self.seg_projection(seg_token_hidden)
#             # seg_embeddings: [num_seg_tokens, sam_embed_dim]
#             sam_outputs = self.sam.vision_encoder(sam_pixel_values)
#             image_embeddings = sam_outputs[0]
#             # image_embeddings: [batch_size, embed_dim, feat_height, feat_width]
#             predicted_masks = []
#             iou_predictions = []
#             seg_idx = 0
#             for batch_idx in range(batch_size):
#                 batch_seg_mask = seg_mask[batch_idx]
#                 # batch_seg_mask: [sequence_length]
#                 if not batch_seg_mask.any():
#                     continue
#                 num_seg_tokens = batch_seg_mask.sum().item()
#                 for _ in range(num_seg_tokens):
#                     seg_embed = seg_embeddings[seg_idx:seg_idx+1]
#                     # seg_embed: [1, sam_embed_dim]
#                     sparse_embeddings = seg_embed.unsqueeze(1)
#                     # sparse_embeddings: [1, 1, sam_embed_dim]
#                     dense_embeddings = torch.zeros((1, sparse_embeddings.shape[1], self.sam.config.mask_decoder_config.hidden_size), device=sparse_embeddings.device, dtype=sparse_embeddings.dtype)
#                     # dense_embeddings: [1, 1, decoder_hidden_size]
#                     image_positional_embeddings = self.sam.get_image_wide_positional_embeddings()
#                     # image_positional_embeddings: [embed_dim, feat_height, feat_width]
#                     image_positional_embeddings = image_positional_embeddings.unsqueeze(0).expand(1, -1, -1, -1)
#                     # image_positional_embeddings: [1, embed_dim, feat_height, feat_width]
#                     mask_decoder_output = self.sam.mask_decoder(
#                         image_embeddings=image_embeddings[batch_idx:batch_idx+1],
#                         image_positional_embeddings=image_positional_embeddings,
#                         sparse_prompt_embeddings=sparse_embeddings,
#                         dense_prompt_embeddings=dense_embeddings,
#                         multimask_output=False,
#                         output_attentions=False
#                     )
#                     # mask_decoder_output[0]: [1, 1, mask_height, mask_width]
#                     # mask_decoder_output[1]: [1, 1]
#                     predicted_masks.append(mask_decoder_output[0])
#                     iou_predictions.append(mask_decoder_output[1])
#                     seg_idx += 1
#             if predicted_masks:
#                 predicted_masks = torch.cat(predicted_masks, dim=0)
#                 # predicted_masks: [num_seg_tokens, 1, mask_height, mask_width]
#                 iou_predictions = torch.cat(iou_predictions, dim=0)
#                 # iou_predictions: [num_seg_tokens, 1]
#                 target_masks = masks[seg_mask]
#                 # target_masks: [num_seg_tokens, height, width]
#                 target_masks_resized = torch.nn.functional.interpolate(target_masks.unsqueeze(1).float(), size=predicted_masks.shape[-2:], mode='bilinear', align_corners=False)
#                 # target_masks_resized: [num_seg_tokens, 1, mask_height, mask_width]
#                 mask_loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted_masks.squeeze(1), target_masks_resized.squeeze(1))
#                 target_iou = (predicted_masks.sigmoid().squeeze(1) > 0.5).float()
#                 # target_iou: [num_seg_tokens, mask_height, mask_width]
#                 target_iou = (target_iou * target_masks_resized.squeeze(1)).sum(dim=(-2, -1)) / (target_iou.sum(dim=(-2, -1)) + target_masks_resized.squeeze(1).sum(dim=(-2, -1)) - (target_iou * target_masks_resized.squeeze(1)).sum(dim=(-2, -1)) + 1e-6)
#                 # target_iou: [num_seg_tokens]
#                 iou_loss = torch.nn.functional.mse_loss(iou_predictions.squeeze(-1), target_iou)
#                 seg_loss = mask_loss + iou_loss
#         total_loss = None
#         if lm_loss is not None and seg_loss is not None:
#             total_loss = lm_loss + seg_loss
#         elif lm_loss is not None:
#             total_loss = lm_loss
#         elif seg_loss is not None:
#             total_loss = seg_loss
#         return {
#             'loss': total_loss,
#             'lm_loss': lm_loss,
#             'seg_loss': seg_loss,
#             'logits': outputs.logits  # [batch_size, sequence_length, vocab_size]
#         }

# def setup_lora(model: torch.nn.Module, lora_config: u.Dict, apply_to_llm: bool = True, apply_to_vision: bool = False, apply_to_sam: bool = False) -> torch.nn.Module:
#     """Setup LoRA for different model components."""
#     if apply_to_llm:
#         lora_config_llm = p.LoraConfig(
#             r=lora_config['r'],
#             lora_alpha=lora_config['alpha'],
#             target_modules=lora_config['target_modules'],
#             lora_dropout=lora_config['dropout'],
#             bias='none',
#             task_type=p.TaskType.CAUSAL_LM
#         )
#         if hasattr(model, 'vlm'):
#             model.vlm = p.prepare_model_for_kbit_training(model.vlm)
#             model.vlm = p.get_peft_model(model.vlm, lora_config_llm)
#         elif hasattr(model, 'llm'):
#             model.llm = p.prepare_model_for_kbit_training(model.llm)
#             model.llm = p.get_peft_model(model.llm, lora_config_llm)
#     if apply_to_vision and hasattr(model, 'vision_encoder'):
#         lora_config_vision = p.LoraConfig(
#             r=lora_config.get('vision_r', lora_config['r']),
#             lora_alpha=lora_config.get('vision_alpha', lora_config['alpha']),
#             target_modules=lora_config.get('vision_target_modules', ['q_proj', 'v_proj']),
#             lora_dropout=lora_config['dropout'],
#             bias='none',
#         )
#         model.vision_encoder.vision_model = p.prepare_model_for_kbit_training(model.vision_encoder.vision_model)
#         model.vision_encoder.vision_model = p.get_peft_model(model.vision_encoder.vision_model, lora_config_vision)
#     if apply_to_sam:
#         lora_config_sam = p.LoraConfig(
#             r=lora_config.get('sam_r', lora_config['r']),
#             lora_alpha=lora_config.get('sam_alpha', lora_config['alpha']),
#             target_modules=lora_config.get('sam_target_modules', ['qkv']),
#             lora_dropout=lora_config['dropout'],
#             bias='none',
#         )
#         model.sam = p.prepare_model_for_kbit_training(model.sam)
#         model.sam = p.get_peft_model(model.sam, lora_config_sam)
#     return model

# def collate_fn(batch: u.List[u.Dict]) -> u.Dict[str, torch.Tensor]:
#     """Custom collate function for batching."""
#     pixel_values = torch.stack([item['pixel_values'] for item in batch])
#     # pixel_values: [batch_size, channels, height, width]
#     sam_pixel_values = torch.stack([item['sam_pixel_values'] for item in batch])
#     # sam_pixel_values: [batch_size, channels, height, width]
#     input_ids = torch.stack([item['input_ids'] for item in batch])
#     # input_ids: [batch_size, sequence_length]
#     target_ids = torch.stack([item['target_ids'] for item in batch])
#     # target_ids: [batch_size, sequence_length]
#     masks = None
#     if batch[0]['mask'] is not None:
#         masks = torch.stack([item['mask'] for item in batch])
#         # masks: [batch_size, height, width]
#     return {
#         'pixel_values': pixel_values,
#         'sam_pixel_values': sam_pixel_values,
#         'input_ids': input_ids,
#         'target_ids': target_ids,
#         'masks': masks
#     }

# def parse_args():
#     """Parse command line arguments."""
#     parser = g.ArgumentParser(description='Train VLM+SAM for segmentation')
#     parser.add_argument('--use_vlm', action='store_true', help='Use VLM instead of separate vision encoder + LLM')
#     parser.add_argument('--vlm_name', type=str, default='llava-hf/llava-1.5-7b-hf', help='VLM model name')
#     parser.add_argument('--llm_name', type=str, default='meta-llama/Llama-2-7b-hf', help='LLM model name')
#     parser.add_argument('--vision_encoder_name', type=str, default='openai/clip-vit-large-patch14', help='Vision encoder name')
#     parser.add_argument('--sam_model_name', type=str, default='facebook/sam-vit-huge', help='SAM model name from HuggingFace')
#     parser.add_argument('--train_data', type=str, required=True, help='Training data JSON path')
#     parser.add_argument('--image_dir', type=str, required=True, help='Image directory path')
#     parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
#     parser.add_argument('--sam_trainable', action='store_true', help='Make SAM fully trainable')
#     parser.add_argument('--use_lora', action='store_true', help='Use LoRA for fine-tuning')
#     parser.add_argument('--lora_on_llm', action='store_true', help='Apply LoRA to LLM/VLM')
#     parser.add_argument('--lora_on_vision', action='store_true', help='Apply LoRA to vision encoder')
#     parser.add_argument('--lora_on_sam', action='store_true', help='Apply LoRA to SAM')
#     parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
#     parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
#     parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
#     parser.add_argument('--lora_target_modules', type=str, nargs='+', default=['q_proj', 'v_proj'], help='LoRA target modules for LLM')
#     parser.add_argument('--lora_vision_target_modules', type=str, nargs='+', default=['q_proj', 'v_proj'], help='LoRA target modules for vision encoder')
#     parser.add_argument('--lora_sam_target_modules', type=str, nargs='+', default=['qkv'], help='LoRA target modules for SAM')
#     parser.add_argument('--sam_lora_r', type=int, default=8, help='LoRA rank for SAM')
#     parser.add_argument('--sam_lora_alpha', type=int, default=32, help='LoRA alpha for SAM')
#     parser.add_argument('--vision_lora_r', type=int, default=8, help='LoRA rank for vision encoder')
#     parser.add_argument('--vision_lora_alpha', type=int, default=32, help='LoRA alpha for vision encoder')
#     parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='Mixed precision training')
#     parser.add_argument('--batch_size', type=int, default=4, help='Batch size per device')
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
#     parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
#     parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
#     parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
#     parser.add_argument('--save_steps', type=int, default=1000, help='Save checkpoint every N steps')
#     parser.add_argument('--logging_steps', type=int, default=100, help='Log every N steps')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     accelerator = a.Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
#     model = VLMWithSAM(
#         vlm_name=args.vlm_name if args.use_vlm else None,
#         llm_name=args.llm_name if not args.use_vlm else None,
#         vision_encoder_name=args.vision_encoder_name if not args.use_vlm else None,
#         sam_model_name=args.sam_model_name,
#         use_vlm=args.use_vlm,
#         sam_trainable=args.sam_trainable
#     )
#     if args.use_lora:
#         lora_config = {
#             'r': args.lora_r,
#             'alpha': args.lora_alpha,
#             'dropout': args.lora_dropout,
#             'target_modules': args.lora_target_modules,
#             'vision_r': args.vision_lora_r,
#             'vision_alpha': args.vision_lora_alpha,
#             'vision_target_modules': args.lora_vision_target_modules,
#             'sam_r': args.sam_lora_r,
#             'sam_alpha': args.sam_lora_alpha,
#             'sam_target_modules': args.lora_sam_target_modules,
#         }
#         model = setup_lora(
#             model,
#             lora_config,
#             apply_to_llm=args.lora_on_llm,
#             apply_to_vision=args.lora_on_vision,
#             apply_to_sam=args.lora_on_sam
#         )
#     dataset = SegmentationDataset(
#         data_path=args.train_data,
#         image_dir=args.image_dir,
#         processor=model.processor if args.use_vlm else model.vision_encoder.processor,
#         tokenizer=model.tokenizer,
#         sam_processor=model.sam_processor,
#         seg_token_id=model.seg_token_id
#     )
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         collate_fn=collate_fn,
#         num_workers=4,
#         pin_memory=True
#     )
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
#     num_training_steps = len(dataloader) * args.num_epochs
#     scheduler = t.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
#     model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
#     accelerator.print(f"Starting training for {args.num_epochs} epochs")
#     accelerator.print(f"Total training steps: {num_training_steps}")
#     accelerator.print(f"SAM trainable: {args.sam_trainable}")
#     if args.use_lora:
#         accelerator.print(f"LoRA on LLM: {args.lora_on_llm}")
#         accelerator.print(f"LoRA on Vision: {args.lora_on_vision}")
#         accelerator.print(f"LoRA on SAM: {args.lora_on_sam}")
#     global_step = 0
#     for epoch in range(args.num_epochs):
#         model.train()
#         epoch_loss = 0
#         for step, batch in enumerate(dataloader):
#             with accelerator.accumulate(model):
#                 outputs = model(
#                     pixel_values=batch['pixel_values'],
#                     sam_pixel_values=batch['sam_pixel_values'],
#                     input_ids=batch['input_ids'],
#                     target_ids=batch['target_ids'],
#                     masks=batch['masks']
#                 )
#                 loss = outputs['loss']
#                 accelerator.backward(loss)
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()
#                 epoch_loss += loss.item()
#                 global_step += 1
#                 if global_step % args.logging_steps == 0:
#                     avg_loss = epoch_loss / (step + 1)
#                     accelerator.print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
#                     if outputs['lm_loss'] is not None:
#                         accelerator.print(f"  LM Loss: {outputs['lm_loss'].item():.4f}")
#                     if outputs['seg_loss'] is not None:
#                         accelerator.print(f"  Seg Loss: {outputs['seg_loss'].item():.4f}")
#                 if global_step % args.save_steps == 0:
#                     save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
#                     accelerator.wait_for_everyone()
#                     unwrapped_model = accelerator.unwrap_model(model)
#                     if accelerator.is_main_process:
#                         os.makedirs(save_path, exist_ok=True)
#                         if hasattr(unwrapped_model, 'vlm'):
#                             unwrapped_model.vlm.save_pretrained(os.path.join(save_path, "vlm"))
#                         elif hasattr(unwrapped_model, 'llm'):
#                             unwrapped_model.llm.save_pretrained(os.path.join(save_path, "llm"))
#                         if hasattr(unwrapped_model, 'vision_encoder'):
#                             if hasattr(unwrapped_model.vision_encoder.vision_model, 'save_pretrained'):
#                                 unwrapped_model.vision_encoder.vision_model.save_pretrained(os.path.join(save_path, "vision_encoder"))
#                         unwrapped_model.sam.save_pretrained(os.path.join(save_path, "sam"))
#                         unwrapped_model.tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
#                         torch.save(unwrapped_model.seg_projection.state_dict(), os.path.join(save_path, "seg_projection.pt"))
#                         accelerator.print(f"Checkpoint saved to {save_path}")
#         epoch_avg_loss = epoch_loss / len(dataloader)
#         accelerator.print(f"Epoch {epoch} completed | Average Loss: {epoch_avg_loss:.4f}")
#     final_save_path = os.path.join(args.output_dir, "final_model")
#     accelerator.wait_for_everyone()
#     unwrapped_model = accelerator.unwrap_model(model)
#     if accelerator.is_main_process:
#         os.makedirs(final_save_path, exist_ok=True)
#         if hasattr(unwrapped_model, 'vlm'):
#             unwrapped_model.vlm.save_pretrained(os.path.join(final_save_path, "vlm"))
#         elif hasattr(unwrapped_model, 'llm'):
#             unwrapped_model.llm.save_pretrained(os.path.join(final_save_path, "llm"))
#         if hasattr(unwrapped_model, 'vision_encoder'):
#             if hasattr(unwrapped_model.vision_encoder.vision_model, 'save_pretrained'):
#                 unwrapped_model.vision_encoder.vision_model.save_pretrained(os.path.join(final_save_path, "vision_encoder"))
#         unwrapped_model.sam.save_pretrained(os.path.join(final_save_path, "sam"))
#         unwrapped_model.tokenizer.save_pretrained(os.path.join(final_save_path, "tokenizer"))
#         torch.save(unwrapped_model.seg_projection.state_dict(), os.path.join(final_save_path, "seg_projection.pt"))
#         accelerator.print(f"Final model saved to {final_save_path}")
#     accelerator.print("Training completed!")

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
"""
Training script for Vision Language Model + SAM for image segmentation.
Supports multiple architectures with LoRA fine-tuning and mixed precision.
Uses HuggingFace SAM implementation.
"""

import os
import torch
import json as j
import peft as p
import typing as u
import random as rn
import argparse as g
import accelerate as a
import transformers as t
from PIL import Image, ImageDraw
import numpy as np

class SegmentationDataset(torch.utils.data.Dataset):
    """Dataset for image segmentation with text prompts."""
    def __init__(self, data_path: str, image_dir: str, processor, tokenizer, sam_processor, seg_token_id: int):
        super(SegmentationDataset, self).__init__()
        self.data = self.load_data(data_path)
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.sam_processor = sam_processor
        self.seg_token_id = seg_token_id

    def load_data(self, data_path: str) -> u.List[u.Dict]:
        with open(data_path, 'r') as f:
            return j.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def polygon_to_mask(self, points: u.List[u.List[float]], img_width: int, img_height: int) -> np.ndarray:
        """Convert polygon points to binary mask."""
        mask = Image.new('L', (img_width, img_height), 0)
        draw = ImageDraw.Draw(mask)
        points_tuple = [tuple(point) for point in points]
        draw.polygon(points_tuple, outline=1, fill=1)
        return np.array(mask, dtype=np.float32)

    def __getitem__(self, idx: int) -> u.Dict:
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item['image'])
        json_path = os.path.join(self.image_dir, item['json'])
        image = Image.open(image_path).convert('RGB')
        img_width, img_height = image.size
        with open(json_path, 'r') as f:
            json_data = j.load(f)
        text_variants = json_data['text']
        # text = text_variants[0] if isinstance(text_variants, list) else text_variants
        text = text_variants[rn.choice(text_variants)] if isinstance(text_variants, list) else text_variants
        query = item['query']
        outputs = item['outputs']
        # input_text = f"{query}"
        input_text = f"{text}"
        target_text = f"{outputs}"
        shapes = json_data.get('shapes', [])
        mask = None
        if shapes and len(shapes) > 0:
            shape = shapes[0]
            points = shape['points']
            mask = self.polygon_to_mask(points, img_width, img_height)
        if hasattr(self.processor, 'image_processor'):
            pixel_values = self.processor.image_processor(image, return_tensors='pt')['pixel_values'][0]
        else:
            pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values'][0]
        # pixel_values: [channels, height, width]
        sam_inputs = self.sam_processor(images=image, return_tensors='pt')
        sam_pixel_values = sam_inputs['pixel_values'][0]
        # sam_pixel_values: [channels, height, width]
        input_ids = self.tokenizer(input_text, return_tensors='pt', padding='max_length', max_length=512, truncation=True)['input_ids'][0]
        # input_ids: [sequence_length]
        target_ids = self.tokenizer(target_text, return_tensors='pt', padding='max_length', max_length=512, truncation=True)['input_ids'][0]
        # target_ids: [sequence_length]
        return {
            'pixel_values': pixel_values,
            'sam_pixel_values': sam_pixel_values,
            'input_ids': input_ids,
            'target_ids': target_ids,
            'mask': torch.tensor(mask) if mask is not None else None,  # [height, width]
        }

class VisionEncoder(torch.nn.Module):
    """Modular vision encoder supporting CLIP, ViT, etc."""
    def __init__(self, encoder_name: str, projection_dim: int):
        super(VisionEncoder, self).__init__()
        self.encoder_name = encoder_name
        if 'clip' in encoder_name.lower():
            self.vision_model = t.CLIPVisionModel.from_pretrained(encoder_name)
            self.processor = t.CLIPImageProcessor.from_pretrained(encoder_name)
        else:
            self.vision_model = t.AutoModel.from_pretrained(encoder_name)
            self.processor = t.AutoProcessor.from_pretrained(encoder_name)
        vision_hidden_size = self.vision_model.config.hidden_size
        self.projection = torch.nn.Linear(vision_hidden_size, projection_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [batch_size, channels, height, width]
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        if hasattr(vision_outputs, 'last_hidden_state'):
            vision_features = vision_outputs.last_hidden_state
            # vision_features: [batch_size, num_patches, hidden_size]
        else:
            vision_features = vision_outputs.pooler_output.unsqueeze(1)
            # vision_features: [batch_size, 1, hidden_size]
        projected_features = self.projection(vision_features)
        # projected_features: [batch_size, num_patches, projection_dim]
        return projected_features


class VLMWithSAM(torch.nn.Module):
    """Combined Vision Language Model with SAM segmentation head."""
    def __init__(self, vlm_name: u.Optional[str], llm_name: u.Optional[str], vision_encoder_name: u.Optional[str], sam_model_name: str, seg_token: str = '<SEG>', use_vlm: bool = True, sam_trainable: bool = True):
        super(VLMWithSAM, self).__init__()
        self.seg_token = seg_token
        self.use_vlm = use_vlm
        self.sam_trainable = sam_trainable
        if use_vlm:
            self.vlm = t.LlavaForConditionalGeneration.from_pretrained(vlm_name, dtype=torch.float32)
            self.processor = t.LlavaProcessor.from_pretrained(vlm_name)
            self.tokenizer = self.processor.tokenizer
            llm_hidden_size = self.vlm.config.text_config.hidden_size
        else:
            self.llm = t.AutoModelForCausalLM.from_pretrained(llm_name, dtype=torch.float32)
            self.tokenizer = t.AutoTokenizer.from_pretrained(llm_name)
            self.vision_encoder = VisionEncoder(vision_encoder_name, self.llm.config.hidden_size)
            llm_hidden_size = self.llm.config.hidden_size
        self.tokenizer.add_tokens([seg_token])
        self.seg_token_id = self.tokenizer.convert_tokens_to_ids(seg_token)
        if use_vlm:
            self.vlm.resize_token_embeddings(len(self.tokenizer))
            self.language_model = self.vlm.language_model
        else:
            self.llm.resize_token_embeddings(len(self.tokenizer))
            self.language_model = self.llm
        self.sam = t.SamModel.from_pretrained(sam_model_name)
        self.sam_processor = t.SamProcessor.from_pretrained(sam_model_name)
        if not sam_trainable:
            for param in self.sam.parameters():
                param.requires_grad = False
        sam_embed_dim = self.sam.config.vision_config.output_channels
        self.seg_projection = torch.nn.Linear(llm_hidden_size, sam_embed_dim)

    def forward(self, pixel_values: torch.Tensor, sam_pixel_values: torch.Tensor, input_ids: torch.Tensor, target_ids: u.Optional[torch.Tensor] = None, masks: u.Optional[torch.Tensor] = None) -> u.Dict[str, torch.Tensor]:
        # pixel_values: [batch_size, channels, height, width]
        # sam_pixel_values: [batch_size, channels, height, width]
        # input_ids: [batch_size, sequence_length]
        # target_ids: [batch_size, sequence_length] or None
        # masks: [batch_size, height, width] or None
        batch_size = input_ids.shape[0]
        if self.use_vlm:
            outputs = self.vlm(pixel_values=pixel_values, input_ids=input_ids, labels=target_ids, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[-1]
            # hidden_states: [batch_size, sequence_length, hidden_size]
            lm_loss = outputs.loss if target_ids is not None else None
        else:
            vision_features = self.vision_encoder(pixel_values)
            # vision_features: [batch_size, num_patches, hidden_size]
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            # inputs_embeds: [batch_size, sequence_length, hidden_size]
            inputs_embeds = torch.cat([vision_features, inputs_embeds], dim=1)
            # inputs_embeds: [batch_size, num_patches + sequence_length, hidden_size]
            if target_ids is not None:
                vision_labels = torch.full((batch_size, vision_features.shape[1]), -100, dtype=torch.long, device=target_ids.device)
                # vision_labels: [batch_size, num_patches]
                labels = torch.cat([vision_labels, target_ids], dim=1)
                # labels: [batch_size, num_patches + sequence_length]
            else:
                labels = None
            outputs = self.llm(inputs_embeds=inputs_embeds, labels=labels, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[-1]
            # hidden_states: [batch_size, num_patches + sequence_length, hidden_size]
            lm_loss = outputs.loss if labels is not None else None
        seg_mask = (input_ids == self.seg_token_id) | (target_ids == self.seg_token_id if target_ids is not None else False)
        # seg_mask: [batch_size, sequence_length]
        seg_loss = None
        if seg_mask.any() and masks is not None:
            seg_token_hidden = hidden_states[seg_mask]
            # seg_token_hidden: [num_seg_tokens, hidden_size]
            seg_embeddings = self.seg_projection(seg_token_hidden)
            # seg_embeddings: [num_seg_tokens, sam_embed_dim]
            sam_outputs = self.sam.vision_encoder(sam_pixel_values)
            image_embeddings = sam_outputs[0]
            # image_embeddings: [batch_size, embed_dim, feat_height, feat_width]
            predicted_masks = []
            iou_predictions = []
            seg_idx = 0
            for batch_idx in range(batch_size):
                batch_seg_mask = seg_mask[batch_idx]
                # batch_seg_mask: [sequence_length]
                if not batch_seg_mask.any():
                    continue
                num_seg_tokens = batch_seg_mask.sum().item()
                for _ in range(num_seg_tokens):
                    seg_embed = seg_embeddings[seg_idx:seg_idx+1]
                    # seg_embed: [1, sam_embed_dim]
                    sparse_embeddings = seg_embed.unsqueeze(1)
                    # sparse_embeddings: [1, 1, sam_embed_dim]
                    dense_embeddings = torch.zeros((1, sparse_embeddings.shape[1], self.sam.config.mask_decoder_config.hidden_size), device=sparse_embeddings.device, dtype=sparse_embeddings.dtype)
                    # dense_embeddings: [1, 1, decoder_hidden_size]
                    image_positional_embeddings = self.sam.get_image_wide_positional_embeddings()
                    # image_positional_embeddings: [embed_dim, feat_height, feat_width]
                    image_positional_embeddings = image_positional_embeddings.unsqueeze(0).expand(1, -1, -1, -1)
                    # image_positional_embeddings: [1, embed_dim, feat_height, feat_width]
                    mask_decoder_output = self.sam.mask_decoder(
                        image_embeddings=image_embeddings[batch_idx:batch_idx+1],
                        image_positional_embeddings=image_positional_embeddings,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        output_attentions=False
                    )
                    # mask_decoder_output[0]: [1, 1, mask_height, mask_width]
                    # mask_decoder_output[1]: [1, 1]
                    predicted_masks.append(mask_decoder_output[0])
                    iou_predictions.append(mask_decoder_output[1])
                    seg_idx += 1
            if predicted_masks:
                predicted_masks = torch.cat(predicted_masks, dim=0)
                # predicted_masks: [num_seg_tokens, 1, mask_height, mask_width]
                iou_predictions = torch.cat(iou_predictions, dim=0)
                # iou_predictions: [num_seg_tokens, 1]
                target_masks = masks[seg_mask]
                # target_masks: [num_seg_tokens, height, width]
                target_masks_resized = torch.nn.functional.interpolate(target_masks.unsqueeze(1).float(), size=predicted_masks.shape[-2:], mode='bilinear', align_corners=False)
                # target_masks_resized: [num_seg_tokens, 1, mask_height, mask_width]
                mask_loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted_masks.squeeze(1), target_masks_resized.squeeze(1))
                target_iou = (predicted_masks.sigmoid().squeeze(1) > 0.5).float()
                # target_iou: [num_seg_tokens, mask_height, mask_width]
                target_iou = (target_iou * target_masks_resized.squeeze(1)).sum(dim=(-2, -1)) / (target_iou.sum(dim=(-2, -1)) + target_masks_resized.squeeze(1).sum(dim=(-2, -1)) - (target_iou * target_masks_resized.squeeze(1)).sum(dim=(-2, -1)) + 1e-6)
                # target_iou: [num_seg_tokens]
                iou_loss = torch.nn.functional.mse_loss(iou_predictions.squeeze(-1), target_iou)
                seg_loss = mask_loss + iou_loss
        total_loss = None
        if lm_loss is not None and seg_loss is not None:
            total_loss = lm_loss + seg_loss
        elif lm_loss is not None:
            total_loss = lm_loss
        elif seg_loss is not None:
            total_loss = seg_loss
        return {
            'loss': total_loss,
            'lm_loss': lm_loss,
            'seg_loss': seg_loss,
            'logits': outputs.logits  # [batch_size, sequence_length, vocab_size]
        }


def setup_lora(model: torch.nn.Module, lora_config: u.Dict, apply_to_llm: bool = True, apply_to_vision: bool = False, apply_to_sam: bool = False) -> torch.nn.Module:
    """Setup LoRA for different model components."""
    if apply_to_llm:
        lora_config_llm = p.LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['dropout'],
            bias='none',
            task_type=p.TaskType.CAUSAL_LM
        )
        if hasattr(model, 'vlm'):
            model.vlm = p.prepare_model_for_kbit_training(model.vlm)
            model.vlm = p.get_peft_model(model.vlm, lora_config_llm)
        elif hasattr(model, 'llm'):
            model.llm = p.prepare_model_for_kbit_training(model.llm)
            model.llm = p.get_peft_model(model.llm, lora_config_llm)
    if apply_to_vision and hasattr(model, 'vision_encoder'):
        lora_config_vision = p.LoraConfig(
            r=lora_config.get('vision_r', lora_config['r']),
            lora_alpha=lora_config.get('vision_alpha', lora_config['alpha']),
            target_modules=lora_config.get('vision_target_modules', ['q_proj', 'v_proj']),
            lora_dropout=lora_config['dropout'],
            bias='none',
        )
        model.vision_encoder.vision_model = p.prepare_model_for_kbit_training(model.vision_encoder.vision_model)
        model.vision_encoder.vision_model = p.get_peft_model(model.vision_encoder.vision_model, lora_config_vision)
    if apply_to_sam:
        lora_config_sam = p.LoraConfig(
            r=lora_config.get('sam_r', lora_config['r']),
            lora_alpha=lora_config.get('sam_alpha', lora_config['alpha']),
            target_modules=lora_config.get('sam_target_modules', ['qkv']),
            lora_dropout=lora_config['dropout'],
            bias='none',
        )
        model.sam = p.prepare_model_for_kbit_training(model.sam)
        model.sam = p.get_peft_model(model.sam, lora_config_sam)
    return model


def collate_fn(batch: u.List[u.Dict]) -> u.Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    # pixel_values: [batch_size, channels, height, width]
    sam_pixel_values = torch.stack([item['sam_pixel_values'] for item in batch])
    # sam_pixel_values: [batch_size, channels, height, width]
    input_ids = torch.stack([item['input_ids'] for item in batch])
    # input_ids: [batch_size, sequence_length]
    target_ids = torch.stack([item['target_ids'] for item in batch])
    # target_ids: [batch_size, sequence_length]
    masks = None
    if batch[0]['mask'] is not None:
        masks = torch.stack([item['mask'] for item in batch])
        # masks: [batch_size, height, width]
    return {
        'pixel_values': pixel_values,
        'sam_pixel_values': sam_pixel_values,
        'input_ids': input_ids,
        'target_ids': target_ids,
        'masks': masks
    }

def parse_args():
    """Parse command line arguments."""
    parser = g.ArgumentParser(description='Train VLM+SAM for segmentation')
    parser.add_argument('--use_vlm', action='store_true', help='Use VLM instead of separate vision encoder + LLM')
    parser.add_argument('--vlm_name', type=str, default='llava-hf/llava-1.5-7b-hf', help='VLM model name')
    parser.add_argument('--llm_name', type=str, default='meta-llama/Llama-2-7b-hf', help='LLM model name')
    parser.add_argument('--vision_encoder_name', type=str, default='openai/clip-vit-large-patch14', help='Vision encoder name')
    parser.add_argument('--sam_model_name', type=str, default='facebook/sam-vit-huge', help='SAM model name from HuggingFace')
    parser.add_argument('--train_data', type=str, default='train.json', help='Training data JSON path')
    parser.add_argument('--val_data', type=str, default=None, help='Validation data JSON path')
    parser.add_argument('--train_image_dir', type=str, default='train', help='Training image directory path')
    parser.add_argument('--val_image_dir', type=str, default='val', help='Validation image directory path')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--sam_trainable', action='store_true', help='Make SAM fully trainable')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for fine-tuning')
    parser.add_argument('--lora_on_llm', action='store_true', help='Apply LoRA to LLM/VLM')
    parser.add_argument('--lora_on_vision', action='store_true', help='Apply LoRA to vision encoder')
    parser.add_argument('--lora_on_sam', action='store_true', help='Apply LoRA to SAM')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=['c_attn', 'c_proj'], help='LoRA target modules for LLM')
    parser.add_argument('--lora_vision_target_modules', type=str, nargs='+', default=['q_proj', 'v_proj'], help='LoRA target modules for vision encoder')
    parser.add_argument('--lora_sam_target_modules', type=str, nargs='+', default=['qkv'], help='LoRA target modules for SAM')
    parser.add_argument('--sam_lora_r', type=int, default=8, help='LoRA rank for SAM')
    parser.add_argument('--sam_lora_alpha', type=int, default=32, help='LoRA alpha for SAM')
    parser.add_argument('--vision_lora_r', type=int, default=8, help='LoRA rank for vision encoder')
    parser.add_argument('--vision_lora_alpha', type=int, default=32, help='LoRA alpha for vision encoder')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='Mixed precision training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=100, help='Log every N steps')
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = a.Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    model = VLMWithSAM(
        vlm_name=args.vlm_name if args.use_vlm else None,
        llm_name=args.llm_name if not args.use_vlm else None,
        vision_encoder_name=args.vision_encoder_name if not args.use_vlm else None,
        sam_model_name=args.sam_model_name,
        use_vlm=args.use_vlm,
        sam_trainable=args.sam_trainable
    )
    if args.use_lora:
        lora_config = {
            'r': args.lora_r,
            'alpha': args.lora_alpha,
            'dropout': args.lora_dropout,
            'target_modules': args.lora_target_modules,
            'vision_r': args.vision_lora_r,
            'vision_alpha': args.vision_lora_alpha,
            'vision_target_modules': args.lora_vision_target_modules,
            'sam_r': args.sam_lora_r,
            'sam_alpha': args.sam_lora_alpha,
            'sam_target_modules': args.lora_sam_target_modules,
        }
        model = setup_lora(
            model,
            lora_config,
            apply_to_llm=args.lora_on_llm,
            apply_to_vision=args.lora_on_vision,
            apply_to_sam=args.lora_on_sam
        )
    train_dataset = SegmentationDataset(
        data_path=args.train_data,
        image_dir=args.train_image_dir,
        processor=model.processor if args.use_vlm else model.vision_encoder.processor,
        tokenizer=model.tokenizer,
        sam_processor=model.sam_processor,
        seg_token_id=model.seg_token_id
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_dataloader) * args.num_epochs
    scheduler = t.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    accelerator.print(f"Starting training for {args.num_epochs} epochs")
    accelerator.print(f"Total training steps: {num_training_steps}")
    accelerator.print(f"Training samples: {len(train_dataset)}")
    accelerator.print(f"SAM trainable: {args.sam_trainable}")
    if args.use_lora:
        accelerator.print(f"LoRA on LLM: {args.lora_on_llm}")
        accelerator.print(f"LoRA on Vision: {args.lora_on_vision}")
        accelerator.print(f"LoRA on SAM: {args.lora_on_sam}")
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    pixel_values=batch['pixel_values'],
                    sam_pixel_values=batch['sam_pixel_values'],
                    input_ids=batch['input_ids'],
                    target_ids=batch['target_ids'],
                    masks=batch['masks']
                )
                loss = outputs['loss']
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                global_step += 1
                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    accelerator.print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                    if outputs['lm_loss'] is not None:
                        accelerator.print(f"  LM Loss: {outputs['lm_loss'].item():.4f}")
                    if outputs['seg_loss'] is not None:
                        accelerator.print(f"  Seg Loss: {outputs['seg_loss'].item():.4f}")
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    if accelerator.is_main_process:
                        os.makedirs(save_path, exist_ok=True)
                        if hasattr(unwrapped_model, 'vlm'):
                            unwrapped_model.vlm.save_pretrained(os.path.join(save_path, "vlm"))
                        elif hasattr(unwrapped_model, 'llm'):
                            unwrapped_model.llm.save_pretrained(os.path.join(save_path, "llm"))
                        if hasattr(unwrapped_model, 'vision_encoder'):
                            if hasattr(unwrapped_model.vision_encoder.vision_model, 'save_pretrained'):
                                unwrapped_model.vision_encoder.vision_model.save_pretrained(os.path.join(save_path, "vision_encoder"))
                        unwrapped_model.sam.save_pretrained(os.path.join(save_path, "sam"))
                        unwrapped_model.tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
                        torch.save(unwrapped_model.seg_projection.state_dict(), os.path.join(save_path, "seg_projection.pt"))
                        accelerator.print(f"Checkpoint saved to {save_path}")
        epoch_avg_loss = epoch_loss / len(train_dataloader)
        accelerator.print(f"Epoch {epoch} completed | Average Loss: {epoch_avg_loss:.4f}")
    final_save_path = os.path.join(args.output_dir, "final_model")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        os.makedirs(final_save_path, exist_ok=True)
        if hasattr(unwrapped_model, 'vlm'):
            unwrapped_model.vlm.save_pretrained(os.path.join(final_save_path, "vlm"))
        elif hasattr(unwrapped_model, 'llm'):
            unwrapped_model.llm.save_pretrained(os.path.join(final_save_path, "llm"))
        if hasattr(unwrapped_model, 'vision_encoder'):
            if hasattr(unwrapped_model.vision_encoder.vision_model, 'save_pretrained'):
                unwrapped_model.vision_encoder.vision_model.save_pretrained(os.path.join(final_save_path, "vision_encoder"))
        unwrapped_model.sam.save_pretrained(os.path.join(final_save_path, "sam"))
        unwrapped_model.tokenizer.save_pretrained(os.path.join(final_save_path, "tokenizer"))
        torch.save(unwrapped_model.seg_projection.state_dict(), os.path.join(final_save_path, "seg_projection.pt"))
        accelerator.print(f"Final model saved to {final_save_path}")
    accelerator.print("Training completed!")

if __name__ == '__main__':
    main()
