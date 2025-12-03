import torch as t
import transformers as tf
from PIL import Image

img_path = 'train/48126840_b2a2907656_o.jpg'
image = Image.open(img_path).convert('RGB')
text = 'What does this image represent?'

conversation = [
    {
        'role': 'system',
        'content': [
            {'type': 'text', 'text': "You are a helpful assistant with vision capability. You examine images carefully and understand the user's text in great detail. You respond in rhymes only."}
        ]
    },
    {
        'role': 'user',
        'content': [
            {'type': 'image'},
            {'type': 'text', 'text': text}
        ]
    }
]
print('CONVERSATION:', conversation)
print('=================================================================')

vlm_name = 'llava-hf/llava-1.5-7b-hf'
dtype = 'bf16'

processor = tf.AutoProcessor.from_pretrained(vlm_name)

conversation = processor.apply_chat_template(conversation, add_generation_prompt=True)
print(conversation)
print('=================================================================')

device = 'cuda' if t.cuda.is_available() else 'cpu'
dtype = t.bfloat16 if dtype == 'bf16' else t.float16 if dtype == 'fp16' else t.float32

inputs = processor(images=[image], text=[conversation], return_tensors='pt').to(device)
print('input_ids:', inputs['input_ids'].shape, inputs['input_ids'].min(), inputs['input_ids'].max())
print('attention_mask:', inputs['attention_mask'].shape, inputs['attention_mask'].min(), inputs['attention_mask'].max())
print('pixel_values:', inputs['pixel_values'].shape, inputs['pixel_values'].min(), inputs['pixel_values'].max())
print('*******************************************************')

quantization_config = tf.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
vlm_model = tf.AutoModelForImageTextToText.from_pretrained(vlm_name, quantization_config=quantization_config)
# print(vlm_model)
# print('____________________________________________________')
vlm_model = vlm_model.to(device)

############################################ STANDARD GENERATION ######################################################
with t.no_grad():
    outputs = vlm_model(**inputs).logits
    final_output = vlm_model.generate(**inputs, max_new_tokens=1)
    print(outputs.shape, outputs.min(), outputs.max())
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
# print(final_output)
# print(processor.batch_decode(final_output))
# print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

##################################### COMPONENT BY COMPONENT GENERATION ###############################################
with t.no_grad():
    vision_tower = vlm_model.vision_tower
    vision_outputs = vision_tower(inputs['pixel_values'], output_hidden_states=True)
    vision_outputs = vision_outputs.hidden_states[vlm_model.config.vision_feature_layer] # [1, 577, 1024]
    print(vision_outputs.shape, vision_outputs.min(), vision_outputs.max())
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    if vlm_model.config.vision_feature_select_strategy == "default":
        vision_outputs = vision_outputs[:, 1:, :] # [1, 576, 1024]
    print(vision_outputs.shape, vision_outputs.min(), vision_outputs.max())
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    projector = vlm_model.multi_modal_projector
    projector_outputs = projector(vision_outputs) # [1, 576, 4096]
    print(projector_outputs.shape, projector_outputs.min(), projector_outputs.max())
    print('-----------------------------------------------------------------------------------------')
    word_embeddings = vlm_model.get_input_embeddings()
    text_embeddings = word_embeddings(inputs['input_ids']) # [1, 576 + text_tokens, 4096]
    print(text_embeddings.shape, text_embeddings.min(), text_embeddings.max())
    print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
    image_token_id = vlm_model.config.image_token_index # 32000
    # print(image_token_id)
    mask = inputs['input_ids'] == image_token_id
    _, y_indice = mask.nonzero(as_tuple=True)
    print(y_indice.shape, y_indice.min(), y_indice.max())
    print('oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
    text_embeddings[:, y_indice, :] = projector_outputs[:, y_indice - y_indice.min(), :]
    print(text_embeddings.shape, text_embeddings.min(), text_embeddings.max())
    print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
    llm = vlm_model.language_model
    llm_outputs = llm(inputs_embeds=text_embeddings, attention_mask=inputs['attention_mask']).last_hidden_state
    print(llm_outputs.shape, llm_outputs.min(), llm_outputs.max())
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
    head = vlm_model.lm_head
    logits = head(llm_outputs)[:, -1, :]
    print(logits.shape, logits.min(), logits.max())
    print('88888888888888888888888888888888888888888888888888888888888888888888')
    token = t.argmax(logits, dim=-1)
    token = processor.decode(token)
    print(token)

print('--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--')

# import torch as t
# import transformers as tf
# from PIL import Image

img_path = 'train/64308722_06b7b2f676_o.jpg'
image = Image.open(img_path).convert('RGB')
# image = Image.new('RGB', (2048, 2048), color='green')

input_points = [[[450, 600]]]
input_bboxes = [[[100, 120, 500, 800]]]

device = 'cuda' if t.cuda.is_available() else 'cpu'
dtype = 'fp16'
dtype = t.float16 if dtype == 'bf16' else t.float16 if dtype == 'fp16' else t.float32
quantization_config = tf.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)

sam_model = 'facebook/sam-vit-base'
processor = tf.AutoProcessor.from_pretrained(sam_model)
sam_model = tf.AutoModel.from_pretrained(sam_model, quantization_config=quantization_config)
print(sam_model)
print('_______________________________________________________________________________________')
sam_model = sam_model.to(device=device)

point_inputs = processor(images=[image], input_points=input_points, return_tensors='pt').to(device=device, dtype=dtype)
bbox_inputs = processor(images=[image], input_boxes=input_bboxes, return_tensors='pt').to(device=device, dtype=dtype)
print(point_inputs['pixel_values'].shape)
print(point_inputs['original_sizes'])
print(point_inputs['reshaped_input_sizes'])
print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
print(point_inputs['input_points'].shape)
print(bbox_inputs['input_boxes'].shape)
print('=============================================================================')

with t.no_grad():
    image_encoder = sam_model.vision_encoder
    vision_outputs = image_encoder(point_inputs['pixel_values']).last_hidden_state
    print(vision_outputs.shape, vision_outputs.min(), vision_outputs.max())
    vision_outputs = image_encoder(bbox_inputs['pixel_values']).last_hidden_state
    print(vision_outputs.shape, vision_outputs.min(), vision_outputs.max())
    print('oooooooooooooooooooooooooooooooooooooooooooooo')
    batch_size = vision_outputs.shape[0]
    image_positional_embeddings = sam_model.get_image_wide_positional_embeddings().repeat(batch_size, 1, 1, 1)
    # for i in vision_outputs.hidden_states:
    #     print(i.shape)
    prompt_encoder = sam_model.prompt_encoder
    point_sparse_embeddings, point_dense_embeddings = prompt_encoder(input_points=point_inputs['input_points'], input_labels=t.ones(size=point_inputs['input_points'].shape[:-1], device=device, dtype=dtype), input_boxes=None, input_masks=None)
    print(point_sparse_embeddings.shape, point_sparse_embeddings.min(), point_sparse_embeddings.max())
    print(point_dense_embeddings.shape, point_dense_embeddings.min(), point_dense_embeddings.max())
    print('_________________________________________________________________')
    bboxes_sparse_embeddings, bboxes_dense_embeddings = prompt_encoder(input_points=None, input_labels=None, input_boxes=bbox_inputs['input_boxes'], input_masks=None)
    print(bboxes_sparse_embeddings.shape, bboxes_sparse_embeddings.min(), bboxes_sparse_embeddings.max())
    print(bboxes_dense_embeddings.shape, bboxes_dense_embeddings.min(), bboxes_dense_embeddings.max())
    print('_________________________________________________________________')
    mask_decoder = sam_model.mask_decoder
    point_low_resolution_mask, point_iou_predictions = mask_decoder(image_embeddings=vision_outputs, image_positional_embeddings=image_positional_embeddings, sparse_prompt_embeddings=point_sparse_embeddings, dense_prompt_embeddings=point_dense_embeddings, multimask_output=True)
    print(point_low_resolution_mask.shape, point_low_resolution_mask.min(), point_low_resolution_mask.max())
    print(point_iou_predictions.shape, point_iou_predictions.min(), point_iou_predictions.max())
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    bbox_low_resolution_mask, bbox_iou_predictions = mask_decoder(image_embeddings=vision_outputs, image_positional_embeddings=image_positional_embeddings, sparse_prompt_embeddings=bboxes_sparse_embeddings, dense_prompt_embeddings=bboxes_dense_embeddings, multimask_output=True)
    print(bbox_low_resolution_mask.shape, bbox_low_resolution_mask.min(), bbox_low_resolution_mask.max())
    print(bbox_iou_predictions.shape, bbox_iou_predictions.min(), bbox_iou_predictions.max())
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
