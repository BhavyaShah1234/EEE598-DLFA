import torch
import peft as p
import argparse as g
import datasets as d
import accelerate as a
import transformers as t

allowed_entities = ['AGE', 'EYECOLOR', 'GENDER', 'HEIGHT', 'WEIGHT', 'SEX']
entity_mapping = {
    "ACCOUNTNAME": "account_name",
    "ACCOUNTNUMBER": "account_number",
    "AGE": "age",
    "AMOUNT": "amount",
    "BIC": "bic",
    "BITCOINADDRESS": "bitcoin_address",
    "BUILDINGNUMBER": "building_number",
    "CITY": "city",
    "COMPANYNAME": "company_name",
    "COUNTY": "county",
    "CREDITCARDCVV": "credit_card_cvv",
    "CREDITCARDISSUER": "credit_card_issuer",
    "CREDITCARDNUMBER": "credit_card_number",
    "CURRENCY": "currency",
    "CURRENCYCODE": "currency_code",
    "CURRENCYNAME": "currency_name",
    "CURRENCYSYMBOL": "currency_symbol",
    "DATE": "date",
    "DOB": "dob",
    "EMAIL": "email",
    "ETHEREUMADDRESS": "ethereum_address",
    "EYECOLOR": "eye_color",
    "FIRSTNAME": "first_name",
    "GENDER": "gender",
    "HEIGHT": "height",
    "IBAN": "iban",
    "IP": "ip",
    "IPV4": "ipv4",
    "IPV6": "ipv6",
    "JOBAREA": "job_area",
    "JOBTITLE": "job_title",
    "JOBTYPE": "job_type",
    "LASTNAME": "last_name",
    "LITECOINADDRESS": "litecoin_address",
    "MAC": "mac",
    "MASKEDNUMBER": "masked_number",
    "MIDDLENAME": "middle_name",
    "NEARBYGPSCOORDINATE": "nearby_gps_coordinate",
    "ORDINALDIRECTION": "ordinal_direction",
    "PASSWORD": "password",
    "PHONEIMEI": "phone_imei",
    "PHONENUMBER": "phone_number",
    "PIN": "pin",
    "PREFIX": "prefix",
    "SECONDARYADDRESS": "secondary_address",
    "SEX": "sex",
    "SSN": "ssn",
    "STATE": "state",
    "STREET": "street",
    "TIME": "time",
    "URL": "url",
    "USERAGENT": "user_agent",
    "USERNAME": "username",
    "VEHICLEVIN": "vehicle_vin",
    "VEHICLEVRM": "vehicle_vrm",
    "ZIPCODE": "zip_code"
}

def process_data(x):
    entities = []
    for entity in x['privacy_mask']:
        if entity['label'] not in allowed_entities:
            entities.append({'value': entity['value'], 'label': entity_mapping[entity['label']]})
    instruction = "Extract all the personal information from the `INPUT` and classify it. Output should be list of dictionaries with keys as `value` and `label` for the entity and corresponding class respectively."
    input_text = str(x['source_text'])
    output = str(entities)
    return instruction, input_text, output

def pie_template(instruction, input_text, output):
    return f"### INSTRUCTION: {instruction}\n### INPUT: {input_text}\n### OUTPUT: {output}"

def main():
    parser = g.ArgumentParser(description="Fine-tune a language model for PII masking.")

    # Model and dataset arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Model name from HuggingFace Hub.")
    parser.add_argument("--dataset_name", type=str, default="ai4privacy/pii-masking-200k", help="Dataset name from HuggingFace Hub.")

    # Quantization arguments
    parser.add_argument("--quantization", action="store_true", help="Enable quantization.")
    parser.add_argument("--quantization_bits", type=str, choices=["4", "8"], default="8", help="Quantization bits (4 or 8).")

    # LoRA arguments
    parser.add_argument("--lora", action="store_true", help="Enable LoRA.")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")

    # Training arguments
    parser.add_argument("--use_mixed_precision", action="store_true", help="Use mixed precision training (bf16).")
    parser.add_argument("--completion_only_loss", action="store_true", help="Calculate loss only on completion tokens.")
    parser.add_argument("--output_dir", type=str, default="/scratch/bminesh-shah/ner-transformers/", help="Output directory for the fine-tuned model.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of steps for gradient accumulation.")

    # Early stopping arguments
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs with no improvement to wait before stopping.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "constant", "constant_with_warmup", "inverse_sqrt", "polynomial"], help="Learning rate scheduler type.")
    parser.add_argument("--lr_end", type=float, default=1e-8, help="Final learning rate for polynomial scheduler.")

    args = parser.parse_args()

    # Dynamic output directory based on model name and hyperparameters
    model_output_dir = f"{args.output_dir}/{args.model_name.replace('/', '-')}_pie_transformers"
    if args.lora:
        model_output_dir += f"_r{args.lora_rank}"
    if args.quantization:
        model_output_dir += f"_q{args.quantization_bits}"
    if args.use_mixed_precision:
        model_output_dir += "_bf16"

    dataset = d.load_dataset(args.dataset_name)
    dataset = dataset.filter(lambda x: x['language'] == 'en')
    dataset = dataset.remove_columns(['target_text', 'span_labels', 'mbert_text_tokens', 'mbert_bio_labels', 'id', 'language', 'set'])['train']
    dataset = dataset.train_test_split(test_size=0.3, seed=24, shuffle=True)
    temp = dataset['test'].train_test_split(test_size=0.5, seed=9)
    dataset['val'] = temp['train']
    dataset['test'] = temp['test']
    print(dataset, end='\n=================================================\n')
    print(dataset['train'][0], end='\n=================================================\n')

    tokenizer = t.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_data(x):
        instruction, input_text, output = process_data(x)
        full_text = pie_template(instruction, input_text, output)
        result = tokenizer(full_text, add_special_tokens=True)
        label = tokenizer(output, add_special_tokens=False)['input_ids']
        padding_length = len(result['input_ids']) - len(label)
        label = [-100] * padding_length + label
        result['labels'] = label
        return result

    dataset = dataset.map(tokenize_data, batched=False, remove_columns=['source_text', 'privacy_mask'])
    print(dataset)
    # print(dataset['train'][0], end='\n=================================================\n')
    # for k, (i, j) in enumerate(zip(dataset['train'][0]['input_ids'], dataset['train'][0]['labels'])):
    #     if j != -100:
    #         print(k, '|', i, tokenizer.convert_ids_to_tokens(i), '---->', tokenizer.convert_ids_to_tokens(j), j)
    #     else:
    #         print(k, '|', i, tokenizer.convert_ids_to_tokens(i), '---->', "-100", j)

    bnb_config = None
    if args.quantization:
        if args.quantization_bits == "4":
            bnb_config = t.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        elif args.quantization_bits == "8":
            bnb_config = t.BitsAndBytesConfig(load_in_8bit=True)
    print(bnb_config)

    model = t.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={'': a.Accelerator().process_index},
        dtype=torch.bfloat16 if args.use_mixed_precision else torch.float32,
        trust_remote_code=True
    )
    model.config.pretraining_tp = 1
    model.config.pad_token_id = model.config.eos_token_id

    if args.quantization:
        model = p.prepare_model_for_kbit_training(model)

    lora_config = None
    if args.lora:
        lora_config = p.LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
        model = p.get_peft_model(model, lora_config)

    training_args = t.TrainingArguments(
        output_dir=model_output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.use_mixed_precision,
        optim='adamw',
        logging_strategy='epoch',
        save_strategy='epoch',
        eval_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        warmup_ratio=0.05,
        seed=24,
        dataloader_num_workers=32,
        dataloader_prefetch_factor=64,
        gradient_checkpointing=False,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_end=args.lr_end,
    )

    # Instantiate the EarlyStoppingCallback
    early_stopping_callback = t.EarlyStoppingCallback(early_stopping_patience=args.patience)

    # Instantiate the custom data collator
    collator = t.DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=512)

    trainer = t.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collator,
        callbacks=[early_stopping_callback],
    )

    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
