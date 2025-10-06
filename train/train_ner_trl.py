import os
import torch
import trl as r
import peft as p
import argparse as g
import datasets as d
import accelerate as a
import transformers as t

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/scratch/bminesh-shah/huggingface_home"

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

def formatting_function(x):
    entities = []
    for entity in x['privacy_mask']:
        if entity['label'] not in allowed_entities:
            entities.append({'value': entity['value'], 'label': entity_mapping[entity['label']]})
    instruction = "Extract all the personal information from the `INPUT` and classify it. Output should be list of dictionaries with keys as `value` and `label` for the entity and corresponding class respectively."
    input_text = str(x['source_text'])
    output = str(entities)
    return f'### INSTRUCTION: {instruction}\n### INPUT: {input_text}\n### OUTPUT: {output}'

def main():
    parser = g.ArgumentParser(description="Fine-tune a language model for PII masking.")

    # Model and dataset arguments
    parser.add_argument("--model_name", type=str, default="openai/gpt2", help="Model name from HuggingFace Hub.")
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
    parser.add_argument("--output_dir", type=str, default="/scratch/bminesh-shah/ner-trl/", help="Output directory for the fine-tuned model.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of steps for gradient accumulation.")

    # Early stopping arguments
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs with no improvement to wait before stopping.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "constant", "constant_with_warmup", "inverse_sqrt", "polynomial"], help="Learning rate scheduler type.")
    parser.add_argument("--lr_end", type=float, default=1e-8, help="Final learning rate for polynomial scheduler.")

    args = parser.parse_args()

    # Dynamic output directory based on model name and hyperparameters
    model_output_dir = f"{args.output_dir}/{args.model_name.replace('/', '-')}_pie"
    if args.lora:
        model_output_dir += f"_r{args.lora_rank}"
    if args.quantization:
        model_output_dir += f"_q{args.quantization_bits}"
    if args.use_mixed_precision:
        model_output_dir += "_bf16"

    dataset = d.load_dataset(args.dataset_name)
    dataset = dataset.filter(lambda x: x['language'] == 'en')
    dataset = dataset.remove_columns(['target_text', 'span_labels', 'mbert_text_tokens', 'mbert_bio_labels', 'id', 'language', 'set'])
    dataset = dataset['train']
    dataset = dataset.train_test_split(test_size=0.2, seed=24, shuffle=True)
    temp = dataset['test']
    temp = temp.train_test_split(test_size=0.5, seed=9)
    dataset['val'] = temp['train']
    dataset['test'] = temp['test']
    print(dataset)

    accelerator = a.Accelerator(mixed_precision="bf16" if args.use_mixed_precision else None, gradient_accumulation_steps=args.gradient_accumulation_steps)

    tokenizer = t.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = None
    if args.quantization:
        if args.quantization_bits == "4":
            bnb_config = t.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        elif args.quantization_bits == "8":
            bnb_config = t.BitsAndBytesConfig(load_in_8bit=True)

    model = t.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={'': accelerator.process_index},
        dtype=torch.bfloat16 if args.use_mixed_precision else torch.float32,
        trust_remote_code=True
    )

    if args.quantization:
        model = p.prepare_model_for_kbit_training(model)
    model.config.pretraining_tp = 1
    model.config.pad_token_id = model.config.eos_token_id

    lora_config = None
    if args.lora:
        lora_config = p.LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])

    sft_config = r.SFTConfig(
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=model_output_dir,
        eval_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.eos_token,
        warmup_ratio=0.05,
        save_strategy='epoch',
        seed=24,
        bf16=args.use_mixed_precision,
        dataloader_num_workers=32,
        dataloader_prefetch_factor=64,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim='adamw_bnb_8bit',
        logging_strategy='epoch',
        lr_scheduler_type=args.lr_scheduler_type,
        ddp_find_unused_parameters=False,
    )

    early_stopping_callback = t.EarlyStoppingCallback(early_stopping_patience=args.patience)

    collator = r.trainer.sft_trainer.DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id, completion_only_loss=args.completion_only_loss)
    trainer = r.SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        args=sft_config,
        data_collator=collator,
        peft_config=lora_config,
        formatting_func=formatting_function,
        callbacks=[early_stopping_callback],
    )
    trainer.train()
    trainer.save_model(sft_config.output_dir)

    # ------------------ ADDED CODE START ------------------ #
    print("\n\n*** Training complete. Performing final evaluation on the test set. ***\n")

    # Reload the best model from the saved output directory
    print(f"Loading best model from {sft_config.output_dir}...")
    final_model = t.AutoModelForCausalLM.from_pretrained(
        sft_config.output_dir,
        quantization_config=bnb_config if args.quantization else None,
        device_map={'': accelerator.process_index},
        trust_remote_code=True,
        local_files_only=True # Prevents re-downloading
    )

    # If LoRA was used, load the adapter
    if args.lora:
        print("Applying LoRA adapter...")
        final_model = p.PeftModel.from_pretrained(final_model, sft_config.output_dir, is_trainable=False)
        final_model = final_model.merge_and_unload() # Merge LoRA weights for inference

    testing_args = t.TrainingArguments(
        output_dir=f"{sft_config.output_dir}/eval_results",
        per_device_eval_batch_size=args.per_device_train_batch_size * 2,
    )

    # Create a new trainer for evaluation with the reloaded model
    eval_trainer = t.Trainer(
        model=final_model,
        args=testing_args,
        data_collator=collator,
        eval_dataset=dataset['test'],
    )

    # Perform evaluation
    test_results = eval_trainer.evaluate()
    print("\n\n*** Test set evaluation results: ***")
    print(test_results)

if __name__ == "__main__":
    main()
