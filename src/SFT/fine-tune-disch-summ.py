#This script was used originally for fine-tuning the first llama models
#using the DS dataset to evaluate the performance of the model

import torch, logging, wandb, os
from dotenv import load_dotenv
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

def generate_prompt(example):
    prompt = (
        f'You are a medical language model trained to write the "Procedere" section of a German-language discharge summary.'
        f'Based on the following clinical information, you are to generate the text for the "Procedere" section.'
        f'The output must be medically accurate, written in German, and adhere to the standard clinical writing style.\n'
        f'<BEGINNING OF CLINICAL INFORMATION>\n{example}\n<END OF CLINICAL INFORMATION>\n'
        f'Please now generate the section.\n'
        f'Procedere: '
    )
    return prompt

def preprocess_function(example, tokenizer):
    prompt = generate_prompt(example["input_text"])
    response = example['label'].strip() + tokenizer.eos_token
    full_text = prompt + response

    tokenized_full = tokenizer(full_text)
    tokenized_prompt = tokenizer(prompt)
    prompt_len = len(tokenized_prompt["input_ids"])

    labels = tokenized_full["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    tokenized_full["labels"] = labels

    if "label" in tokenized_full:
        del tokenized_full["label"]

    return tokenized_full



def main():

    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    DATASET_NAME = "DS"
    RUN_NAME = "DS-FT"
    NUM_EPOCHS = 25
    PER_DEVICE_TRAIN_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 16
    LEARNING_RATE = 5e-5
    EARLY_STOP_PATIENCE = 3
    BNB_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    LORA_CONFIG = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    set_seed(42)
    load_dotenv()
    HF_KEY = os.getenv("HF_KEY")
    WANDB_KEY = os.getenv("WANDB_KEY")
    login(HF_KEY)
    wandb.login(key=WANDB_KEY)
    dataset = load_from_disk(DATASET_NAME)
    
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    train_dataset.remove_columns(["report", "piis"])
    val_dataset.remove_columns(["report", "piis"])

    logging.info(f"Training set size: {len(dataset['train'])}")
    logging.info(f"Validation set size: {len(dataset['validation'])}")
    logging.info(f"Test set size: {len(dataset['test'])}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BNB_CONFIG,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    
    def tokenize_example(example):
        return preprocess_function(example, tokenizer)

    tokenized_train_dataset = train_dataset.map(tokenize_example, batched=False, remove_columns=["label"])
    tokenized_val_dataset = val_dataset.map(tokenize_example, batched=False, remove_columns=["label"])

    training_args = SFTConfig(
        output_dir="./fine_tuned_model",
        max_seq_length=2048,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_checkpointing=True,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,
        optim="paged_adamw_32bit", #8bit
        dataloader_num_workers=4,
        bf16=True,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        logging_steps=10,
        eval_steps=500,
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        save_strategy="steps",
        save_total_limit=2,
        report_to="wandb",
        run_name=RUN_NAME,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        use_liger=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    early_stopping = EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)
    torch.cuda.empty_cache()
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping]
    )

    wandb.init(
        project="DS",
        name=RUN_NAME,
        config={
            "model_name": MODEL_NAME,
            "dataset": dataset,
            "num_epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
        }
    )
    
    trainer.train()

    path = os.path.join(training_args.run_name, training_args.output_dir)
    trainer.save_model(path)
    print("Training complete and model saved.")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()