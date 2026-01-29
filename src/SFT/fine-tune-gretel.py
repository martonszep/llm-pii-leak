#This script was used originally for fine-tuning the first llama models
#using the DS dataset to evaluate the performance of the model

import torch, logging, wandb, os
from dotenv import load_dotenv
from datasets import load_from_disk, Dataset
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
from sklearn.model_selection import train_test_split

def prepare_dataset(tokenizer):
    dataset = load_from_disk(f'datasets/{DATASET_NAME}')
    
    train_df, val_df = train_test_split(
        dataset['train'].to_pandas(),
        test_size=0.05,
        random_state=RANDOM_SEED
    )

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    document_types = sorted(set(train_dataset['document_type']))

    def format_instruction(examples):
        texts = []
        for text, label in zip(examples['generated_text'], examples['document_type']):
            formatted = f"""Classify the following text into one of the following labels: 
{', '.join(document_types)}

Text:
{text}

Document type: {label}"""
            texts.append(formatted)
        return {'text': texts}
    
    def tokenize_function(examples):
        labels = tokenizer(examples["text"].split('\nDocument type: ')[1])
        labels_len = len(labels['input_ids'])
        output = tokenizer(
            examples["text"]
        )
        output["labels"] = [-100] * len(output["input_ids"][:-labels_len])
        output["labels"] = output["labels"] + labels['input_ids']
        return output
    
    columns_to_remove = [
        'level_0', 'index', 'document_type', 'document_description', 
        'expanded_type', 'expanded_description', 'language', 
        'language_description', 'domain', 'generated_text', 
        'pii_spans', 'conformance_score', 'quality_score', 
        'toxicity_score', 'bias_score', 'groundedness_score'
    ]
    
    train_dataset = train_dataset.map(
        format_instruction,
        batched=True,
        remove_columns=columns_to_remove
    )
    train_dataset = train_dataset.map(
        tokenize_function,
        # batched=True,
        remove_columns=["text"]
    )

    val_dataset = val_dataset.map(
        format_instruction,
        batched=True,
        remove_columns=columns_to_remove
    )
    val_dataset = val_dataset.map(
        tokenize_function,
        # batched=True,
        remove_columns=["text"]
    )

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])    
    
    return train_dataset, val_dataset


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, val_dataset = prepare_dataset(tokenizer)
    
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

    training_args = SFTConfig(
      output_dir="./fine_tuned_model",
      max_seq_length=1024,
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
      eval_strategy="steps",
      logging_steps=10,
      eval_steps=500,
      save_strategy="steps",
      save_total_limit=2,
      report_to="wandb",
      run_name=RUN_NAME,
      metric_for_best_model="eval_loss",
      load_best_model_at_end=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    early_stopping = EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)
    torch.cuda.empty_cache()
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=data_collator,
        callbacks=[early_stopping]
    )

    wandb.init(
        project="DS",
        name=RUN_NAME,
        config={
            "model_name": MODEL_NAME,
            "dataset": DATASET_NAME,
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
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    DATASET_NAME = "gretelai_synthetic_pii_finance_multilingual_curated"
    RUN_NAME = "Gretel-FT"
    NUM_EPOCHS = 5
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
    RANDOM_SEED = 42
    set_seed(RANDOM_SEED)
    load_dotenv()
    HF_KEY = os.getenv("HF_KEY")
    WANDB_KEY = os.getenv("WANDB_KEY")
    login(HF_KEY)
    wandb.login(key=WANDB_KEY)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()