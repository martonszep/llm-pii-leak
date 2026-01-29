# FT for ortho dataset using adding dynamic padding 
# to save memory and speed up training usng Seq2Seq

import torch, wandb, os
from dotenv import load_dotenv
from datasets import load_from_disk
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
    BitsAndBytesConfig

)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils.TumorClassification import TumorClassificationSimple

def generate_prompt(example, input_language, task_description, schema):
    """
    Given one example (with a pathology report in example['text'] and a label in example['label']),
    generate a prompt that instructs the model to extract tumor-related information.
    """
    prompt = (
    f"You are a helpful assistant. Your task is to extract the following tumor related information from a {input_language}"
    f"bone and soft tissue tumor pathology report:\n"
    f"{task_description}\n"
    f"Besides classifying the tumor based on the given criteria, you should also retrieve the text sequences that contain "
    f"the tumor information. The tumor information should be classified according to the following schema:\n"
    f"{schema}\n"
    f"Retrieve and extract the relevant information from the following report:\n"
    f"{example['text']}\n\n"
    )
    return prompt

def preprocess_function(example, tokenizer, input_language, task_description, schema):
    """
    Build the training instance. We concatenate a prompt (constructed from the pathology report and instructions)
    with the expected response (the JSON label). Then, we tokenize the full text and set the label values corresponding
    to the prompt part to -100 so that the loss is computed only on the response.
    """
    prompt = generate_prompt(example, input_language, task_description, schema)
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

    model_name = "meta-llama/Llama-3.2-1B"
    dataset_path = "Patho"
    RUN_NAME = "patho-ft"
    NUM_EPOCHS = 25
    PER_DEVICE_TRAIN_BATCH_SIZE = 6
    GRADIENT_ACCUMULATION_STEPS = 10
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

    input_language = "German"
    
    with open("prompt_eng_new.txt", "r") as f:
        task_description = f.read()
    schema = TumorClassificationSimple.model_json_schema()

    dataset = load_from_disk(f"{dataset_path}")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BNB_CONFIG,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    
    def tokenize_example(example):
        return preprocess_function(example, tokenizer, input_language, task_description, schema)

    tokenized_train_dataset = train_dataset.map(tokenize_example, batched=False, remove_columns=["label", "text"])
    tokenized_val_dataset = val_dataset.map(tokenize_example, batched=False, remove_columns=["label", "text"])

    training_args = SFTConfig(
        output_dir="./fine_tuned_model",
        overwrite_output_dir=True,
        max_seq_length=10000,
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
        project="patho-ft",
        name=RUN_NAME,
        config={
            "model_name": model_name,
            "dataset": dataset_path,
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
