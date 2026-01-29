# Custom DP-FT training loop with Differential Privacy

import torch, wandb, os
from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from datasets import load_from_disk
import bitsandbytes as bnb
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    get_scheduler,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

def save_checkpoint(wrapped_model, tokenizer, path):
    """
    Save a checkpoint of the model, handling the DP wrapper appropriately
    """
    os.makedirs(path, exist_ok=True)
    
    # Get the original model from the DP wrapper
    original_model = wrapped_model._module if hasattr(wrapped_model, '_module') else wrapped_model
    
    # Save the model state
    original_model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def generate_prompt(example):
    prompt = (
        f'You are a medical language model trained to write the "Procedere" section of a German-language discharge summary from an orthopedic clinic.'
        f'Based on the following clinical information, you are to generate the text for the "Procedere" section.'
        f'The output must be medically accurate, written in German, and adhere to the standard clinical writing style.\n'
        f'<BEGINNING OF CLINICAL INFORMATION>\n{example}\n<END OF CLINICAL INFORMATION>\n'
        f'Please now generate the section.\n'
        f'Procedere: '
    )
    return prompt

def preprocess_function(example, tokenizer):
    prompt = generate_prompt(example['input_text'])
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


def get_dataloaders(tokenizer):
    dataset = load_from_disk(DATASET_NAME)

    train_dataset = dataset['train']
    val_dataset = dataset['validation']

    def tokenize_example(example):
      return preprocess_function(example, tokenizer)

    train_dataset = train_dataset.map(tokenize_example, batched=False, remove_columns=["label", "report", "piis", "input_text", "file", "id", "__index_level_0__"])
    val_dataset = val_dataset.map(tokenize_example, batched=False, remove_columns=["label", "report", "piis", "input_text", "file", "id", "__index_level_0__"])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=MAX_PHYSICAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        collate_fn=data_collator,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=MAX_PHYSICAL_BATCH_SIZE,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4
    )
    
    return train_dataloader, val_dataloader

def train_epoch(epoch, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device, tokenizer):
    model.train()
    total_loss = 0.0
    running_loss = 0.0
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        running_loss += loss.item()
        total_loss += loss.item()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        # Log training progress
        if step > 0 and step % LOGGING_STEPS == 0:
            avg_loss = running_loss / LOGGING_STEPS
            wandb.log({
                "train/loss": avg_loss,
                "train/lr": lr_scheduler.get_last_lr()[0],
                "train/step": step,
                "train/epoch": epoch + (step/len(train_dataloader))
            })
            running_loss = 0.0

        if step % EVAL_STEPS == 0:
            val_loss = evaluate(model, val_dataloader, device)
            wandb.log({
                "eval/loss": val_loss,
                "eval/epoch": epoch + (step/len(train_dataloader)),
                "eval/step": step
            })
            model.train()
    
    return total_loss / len(train_dataloader)

def evaluate(model, val_dataloader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    
    return total_loss / len(val_dataloader)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda")

    model = AutoModelForCausalLM.from_pretrained(
      MODEL_NAME,
      quantization_config=BNB_CONFIG,
      torch_dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
      use_cache=False,
      device_map="auto"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    # Prepare data loaders with input language, task description, and schema
    train_dataloader, val_dataloader = get_dataloaders(tokenizer)

    optimizer = bnb.optim.PagedAdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    privacy_engine = PrivacyEngine()
    model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
      module=model,
      optimizer=optimizer,
      data_loader=train_dataloader,
      epochs=NUM_EPOCHS,
      target_epsilon=TARGET_EPSILON,
      target_delta=DELTA,
      max_grad_norm=MAX_GRAD_NORM,
      poisson_sampling=POISSON_SAMPLING
    )

    wandb.init(
      project="DS",
      name=RUN_NAME,
      config={
          "run_name": RUN_NAME,
          "model_name": MODEL_NAME,
          "dataset": DATASET_NAME,
          "num_epochs": NUM_EPOCHS,
          "learning_rate": LEARNING_RATE,
          "batch_size": MAX_PHYSICAL_BATCH_SIZE,
          "grad_accumulation": GRADIENT_ACCUMULATION_STEPS,
          "optimizer": {
              "name": optimizer.__class__.__name__,
              "lr": LEARNING_RATE,
              "weight_decay": WEIGHT_DECAY
          },
          "lr_scheduler": {
              "name": "cosine",
              "warmup_ratio": WARMUP_RATIO
          },
          "quantization": {
              "load_in_4bit": BNB_CONFIG.load_in_4bit,
              "quant_type": BNB_CONFIG.bnb_4bit_quant_type,
              "compute_dtype": str(BNB_CONFIG.bnb_4bit_compute_dtype),
              "double_quant": BNB_CONFIG.bnb_4bit_use_double_quant
          },
          "lora": {
              "r": LORA_CONFIG.r,
              "lora_alpha": LORA_CONFIG.lora_alpha,
              "target_modules": LORA_CONFIG.target_modules,
              "lora_dropout": LORA_CONFIG.lora_dropout,
              "bias": LORA_CONFIG.bias,
              "task_type": LORA_CONFIG.task_type
          },
          "privacy": {
              "epsilon": TARGET_EPSILON,
              "delta": DELTA,
              "max_grad_norm": MAX_GRAD_NORM,
              "poisson_sampling": POISSON_SAMPLING
          }
      }
    )
   
    with BatchMemoryManager(
      data_loader=train_dataloader,
      max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
      optimizer=optimizer
    ) as memory_safe_data_loader:
        
    # We create the scheduler and define the training steps here to ensure that the number of training steps is correct
        num_training_steps = (
        len(memory_safe_data_loader) * NUM_EPOCHS
        )
        lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * WARMUP_RATIO),
        num_training_steps=num_training_steps
        )
        
        for epoch in tqdm(range(NUM_EPOCHS)):
            print("Starting new epoch")
            train_loss = train_epoch(
                epoch+1, model, memory_safe_data_loader, val_dataloader,
                optimizer, lr_scheduler, device, tokenizer
            )
            val_loss = evaluate(model, val_dataloader, device)
            
            wandb.log({
                "train/epoch_loss": train_loss,
                "eval/epoch_loss": val_loss,
                "epoch": epoch
            })
            
            try:
                print(f"Epoch {epoch} | Train Loss: {train_loss} | Eval Loss: {val_loss} | Epsilon {privacy_engine.get_epsilon(delta=DELTA)}")
            except:
                print(f"Epoch {epoch} completed")

    print("Saving final model")
    final_dir = f"{OUTPUT_DIR}/{RUN_NAME}/final_model"
    save_checkpoint(model, tokenizer, final_dir)
    os.makedirs(final_dir, exist_ok=True)

    try:
      epsilon = privacy_engine.get_epsilon(delta=DELTA)
      print(f"Final Epsilon: {epsilon:.2f}, Delta: {DELTA}\n")
      with open(os.path.join(final_dir, "dp_stats.txt"), "w") as f:
        f.write(f"Final Epsilon: {epsilon:.2f}, Delta: {DELTA}\n")
    except Exception as e:
      print("Error writing epsilon value", e)

if __name__ == '__main__':
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    DATASET_NAME = "DS"

    RUN_NAME = "DP-FT"

    NUM_EPOCHS = 20
    MAX_PHYSICAL_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 1
    WARMUP_RATIO = 0.03
    MAX_GRAD_NORM = 1.0
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0
    LOGGING_STEPS = 10
    EVAL_STEPS = 500
    EARLY_STOP_PATIENCE = 3
    TARGET_EPSILON = 2.0
    DELTA = 1e-5
    POISSON_SAMPLING = False
    RANDOM_SEED = 42
    OUTPUT_DIR = "DP-FT-models"

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

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(RANDOM_SEED)
    load_dotenv()
    HF_KEY = os.getenv("HF_KEY")
    WANDB_KEY = os.getenv("WANDB_KEY")
    login(HF_KEY)
    wandb.login(key=WANDB_KEY)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
      main()
    except Exception as e:
      print(f"An error occurred: {e}")