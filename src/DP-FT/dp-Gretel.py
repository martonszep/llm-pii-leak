# DPFT with distributed training and differential privacy

import torch, wandb, os
from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from sklearn.model_selection import train_test_split
import bitsandbytes as bnb
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
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
import torch.distributed as dist


def get_dataloaders(tokenizer):
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

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer
    )
    
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

def train_epoch(epoch, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device, local_rank):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_dataloader):
        if step % 100 == 0:
            torch.cuda.synchronize()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        # Log training progress from the main process only
        if step % LOGGING_STEPS == 0 and local_rank == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": lr_scheduler.get_last_lr()[0],
                "train/step": step,
                "train/epoch": epoch
            })

        if step % EVAL_STEPS == 0 and local_rank == 0:
            val_loss = evaluate(model, val_dataloader, device)
            wandb.log({
                "eval/loss": val_loss,
                "eval/epoch": epoch,
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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)

    model.to(device)
    model = DPDDP(model)

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
    # model = torch.compile(model)


    best_val_loss = float('inf')
    early_stop_count = 0
    if local_rank == 0:
        wandb.login(key=WANDB_KEY)
        wandb.init(
            project="LLaMa-3.2-1B-finance_multilingual-FT",
            name=RUN_NAME,
            config={
                "run_name": RUN_NAME,
                "model_name": MODEL_ID,
                "dataset": DATASET_NAME,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "batch_size": MAX_PHYSICAL_BATCH_SIZE,
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

         # Before distributed training, add some diagnostics
        print(f"Local Rank: {local_rank}")
        print(f"Total GPUs: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
    

   
    with BatchMemoryManager(
        data_loader=train_dataloader,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer
    ) as memory_safe_data_loader:
        
        # We create the scheduler and define the training steps here to ensure that the number of training steps is correct
        # BatchMemoryManager can change the number of training steps. It automatically handles the optimizer, but not the lr
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
                epoch, model, memory_safe_data_loader, val_dataloader,
                optimizer, lr_scheduler, device, local_rank
            )
        
            dist.barrier()
            val_loss = evaluate(model, val_dataloader, device)
            
            if local_rank == 0:
                wandb.log({
                    "train/epoch_loss": train_loss,
                    "eval/epoch_loss": val_loss,
                    "epoch": epoch
                })
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_count = 0
                    checkpoint_dir = f"{OUTPUT_DIR}/{RUN_NAME}/checkpoint-best"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    print("Saving checkpoint")
                    torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
                else:
                    early_stop_count += 1
                    if early_stop_count >= EARLY_STOPPING_PATIENCE:
                        break
                
                
                try:
                    print(f"Epoch {epoch} | Train Loss: {train_loss} | Eval Loss: {val_loss} | Epsilon {privacy_engine.get_epsilon(delta=DELTA)}")
                except:
                    print(f"Epoch {epoch} completed")

            dist.barrier()

    if local_rank == 0:
        print("Saving final model")
        final_dir = f"{OUTPUT_DIR}/{RUN_NAME}/final_model"
        os.makedirs(final_dir, exist_ok=True)
        model.module.save_pretrained(final_dir)

        try:
            epsilon = privacy_engine.get_epsilon(delta=DELTA)
            print(f"Final Epsilon: {epsilon:.2f}, Delta: {DELTA}\n")
            with open(os.path.join(f"{OUTPUT_DIR}/{RUN_NAME}", "dp_stats.txt"), "w") as f:
                f.write(f"Final Epsilon: {epsilon:.2f}, Delta: {DELTA}\n")
        except Exception as e:
            print("Error writing epsilon value", e)

    dist.destroy_process_group()

if __name__ == "__main__":
    load_dotenv()
    HF_KEY = os.getenv("HF_KEY")
    WANDB_KEY = os.getenv("WANDB_KEY")

    NUM_EPOCHS = 5
    MAX_PHYSICAL_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 16
    WARMUP_RATIO = 0.03
    MAX_GRAD_NORM = 1.0
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0
    LOGGING_STEPS = 10
    EVAL_STEPS = 250
    EARLY_STOPPING_PATIENCE = 3
    TARGET_EPSILON = 2.0
    DELTA = 1e-5
    POISSON_SAMPLING = True
    OUTPUT_DIR = "custom-ft"
    MODEL_ID = "meta-llama/Llama-3.2-1B"
    DATASET_NAME = "gretelai_with_enron_5class"
    RUN_NAME = "LR_Test"
    RANDOM_SEED = 42

    BNB_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    LORA_CONFIG = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    set_seed(RANDOM_SEED)
    try:
        main()
    except Exception as e:
          print(f"An error occurred: {e}")
    finally:
        # Ensure process group is destroyed even if an exception occurs
        if dist.is_initialized():
            dist.destroy_process_group()