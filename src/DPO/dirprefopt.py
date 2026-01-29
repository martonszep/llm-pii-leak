# Script to fine-tune a language model using Direct Preference Optimization (DPO)
# Valid for all 3 datasets used in our experiments

from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_from_disk
import os

set_seed(42)
model_name = "meta-llama/LLaMa-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_from_disk("DPO_DS")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

for name, param in model.named_parameters():
    if 'lora' in name.lower():
        param.requires_grad = True
    else:
        param.requires_grad = False 

config = DPOConfig(
    beta=0.01,  
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-7,
    max_length=4096,
    num_train_epochs=5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=250,
    save_total_limit=2,
    report_to="wandb",
    run_name="DPO_DS",
    load_best_model_at_end=False,
    eval_steps=250,
    output_dir="dpo_output",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None, 
    args=config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

path = os.path.join(config.run_name, "final_model")
trainer.train()
trainer.save_model(path)