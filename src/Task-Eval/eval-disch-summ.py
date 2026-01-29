import json, os
from pathlib import Path
from datasets import load_from_disk
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from dotenv import load_dotenv
from huggingface_hub import login
from evaluate import load

def generate_prompt(example):
    prompt = (
        f'You are a medical language model trained to write the "Procedere" section of a German-language discharge summary from an orthopedic clinic.'
        f'Based on the following clinical information, you are to generate the text for the "Procedere" section.'
        f'The output must be medically accurate, written in German, and adhere to the standard clinical writing style.\n'
        f'<BEGINNING OF CLINICAL INFORMATION>\n{example}\n<END OF CLINICAL INFORMATION>\n'
        f'Please now generate the section.'
        f'Procedere: '
    )
    return prompt

def compute_perplexity(model, dataloader, device, max_length=2048):
    model.to(device).eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                max_new_tokens=512
            )
            # outputs.loss: average per non-masked token
            loss = outputs.loss.item()
            num_tokens = (labels != -100).sum().item()

            total_loss += loss * num_tokens
            total_tokens += num_tokens

    avg_nll = total_loss / total_tokens
    return float(torch.exp(torch.tensor(avg_nll)))


if __name__ == '__main__':

    load_dotenv()
    HF_KEY = os.getenv("HF_KEY")
    WANDB_KEY = os.getenv("WANDB_KEY")
    login(HF_KEY)
  
    model_path = 'meta-llama/Llama-3.2-1B'
    dataset_name = "DS"
    batch_size = 8
    max_length = 2048
    output_dir = 'eval_results'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Load dataset
    dataset = load_from_disk(dataset_name)['test']

    # Prepare prompt + label fields
    def apply_prompt(example):
        prompt = generate_prompt(example['input_text'])
        return {'prompt': prompt, 'label': example['label']}
    dataset = dataset.map(apply_prompt, remove_columns=[c for c in dataset.column_names if c not in ['prompt', 'label']])

    def tokenize_for_ppl(example):
        enc = tokenizer(
            example['prompt'] + example['label'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
        )
        
        enc['labels'] = [tid if tid != tokenizer.pad_token_id else -100 for tid in enc['input_ids']]
        return enc

    ppl_ds = dataset.map(tokenize_for_ppl, batched=False, remove_columns=['label','prompt'])
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    ppl_loader = DataLoader(ppl_ds, batch_size=batch_size, collate_fn=collator)
    perplexity = compute_perplexity(model, ppl_loader, device)
    print(f"Perplexity: {perplexity:.2f}")

    # ================================
    # 2) Generation
    # ================================
    def tok_batch(examples):
    # Tokenize prompts in batch; return tensors in map results
        return tokenizer(
            examples['prompt'],
            truncation=True,
            max_length=max_length,
            padding='longest',
        )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    gen_ds = dataset.map(
        tok_batch,
        batched=True,
        remove_columns=['label', 'prompt'],
    )
    gen_loader = DataLoader(
        gen_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    all_preds = []
    model.to(device).eval()
    with torch.no_grad():
        for batch in gen_loader:
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            outs = model.generate(input_ids=input_ids,
                                  attention_mask=attn,
                                  max_new_tokens=512
                                  )
            texts = tokenizer.batch_decode(outs, skip_special_tokens=True)
            # strip off the prompt prefix to get only the generated reply
            for inp, full in zip(batch['input_ids'], texts):
                prompt = tokenizer.decode(inp, skip_special_tokens=True)
                reply = full[len(prompt):]
                all_preds.append(reply)

    # Save generated texts and labels to a DataFrame
    try:
        df = pd.DataFrame({
            "prompt": dataset['prompt'],
            "generated_text": all_preds,
            "reference_text": dataset['label']
        })

        # Save the DataFrame to a JSON file
        df.to_json(f'{output_dir}/{model_path.split("/")[0]}_generation-eng.json', orient="records", indent=2)
        print(f'Saved generation results to {output_dir}/{model_path.split("/")[0]}_generation.json')
    except Exception as e:
        print(f"Error saving generation results: {e}")
    # ================================
    # 3) Metrics: BLEU, ROUGE, BERTScore
    # ================================
    refs = [[lbl] for lbl in dataset['label']]
    preds = all_preds

    bleu = load('bleu').compute(predictions=preds, references=refs)
    rouge = load('rouge').compute(predictions=preds, references=refs)
    bert = load('bertscore').compute(predictions=preds, references=refs, lang='de')
    
    print(f"Perplexity: {perplexity:.2f}")
    print(f"BLEU: {bleu['bleu']:.4f}")
    print(f"ROUGE-L: {rouge['rougeL']:.4f}")
    # print(f"BertScore: {bert}")
    print(f"BERTScore Precision: {sum(bert['precision'])/len(bert['precision']):.4f}")
    print(f"BERTScore Recall: {sum(bert['recall'])/len(bert['recall']):.4f}")
    print(f"BERTScore F1: {sum(bert['f1'])/len(bert['f1']):.4f}")

    # ================================
    # 4) Save detailed results for manual analysis
    # ================================
    bert_f1 = sum(bert['f1'])/len(bert['f1'])
    metrics = {
      'bleu': bleu['bleu'],
      'rougeL': rouge['rougeL'],
      'bert_f1': bert_f1,
      'perplexity': perplexity,
    }
    with open(f'{output_dir}/{model_path.split("/")[0]}-eng.json', 'w') as f:
        json.dump(metrics, f, indent=2)
