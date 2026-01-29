from datasets import load_from_disk
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import math

def compute_results(model, tokenizer, prefixes, labels, minibatch_size=8):
    
    device = model.device
    all_results = []

    for i in tqdm(range(0, len(prefixes), minibatch_size), desc="Computing"):
        prefix_batch = prefixes[i:i + minibatch_size]
        label_batch = labels[i:i + minibatch_size]

        full_texts = [p + l for p, l in zip(prefix_batch, label_batch)]
        encodings = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        prefix_encodings = tokenizer(prefix_batch, return_tensors="pt", padding=True, truncation=True)
        prefix_lengths = prefix_encodings["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1).to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        for j in range(len(prefix_batch)):
            prefix_len = prefix_lengths[j].item()
            label_token_log_probs = token_log_probs[j, prefix_len - 1:]  # From first label token onward
            label_mask = shift_mask[j, prefix_len - 1:]

            total_log_likelihood = label_token_log_probs[label_mask.bool()].sum().item()
            token_count = label_mask.sum().item()

            avg_log_likelihood = total_log_likelihood / token_count if token_count > 0 else float("nan")
            perplexity = math.exp(-avg_log_likelihood) if token_count > 0 else float("inf")

            all_results.append({
                "prefix": prefix_batch[j],
                "label": label_batch[j],
                "log_likelihood": total_log_likelihood,
                "avg_log_likelihood": avg_log_likelihood,
                "perplexity": perplexity,
                "label_token_count": token_count
            })

    return all_results


def main():

    model_path = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )
    model.eval()

    dataset = load_from_disk("")

    prefixes = dataset["input"]
    labels = dataset["label"]
    batch_size = 64

    results = compute_results(
        model, tokenizer, prefixes, labels, batch_size
    )

    df = pd.DataFrame(results)
    df.to_csv(f"{model_path.replace('/','-')}-new.csv", index=False)
    print(f"Resutls saved to {model_path.replace('/','-')}-new.csv")

if __name__ == "__main__":
    main()
