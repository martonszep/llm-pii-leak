# Description: This script evaluates the model on the test set of datasets 
# based on the gretelai-synthetic-multilingual and saves the results to a CSV file.
# Best prompt template is number 0, we keep old 2 for reference.

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from huggingface_hub import login

load_dotenv()
HF_KEY = os.getenv("HF_KEY")

login(HF_KEY)
dataset = load_from_disk("datasets/gretelai_synthetic_pii_finance_multilingual_curated")
model_path = 'undial/AT-GretelAiComplete-v2'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')

candidate_labels = list(set(dataset['train']['document_type']))
print(candidate_labels)


def batch_classify_documents(batch_texts, model, tokenizer, candidate_labels, prompt_template=0):
    # Create prompts for all texts in the batch
    if prompt_template == 0:
        prompts = [
          f"Classify the following text into one of the following labels:\n"
          f"{', '.join(candidate_labels)}\n\n"
          f"Text:\n{text}\n\n"
          f"Document type:"
          for text in batch_texts
        ]
    elif prompt_template == 1:
        prompts = [
          f"### Instruction:\n"
          f"Classify the following text into exactly one of these categories: {', '.join(candidate_labels)}.\n\n"
          f"### Input:\n"
          f"{text}\n\n"
          f"### Response:\n"
          for text in batch_texts
        ]
    else:
        prompts = [
          f"<s>[INST] Classify the following text into exactly one of these categories: {', '.join(candidate_labels)}.\n\n"
          f"Text: {text} [/INST]"
          for text in batch_texts
        ]

    # Tokenize all prompts at once
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(model.device)

    # Generate predictions for the batch
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model.generate(
          **inputs,
          max_new_tokens=10,
          pad_token_id=tokenizer.pad_token_id,
          num_return_sequences=1,
          do_sample=False,
          top_p=0
        )

    # Decode all outputs at once
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Extract predictions
    if prompt_template == 0:
        split_string = "Document type:"
    elif prompt_template == 1:
        split_string = "### Response:"
    else:
        split_string = "[/INST]"

    predictions = [
        text.split(split_string)[-1].strip().split('\n')[0].split(',')[0]
        for text in generated_texts
    ]

    # Clear GPU memory after processing
    del inputs, outputs
    torch.cuda.empty_cache()

    return predictions

def evaluate_model(test_data, model, tokenizer, candidate_labels, batch_size=32, template=0):
    # Create DataLoader for batching
    dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    all_predictions = []
    all_true_labels = []

    # Process batches
    try:
        for batch in tqdm(dataloader):
            texts = batch['generated_text']
            true_labels = batch['document_type']

            # Get predictions for the batch
            batch_predictions = batch_classify_documents(
                texts,
                model,
                tokenizer,
                candidate_labels,
                prompt_template=template
            )

            all_predictions.extend(batch_predictions)
            all_true_labels.extend(true_labels)
    except Exception as e:
        print(e)

    return all_predictions, all_true_labels

# Run evaluation
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = 'left'

def filter_columns(dataset):
    # Function to keep only the required columns in each split
    return dataset.remove_columns([col for col in dataset.column_names if col not in ['generated_text', 'document_type']])

# Apply to all splits in the dataset
dataset = dataset.map(
    lambda x: x,  # Identity function as we're just using this to trigger the remove_columns
    remove_columns=[col for col in dataset['train'].column_names if col not in ['generated_text', 'document_type']]
)

split = 'test'
test_data = dataset[split]
# test_data = test_data.filter(lambda x: x['document_type'] == 'Email')
print(f"Filtered train dataset size: {len(test_data)}")
batch_size = 8  # Adjust based on your GPU memory

# Evaluate using different templates
for template_id in [0]:  # Original = 0, Alpaca = 1, ChatML = 1
    print(f"\nEvaluating Template {template_id}")
    predictions, true_labels = evaluate_model(
        test_data,
        model,
        tokenizer,
        candidate_labels,
        batch_size=batch_size,
        template=template_id
    )
    
    # Save results for this template
    pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': predictions
    }).to_csv(f'{model_path.replace("/","-")}_results_{split}_{template_id}.csv', index=False)
    