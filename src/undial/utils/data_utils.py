import torch
import pandas as pd
import re
import os
import pickle
import numpy as np
# import spacy
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainerCallback
import datasets
import random
from datasets import load_dataset
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def load_extract_challenge_data(path):
    dataset_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    data = np.load(path).astype(np.float32)
    data = torch.from_numpy(data)
    data = dataset_tokenizer.batch_decode(data, padding=True, truncation=True, )
    return data

def find_sublist_indices(main_list, sublist):
    start_index = None
    end_index = None

    for i in range(len(main_list)):
        if main_list[i:i+len(sublist)] == sublist:
            start_index = i
            end_index = i + len(sublist) - 1
            break
        
    if start_index == None:
        return None
    else:
        key_indices = list(range(start_index, end_index+1))
    return key_indices

# AI-Optimized version of find_focus_idx_old
def find_focus_idx_new(text, tokenizer, nlp=None, focus_type="entity", focus_keywords=None, prefix_size=0):
    # 1) Build list of (char_start, char_end) spans for all keywords/entities
    spans = []
    if focus_type == "entity":
        if not focus_keywords:
            return []
        for kw in focus_keywords:
            start = 0
            while True:
                idx = text.find(kw, start)
                if idx < 0:
                    break
                spans.append((idx, idx + len(kw)))
                start = idx + len(kw)
    elif focus_type == "none":
        # Use spaCy noun-chunks
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            spans.append((chunk.start_char, chunk.end_char))
    else:
        raise NotImplementedError(f"Unknown focus_type {focus_type}")

    if not spans:
        return []

    # 2) Tokenize once, get offsets mapping tokens --> (char_start, char_end)
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
    )
    offsets = encoding["offset_mapping"]
    # offsets[0] is usually [0,0] for [CLS], so be mindful

    # 3) For each token, check if its char-range overlaps any focus span
    token_indices = set()
    for tok_idx, (tok_s, tok_e) in enumerate(offsets):
        # skip special tokens
        if tok_s == tok_e == 0:
            continue
        # any overlap?
        for span_s, span_e in spans:
            if tok_e > span_s and tok_s < span_e:
                token_indices.add(tok_idx)
                break

    if not token_indices:
        return []

    # 4) Expand each index by prefix_size
    all_idxs = set()
    for idx in token_indices:
        start = max(0, idx - prefix_size)
        all_idxs.update(range(start, idx + 1))

    return sorted(all_idxs)

# Old version of my implementation of find_focus_idx, kept for reference.
# Looks for tge focus_keywords in the text and returns their token indices to apply focus on UnDial's loss.
# def find_focus_idx_old(text, tokenizer, nlp, focus_type="entity", focus_keywords=None):
#     # Existing keyword detection logic
#     key_words = []
#     if focus_type == 'entity':
#         key_words = focus_keywords
#     elif focus_type == 'none':
#         for entity in doc.noun_chunks:  
#             key_words.append(str(entity.text))
#     else:
#         raise NotImplementedError("Focus type not implemented")

#     tokenized_text = tokenizer(text)['input_ids']
#     key_tokens = tokenizer(key_words)['input_ids']

#     key_idx = []
#     for key_token in key_tokens:
#         tmp = find_sublist_indices(tokenized_text, key_token[1:])
#         if tmp is not None:
#             key_idx += tmp
#     key_idx = list(set(key_idx))


#     if not key_idx:
#         return []

#     # Group consecutive indices into spans
#     sorted_indices = sorted(key_idx)
#     spans = []
#     current_span = [sorted_indices[0], sorted_indices[0]]
    
#     for idx in sorted_indices[1:]:
#         if idx == current_span[1] + 1:
#             current_span[1] = idx
#         else:
#             spans.append(tuple(current_span))
#             current_span = [idx, idx]
#     spans.append(tuple(current_span))

#     # Expand each span with prefix
#     expanded_indices = []
#     max_length = len(tokenized_text)
    
#     for start, end in spans:
#         prefix_start = max(0, start - 25)
#         expanded_indices.extend(range(prefix_start, end + 1))

#     # Remove duplicates and ensure order
#     expanded_indices = sorted(list(set(expanded_indices)))
    
#     return expanded_indices


def format_instruction(example, document_types):
    return f"""Classify the following text into one of the following labels: 
{', '.join(document_types)}

Text:\n{example['generated_text']}

Document type: {example['document_type']}"""

class GenEvalDataset(Dataset):
    def __init__(self, args, tokenizer, gen_data_type):
        if gen_data_type == "wikitext":
            data = datasets.load_dataset('wikitext', 'wikitext-103-raw-v1', split=f'train[:{args.eval_num}]', verification_mode='no_checks')
        if gen_data_type == "news":
            data = datasets.load_dataset("cc_news", split=f'train[:{args.eval_num}]', verification_mode='no_checks')
        if gen_data_type == "wikitext_valid":
            data = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split=f'validation[:1000]', verification_mode='no_checks')
        text_column_name = "text"
        assert text_column_name in data.column_names
        
        self.data = []
        for dp in tqdm(data, desc="Tokenizing data"):
            text = dp[text_column_name].replace(' <newline>', '\n')
            # text = tokenizer(text, add_special_tokens=False, return_tensors="pt")['input_ids'][0].to(args.device)
            input = tokenizer(text, add_special_tokens=False, return_tensors="pt")
            input_ids = input['input_ids'][0][:32]
            gold = input['input_ids'][0][32:160]
            attention_mask = torch.Tensor([1] * len(input_ids))
            if len(input['input_ids'][0]) < 160 or "=" in text:
                continue
            self.data.append({"input_ids": input_ids.long(), "attention_mask": attention_mask.long(), "gold": gold.long()})
        print(f"Generation Dataset size: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# class ExtractChallengeDataset(Dataset):
#     def __init__(self, args, eval_window_size, tokenizer, split):
#         self.data = []
#         self.test_data_path = os.path.join(args.cache_dir, f"test_window_{eval_window_size}_num_{args.eval_num}_filtered_{args.focus}.pkl")
#         self.train_data_path = os.path.join(args.cache_dir, f"train_num_{args.train_num}_focus_{args.focus}_type_{args.focus_type}.pkl")
#         self.raw_data_path = args.filtered_extract_challenge_data_path if (args.focus or args.focus_dataset) else args.extract_challenge_data_path
#         print("using filtered data" if (args.focus or args.focus_dataset) else "using raw data")
        
#         if args.focus:
#             nlp = spacy.load("en_core_web_sm")
            
#         if split == "train":
#             raw_data = load_extract_challenge_data(self.raw_data_path)
#             if not os.path.exists(self.train_data_path):
#                 for text in tqdm(raw_data):
#                     input = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=200, return_tensors="pt")
#                     if args.focus:
#                         try:
#                             focus_idx = find_focus_idx(text, tokenizer, nlp, focus_type=args.focus_type)
#                         except:
#                             focus_idx = []
#                             # print("Focus idx not found")
#                     else:
#                         focus_idx = []
#                     focus_idx = torch.Tensor(focus_idx + [0] * (200 - len(focus_idx)))
#                     self.data.append({"input_ids": input['input_ids'][0], "attention_mask": input['attention_mask'][0], "focus_idx": focus_idx})
#                 print(self.data[0])
#                 # Save the processed data to a file
#                 with open(self.train_data_path, 'wb') as f:
#                     pickle.dump(self.data, f)    
#             else:
#                 with open(self.train_data_path, 'rb') as f:
#                     self.data = pickle.load(f)
#             print(f"Train dataset size: {len(self.data)}")

        # elif split == "test":
        #     # raw_data = load_extract_challenge_data(self.raw_data_path)[:args.eval_num]
        #     raw_data = load_extract_challenge_data(self.raw_data_path)
        #     if not os.path.exists(self.test_data_path):
        #         tokenized_text = tokenizer(raw_data, padding=True, truncation=True, max_length=200)['input_ids']
        #         for dp in tqdm(tokenized_text):
        #             for i in range(1, len(dp) // eval_window_size):
        #                 prompt, completion = dp[ : i*eval_window_size], dp[i*eval_window_size : ]
        #                 if len(prompt) == 0 or len(completion) == 0:
        #                     continue
        #                 input_ids = torch.Tensor([tokenizer.pad_token_id] * (200 - len(prompt)) + prompt)
        #                 attention_masks = torch.Tensor([0] * (200 - len(prompt)) + [1] * len(prompt))
        #                 targets = torch.Tensor(completion + [tokenizer.pad_token_id] * (200 - len(completion)))
        #                 self.data.append({"input_ids": input_ids.long(), "attention_mask": attention_masks.long(), "targets": targets.long()})
        #         # Save the processed data to a file
        #         with open(self.test_data_path, 'wb') as f:
        #             pickle.dump(self.data, f)    
        #     else:
        #         with open(self.test_data_path, 'rb') as f:
        #             self.data = pickle.load(f)
        #     print(f"Test dataset size: {len(self.data)}")
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data[idx]

class DischSummDataset(Dataset):
    def generate_prompt(self, example):
        prompt = (
            f'You are a medical language model trained to write the "Procedere" section of a German-language discharge summary.'
            f'Based on the following clinical information, you are to generate the text for the "Procedere" section.'
            f'The output must be medically accurate, written in German, and adhere to the standard clinical writing style.\n'
            f'<BEGINNING OF CLINICAL INFORMATION>\n{example}\n<END OF CLINICAL INFORMATION>\n'
            f'Please now generate the section.'
            f'Procedere: '
        )
        return prompt

    def preprocess_function(self, example, tokenizer):
        prompt = self.generate_prompt(example['input_text'])
        response = example['label'].strip() + tokenizer.eos_token
        full_text = prompt + response


        tokenized_full = tokenizer(
            full_text,
            return_tensors="pt",              
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        tokenized_prompt = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )

        prompt_len = tokenized_prompt["input_ids"].shape[1]
        labels = tokenized_full["input_ids"].clone()
        labels[0, :prompt_len] = -100
        tokenized_full["labels"] = labels

        # return tokenized_full
        return {
            "input_ids": tokenized_full["input_ids"],
            "attention_mask": tokenized_full["attention_mask"],
            "labels": labels
        }

    def __init__(self, args, eval_window_size, tokenizer, split, max_length):
        # self.max_length = max_length
        self.max_length = 1536

        self.raw_data_path = '../DS'
        PIIS_PATH = 'datasets/undialAT-DS'

        self.data = []
        print("PreProcessing raw data...")
        piis_list = datasets.load_from_disk(PIIS_PATH)
        piis_list = piis_list['train']['text']
        print(f"Applying focus to {len(piis_list)} piis")
        piis_list = piis_list + [" " + key_word for key_word in piis_list]
    
        print("using filtered data" if (args.focus or args.focus_dataset) else "using raw data")
    
        if split in ["train", "validation"]:
            data = datasets.load_from_disk(self.raw_data_path)[split]
    
            print("Num samples", len(data))
            for sample in tqdm(data):
                encoding = self.preprocess_function(sample, tokenizer)
                if args.focus:
                    try:
                        focus_idx_list = find_focus_idx_new(
                            sample['input_text'],
                            tokenizer,
                            nlp if args.focus_type == "none" else None,
                            focus_type=args.focus_type,
                            focus_keywords=piis_list,
                            prefix_size=0,
                        )
                    except Exception as e:
                        focus_idx_list = []
                        print("Focus idx not found", e)
                else:
                    focus_idx_list = []

                # pad/truncate to max_length
                focus_idx = torch.zeros(self.max_length, dtype=torch.long)
                if focus_idx_list:
                    valid = torch.tensor(focus_idx_list).clamp(max=self.max_length-1)
                    focus_idx[valid] = 1


                self.data.append({
                    "input_ids": encoding["input_ids"][0],
                    "attention_mask": encoding["attention_mask"][0],
                    "labels": encoding["labels"][0],
                    "focus_idx": focus_idx,
                })

            print(self.data[0], len(self.data))
            print(self.data[0]["focus_idx"])

        elif split == "test":
            raise NotImplementedError
               
        print(f"Train dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class GretelDataset(Dataset):
    def __init__(self, args, eval_window_size, tokenizer, split, max_length):
        self.max_length = max_length
        self.test_data_path = os.path.join(args.cache_dir, f"test_window_{eval_window_size}_num_{args.eval_num}_filtered_{args.focus}.pkl")
        self.train_data_path = os.path.join(args.cache_dir, f"train_num_{args.train_num}_focus_{args.focus}_type_{args.focus_type}.pkl")
        # self.raw_data_path = args.filtered_extract_challenge_data_path if (args.focus or args.focus_dataset) else args.extract_challenge_data_path
        self.raw_data_path = '../datasets/gretelai_with_enron_5class'
        PIIS_PATH = 'datasets/gretelai_with_enron_5class_undial_unique_NoPref_30percent'
        
        self.data = []
        piis_list = datasets.load_from_disk(PIIS_PATH)
        piis_list = piis_list['train']['text']
        print(f"Applying focus to {len(piis_list)} piis")
        piis_list = piis_list + [" " + key_word for key_word in piis_list] #Added from find_focus_idx
    
        print("using filtered data" if (args.focus or args.focus_dataset) else "using raw data")
    
        if split in ["train", "validation"]:
            raw_data = datasets.load_from_disk(self.raw_data_path)
            document_types = sorted(set(raw_data['train']['document_type']))
            train_df, val_df = train_test_split(
                raw_data['train'].to_pandas(),
                test_size=0.05,
                random_state=42
            )

            data = datasets.Dataset.from_pandas(train_df if split == "train" else val_df)
            print("Num samples", len(data))

            # ----- NEW CODE ---------
            for sample in tqdm(data):
                text = format_instruction(sample, document_types)
                encoding = tokenizer(
                    text,
                    add_special_tokens=True,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                if args.focus:
                    try:
                        focus_idx_list = find_focus_idx_new(
                            text,
                            tokenizer,
                            None,
                            focus_type=args.focus_type,
                            focus_keywords=piis_list,
                            # prefix_size=25,
                        )
                    except Exception as e:
                        focus_idx_list = []
                        print("Focus idx not found", e)
                else:
                    focus_idx_list = []

                # pad/truncate to max_length
                focus_idx = torch.zeros(max_length, dtype=torch.long)
                if focus_idx_list:
                    valid = torch.tensor(focus_idx_list).clamp(max=max_length-1)
                    focus_idx[valid] = 1

                self.data.append({
                    "input_ids": encoding["input_ids"][0],
                    "attention_mask": encoding["attention_mask"][0],
                    "labels": encoding["input_ids"][0],
                    "focus_idx": focus_idx,
                })

                # ----- OLD CODE ---------
                # for sample in tqdm(data):
                #     text = format_instruction(sample, document_types)
                #     input = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
                #     if args.focus:
                #         try:
                #             focus_idx = find_focus_idx(text, tokenizer, None, focus_type=args.focus_type, focus_keywords=piis_list)
                #             # focus_idx = find_focus_idx(text, tokenizer, nlp, focus_type=args.focus_type)
                #             # focus_idx = list(range(51, len(tokenizer(text)['input_ids'])))
                #         except Exception as e:
                #             focus_idx = []
                #             print("Focus idx not found", e)
                #     else:
                #         focus_idx = []
                #     focus_idx = torch.Tensor(focus_idx + [0] * (max_length - len(focus_idx)))
                #     self.data.append({"input_ids": input['input_ids'][0], "attention_mask": input['attention_mask'][0], "labels": input['input_ids'][0], "focus_idx": focus_idx})
                print(self.data[0], len(self.data))
                print(self.data[0]["focus_idx"])
                # print(f"Saving preprocessed dataset to {preprocessed_path}")
                # dataset_to_save = datasets.Dataset.from_list(self.data)
                # dataset_to_save.save_to_disk(preprocessed_path)
                # self.data = dataset_to_save
                # self.data.set_format(
                #     type="torch",
                #     columns=["input_ids", "attention_mask", "labels", "focus_idx"]
                # )

        elif split == "test":
            raise NotImplementedError
               
        print(f"Train dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# Custom dataset for applying UnDial on sequences used for the True Prefix Attack
class TPAttackDataset(Dataset):
    def __init__(self, args, eval_window_size, tokenizer, split, max_length, dataset_name):
        self.data = []
        self.test_data_path = os.path.join(args.cache_dir, f"test_window_{eval_window_size}_num_{args.eval_num}_filtered_{args.focus}.pkl")
        self.train_data_path = os.path.join(args.cache_dir, f"train_num_{args.train_num}_focus_{args.focus}_type_{args.focus_type}.pkl")
        # self.raw_data_path = args.filtered_extract_challenge_data_path if (args.focus or args.focus_dataset) else args.extract_challenge_data_path
        self.max_length = max_length
        print("using filtered data" if (args.focus or args.focus_dataset) else "using raw data")
        
        print(f"Using {dataset_name}")
        self.raw_data_path = f'datasets/{dataset_name}'
        print(self.raw_data_path)
        if split == "train":
            raw_data = datasets.load_from_disk(self.raw_data_path)
            train_data = raw_data['train']
            if train_data:
                for text in tqdm(train_data['text']):
                    input_tensor = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
                    if args.focus:
                        try:
                            pii = tokenizer.decode(input_tensor['input_ids'][0,50:])
                            focus_idx = find_focus_idx(text, tokenizer, None, focus_type=args.focus_type, focus_keywords=[pii])
                            print(focus_idx)
                            # focus_idx = find_focus_idx(text, tokenizer, nlp, focus_type=args.focus_type)
                            # focus_idx = list(range(51, len(tokenizer(text)['input_ids'])))
                        except Exception as e:
                            focus_idx = []
                            print("Focus idx not found", e)
                    else:
                        focus_idx = []
                    focus_idx = torch.Tensor(focus_idx + [0] * (max_length - len(focus_idx)))
                    self.data.append({"input_ids": input_tensor['input_ids'][0], "attention_mask": input_tensor['attention_mask'][0], "focus_idx": focus_idx})
                print(self.data[0], len(self.data))

            print(f"Train dataset size: {len(self.data)}")

        elif split == "test":
            raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
def normalize_reply(text: str, version=2) -> str:
    """
    Standardize the capitalization and punctuation spacing of the input text.
    Version 1: Fix sentence start casing, and punctuation.
    Version 2: Add trailing period, if missing.
    """

    switch_list = [(' .', '.'), (' ,', ','), (' ?', '?'), (' !', '!'), (" ' ", "'")]

    # add spaces so that words and punctuation can be seaprated
    new_text = text.lower()

    # normalize in case of human:
    for new, old in switch_list:
        new_text = new_text.replace(old, new).replace('  ', ' ')

    # split on punctuation to find sentence boundaries
    # capitalize stuff
    tokens = new_text.split(' ')
    for i in range(len(tokens)):
        if i == 0:
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in ('i', "i'm", "i've", "i'll", "i'd"):
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in '?.!' and i < len(tokens) - 1:
            tokens[i + 1] = uppercase(tokens[i + 1])
    new_text = ' '.join(tokens)
    new_text = ' ' + new_text + ' '

    for tup in switch_list:
        new_text = new_text.replace(tup[0], tup[1])

    # get rid of surrounding whitespace
    new_text = new_text.strip()
    new_text = new_text.replace('  ', ' ')

    if version > 1 and new_text and new_text[-1] not in '!.?)"\'':
        new_text += '.'

    return new_text

DIALOG_DATASETS = [
    'wizard_of_wikipedia',
    'empathetic_dialogues',
    'blended_skill_talk',
    'wizard_of_internet'
]

class Custom_Dataset(Dataset):
    def __init__(
            self,
            tokenizer,
            dataset_name,
            valid_subset_path,
            type_path,
            input_length,
            output_length,
            args):
        self.args = args
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.output_length = output_length
        self.dataset_name = dataset_name
        self.type_path = type_path
        self.valid_subset_path = valid_subset_path

        if self.type_path == 'train':
            self.dataset = pd.read_csv(dataset_name, lineterminator='\n')
            batch_size = self.args.train_batch_size * \
                self.args.gradient_accumulation_steps * self.args.ngpu
            if len(self.dataset) != batch_size:
                raise Exception(
                    "Effective batch size should be the same as length of train set")

        else:
            if '.csv' in self.dataset_name:
                self.dataset = pd.read_csv(dataset_name, lineterminator='\n')
            elif '.json' in self.dataset_name:
                self.dataset = pd.read_json(dataset_name)
            else: # load from huggingface hub
                if valid_subset_path:
                    dataset = load_dataset(
                        self.dataset_name,
                        valid_subset_path,
                        split=type_path,
                        cache_dir=args.cache_dir,
                        verification_mode='no_checks'
                    )
                else:
                    dataset = load_dataset(
                        self.dataset_name,
                        split=type_path,
                        cache_dir=args.cache_dir,
                        verification_mode='no_checks'
                    )
                self.dataset = dataset.to_pandas()

        # About 4 examples have one more or one less class for some reason,
        # they will cause dataloader error
        if self.dataset_name == 'ai2_arc':
            self.dataset['length'] = self.dataset['choices'].apply(
                lambda x: len(x['text']))
            self.dataset = self.dataset[self.dataset['length'] == 4]

        self.dataset = self.dataset.dropna()

    def __len__(self):
        return len(self.dataset)

    def input_to_target(self, input):
        input_s = input.split(' ')
        input_ = " ".join(input_s[:len(input_s) - 1])
        target = " " + input_s[len(input_s) - 1]
        return input_, target

    def create_dialogue_prompt(self, turns):
        # prompt = 'A converstaion between two Users:\n'
        prompt = ''
        for i, turn in enumerate(turns):
            turn = normalize_reply(turn)
    
            if i % 2 == 0:
                prompt += f'User 1: {turn}\n'
            else:
                prompt += f'User 2: {turn}\n'

        if i % 2:
            prompt += f'User 1:'
        else:
            prompt += f'User 2:'
        return prompt

    def convert_to_features(self, example_batch):
        try:
            doc_id = torch.tensor(example_batch['doc_id'], dtype=torch.int)
        except KeyError:
            doc_id = ''

        choices = []
        answer_index = 0
        task, task_type = '', ''
        if self.type_path == 'train':
            input_ = example_batch['text']
            target_ = example_batch['text']
        else:
            if 'lambada' in self.dataset_name:
                input_, target_ = self.input_to_target(example_batch['text'])
                task_type = 'completion'
                task = 'lambada'
            elif self.dataset_name == 'piqa':
                input_ = example_batch['goal']
                choices = [
                    ' ' + example_batch['sol1'],
                    ' ' + example_batch['sol2']]
                target_ = choices[int(example_batch['label'])]
                answer_index = int(example_batch['label'])
                task_type = 'classification'
            elif self.dataset_name == 'hellaswag':
                input_ = example_batch['ctx']
                choices = []
                choices = [' ' + c for c in example_batch['endings']]
                target_ = choices[int(example_batch['label'])]
                answer_index = int(example_batch['label'])
                task_type = 'classification'
            elif self.dataset_name == 'ai2_arc':
                input_ = example_batch['question']
                choices = [' ' + c for c in example_batch['choices']['text']]
                answer_index = example_batch['choices']['label'].tolist().index(
                    example_batch['answerKey'])
                target_ = choices[answer_index]
                task_type = 'classification'
            elif self.dataset_name == 'winogrande':
                input_, rest = example_batch['sentence'].split(' _')
                choices = [
                    ' ' + example_batch['option1'] + rest,
                    ' ' + example_batch['option2'] + rest]
                answer_index = int(
                    example_batch['answer']) - 1  # Label are '1' or '2'
                target_ = choices[answer_index]
                task_type = 'classification'
            elif self.dataset_name == 'math_qa':
                input_ = example_batch['Problem']
                choices = [c[4:].rstrip(" ,") for c in re.findall(
                    r"[abcd] \) .*?, |e \) .*?$", example_batch["options"])]
                answer_index = [
                    'a', 'b', 'c', 'd', 'e'].index(
                    example_batch['correct'])
                target_ = choices[answer_index]
                task_type = 'classification'
            elif 'pubmed_qa' in self.dataset_name:
                input_ = f"Context: {example_batch['abstract']}\nQuestion: {example_batch['question']}\nAnswer:"
                choices = [' yes', ' maybe', ' no']
                answer_index = ['yes', 'maybe', 'no'].index(
                    example_batch['final_decision'])
                target_ = choices[answer_index]
                task = 'pubmed_qa'
                task_type = 'classification'
            elif self.dataset_name == 'super_glue' and self.valid_subset_path == 'copa':
                input_ = example_batch['premise']
                choices = [
                    ' ' + example_batch['choice1'],
                    ' ' + example_batch['choice2']]
                answer_index = int(example_batch['label'])
                target_ = choices[answer_index]
                task_type = 'classification'
            elif any(d in self.dataset_name for d in DIALOG_DATASETS):
                input_ = self.create_dialogue_prompt(example_batch['text'][:-1])
                target_ = normalize_reply(example_batch['text'][-1])
                task = self.dataset_name.split('.')[0].split('/')[1]
                task_type = 'dialog'
            elif 'pile' in self.dataset_name:
                input_, target_ = example_batch['text'], example_batch['text']
                task = 'pile'
                task_type = 'ppl'
            elif 'wikitext' in self.dataset_name:
                input_, target_ = example_batch['text'], example_batch['text']
                task = 'wikitext'
                task_type = 'ppl'
            else:
                input_, target_ = example_batch['text'], example_batch['text']
                task = 'target'
                task_type = 'target'

        if not task:
            if self.valid_subset_path:
                task = f'{self.dataset_name}_{self.valid_subset_path}'
            else:
                task = f'{self.dataset_name}'

        source = self.tokenizer(
            input_,
            max_length=self.input_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt")

        targets = self.tokenizer(
            target_,
            max_length=self.output_length,
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            return_tensors="pt")
        # targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length,
        # padding='max_length', truncation=True, return_tensors="pt")
        return source, targets, doc_id, task, task_type, choices, answer_index

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        try:
            source, targets, doc_id, task, task_type, choices, answer_index = self.convert_to_features(
                data)
        except:
            print(data)

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids,
                "source_mask": src_mask,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "doc_id": doc_id,
                "task": task,
                "task_type": task_type,
                "choices": choices,
                "answer_index": answer_index}