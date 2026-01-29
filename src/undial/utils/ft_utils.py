import copy
import transformers
import torch
import math
from torch.nn import CrossEntropyLoss, KLDivLoss
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, TrainerCallback
from peft import prepare_model_for_kbit_training, LoraConfig, PrefixTuningConfig, TaskType, PeftType, get_peft_model, get_peft_config, PeftModel, PeftConfig
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker

def merge_model(model_1, model_2, weight_subtraction_coef, operation = 'subtraction'):
    tmp = copy.deepcopy(model_1)
    for param1, param2 in zip(model_1.parameters(), model_2.parameters()):
        if operation == 'subtraction':
            param1.data -= weight_subtraction_coef * param2.data
        else:
            raise NotImplementedError("operation not implemented")
    return model_1

def get_train_args(args):
    train_args = transformers.TrainingArguments(
            per_device_train_batch_size = args.train_batch_size, 
            num_train_epochs=args.num_epochs,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=1,
            save_strategy="no",
            output_dir='outputs',
        )
    if "DI" in args.method:
        train_args.remove_unused_columns = False
    if args.gradient_accu > 1:
        train_args.gradient_accumulation_steps = args.gradient_accu
    if args.warmup_steps > 0:
        train_args.warmup_steps = args.warmup_steps
    if args.weight_decay > 0:
        train_args.weight_decay = args.weight_decay
    # if args.lr_sc:
    #     train_args.learning_rate = args.lr_sc
    return train_args

def get_train_args_di(args):    
    train_args = transformers.TrainingArguments(
            per_device_train_batch_size = args.train_batch_size // 2, 
            num_train_epochs=args.num_epochs_di,
            learning_rate=args.lr_di,
            fp16=True,
            save_strategy="no",
            output_dir='outputs',
            evaluation_strategy="steps",      # Evaluation is done at the end of each epoch
            eval_steps=100,     
            logging_dir="./logs",             # Directory for storing logs
            logging_steps=1,  
        )
    if "DI" in args.method:
        train_args.remove_unused_columns = False
    if args.gradient_accu > 1:
        train_args.gradient_accumulation_steps = args.gradient_accu
    if args.warmup_steps > 0:
        train_args.warmup_steps = args.warmup_steps
    if args.weight_decay > 0:
        train_args.weight_decay = args.weight_decay
    return train_args

def get_train_args_di_custom(args, report=None):    
    train_args = transformers.TrainingArguments(
            per_device_train_batch_size = args.train_batch_size // 2, 
            num_train_epochs=args.num_epochs_di,
            learning_rate=args.lr_di,
            fp16=True,
            save_strategy="no",
            output_dir='outputs',
            # evaluation_strategy="steps",      # Evaluation is done at the end of each epoch
            # eval_steps=100,     
            logging_dir="./logs",             # Directory for storing logs
            logging_steps=10,
            report_to=report
        )
    if "DI" in args.method:
        train_args.remove_unused_columns = True
    if args.gradient_accu > 1:
        train_args.gradient_accumulation_steps = args.gradient_accu
    if args.warmup_steps > 0:
        train_args.warmup_steps = args.warmup_steps
    else:
        train_args.warmup_ratio = 0.03
    if args.weight_decay > 0:
        train_args.weight_decay = args.weight_decay
    if "RT" in args.method or "AT" in args.method:
        train_args.max_grad_norm = 1.0
        train_args.lr_scheduler_type="cosine"
        train_args.optim="paged_adamw_32bit"
    return train_args

def get_lora_model(model, rank=8, lora_alpha=16, path=None):
    if 'gpt_neo' in str(type(model)):
        target_modules = ["q_proj", "v_proj"]
    elif 'llama' in str(type(model)):
        target_modules = "all-linear"
    else:
        raise NotImplementedError("specify lora layer in ft_utils.py")
    
    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = prepare_model_for_kbit_training(model)
    if path:
        print("Loading previous adapters")
        model = PeftModel.from_pretrained(
            model,
            path
        )
    else:
        model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def load_peft_model(args, model, method):
    if method == 'lora':
        return get_lora_model(model, rank=args.rank, lora_alpha=args.lora_alpha, path=args.model_name_or_path)
    elif method == 'adapter':
        raise NotImplementedError

def is_subtoken(sub, main):
  sub_len = len(sub)
  for i in range(len(main) - sub_len + 1):
    if main[i:i+sub_len] == sub:
      return i
  return False

class GradientAscentTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        return -1 * super(GradientAscentTrainer,self).compute_loss(model, inputs, return_outputs=False)
    
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, initial_perplexity, ppl_change):
        self.best_perplexity = initial_perplexity
        self.ppl_change = ppl_change
        self.early_stop_epoch = None
        print("Initial Perplexity: {}".format(self.best_perplexity))

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        current_perplexity = math.exp(kwargs['metrics']['eval_loss'])
        if current_perplexity > self.ppl_change * self.best_perplexity and state.epoch >= 1:
            print("Perplexity increased, stopping training at epoch {}".format(state.epoch))
            control.should_training_stop = True
            self.early_stop_epoch = state.epoch  # Record the epoch number
        print("Current Perplexity: {}, Best Perplexity: {} at epoch {}".format(current_perplexity, self.best_perplexity, state.epoch))

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        if "focus_idx" in batch[0]:
            focus_idx = torch.stack([item["focus_idx"] for item in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "focus_idx": focus_idx}    
        else:
            return {"input_ids": input_ids, "attention_mask": attention_mask}

class ATDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        if "focus_idx" in batch[0]:
            focus_idx = torch.stack([item["focus_idx"] for item in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, 'focus_idx': focus_idx}
        else:
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class DITrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        di_kwargs = kwargs.pop('di_kwargs')
        super().__init__(*args, **kwargs)
        self.di_kwargs = di_kwargs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unlearn_teacher_model = AutoModelForCausalLM.from_pretrained(di_kwargs['teacher_model']).to(self.device)

    def compute_loss(self, model, inputs, num_items_in_batch=-1, return_outputs=False):
        outputs = model(input_ids = inputs['input_ids'].to(self.device), attention_mask = inputs['attention_mask'].to(self.device))
        logits = outputs.logits
        input_ids = inputs['input_ids'].to(self.device)
        shift_labels = input_ids[..., 1:].contiguous().to(self.device)
        shift_logits = logits[..., :-1, :].contiguous().to(self.device)

        with torch.no_grad():
            teacher_logits = self.unlearn_teacher_model(input_ids = input_ids, attention_mask = inputs['attention_mask']).logits
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        
        mask = torch.zeros_like(shift_logits).to(self.device)
        mask[torch.arange(mask.shape[0]).view(-1, 1, 1), torch.arange(mask.shape[1]).view(1, -1, 1), shift_labels.unsqueeze(-1)] = 1
        pre_softmax = shift_teacher_logits - mask * self.di_kwargs['di_strength']
        soft_label = F.softmax(pre_softmax, dim=-1)
        
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), soft_label.view(-1, soft_label.size(-1)))
        
        if self.di_kwargs['focus']:
            # print("Focused Unlikelihood Training")
            # focus_idx = inputs['focus_idx']
            shift_focus_idx = inputs['focus_idx'] - 1   
            shift_focus_idx[shift_focus_idx < 0] = 0  
            shift_focus_idx[shift_focus_idx >= 199] = 0 
            shift_focus_idx = shift_focus_idx[:,: -1]
            
            reweight_vector = torch.zeros(shift_focus_idx.size(0), shift_focus_idx.size(1))            
            rows, cols = torch.meshgrid(torch.arange(shift_focus_idx.size(0)), torch.arange(shift_focus_idx.size(1)))
            rows = rows.to(self.device)
            row_indices = rows[shift_focus_idx!=0].flatten()
            col_indices = shift_focus_idx.flatten()
            col_indices = col_indices[col_indices!=0]
            
            ## check col_indices out of bound
            if not (col_indices < shift_teacher_logits.size(1)).all():
                print("col_indices out of bound")
                import pdb; pdb.set_trace()
            
            if self.di_kwargs['focus_hard']:
                reweight_vector[row_indices.long(), col_indices.long()] = 1
                reweight_vector = reweight_vector.flatten().to(self.device)
            else:
                reweight_vector[row_indices.long(), col_indices.long()] = self.di_kwargs['focus_coeff'] - 1
                reweight_vector = reweight_vector.flatten().to(self.device) + 1
            loss = loss * reweight_vector
        return loss.mean()
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Setting the model to evaluation mode
        self.model.eval()

        total_loss = 0
        total_examples = 0

        for batch in self.get_eval_dataloader(eval_dataset):
            # Move batch to device
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            batch['labels'] = batch['input_ids']

            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_examples += batch['input_ids'].size(0)

        # Calculate average loss
        avg_loss = total_loss / total_examples

        # Calculate perplexity
        perplexity = math.exp(avg_loss)
        print(f"Perplexity: {perplexity}")
        logs = {"eval_loss": avg_loss, "perplexity": perplexity}
        if self.callback_handler is not None:
            self.callback_handler.on_evaluate(self.args, self.state, self.control, logs)

        return {"eval_loss": avg_loss, "perplexity": perplexity}

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

# Code for the linear combination of UnDial and standard loss
# Unsuccessful, but kept for reference, perhaps could work with FUnDial.
class RegularizedTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        di_kwargs = kwargs.pop('di_kwargs')
        super().__init__(*args, **kwargs)
        self.di_kwargs = di_kwargs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unlearn_teacher_model = AutoModelForCausalLM.from_pretrained(di_kwargs['teacher_model']).to(self.device)

    def compute_loss(self, model, inputs, num_items_in_batch=-1, return_outputs=False):
        weight_factor = 0.5
        outputs = model(**inputs)
        logits = outputs.logits
        input_ids = inputs['input_ids'].to(self.device)
        shift_labels = input_ids[..., 1:].contiguous().to(self.device)
        shift_logits = logits[..., :-1, :].contiguous().to(self.device)
        # HF Loss computation
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                hf_loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                hf_loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                hf_loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            hf_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            hf_loss *= self.accelerator.num_processes


        #UnDial loss computation
        with torch.no_grad():
            teacher_logits = self.unlearn_teacher_model(input_ids = input_ids, attention_mask = inputs['attention_mask']).logits
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        
        mask = torch.zeros_like(shift_logits).to(self.device)
        mask[torch.arange(mask.shape[0]).view(-1, 1, 1), torch.arange(mask.shape[1]).view(1, -1, 1), shift_labels.unsqueeze(-1)] = 1
        pre_softmax = shift_teacher_logits - mask * self.di_kwargs['di_strength']
        soft_label = F.softmax(pre_softmax, dim=-1)
        
        loss_fct = CrossEntropyLoss(reduction='none')
        undial_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), soft_label.view(-1, soft_label.size(-1)))
        
        if self.di_kwargs['focus']:
            print("Focused Unlikelihood Training")
            # focus_idx = inputs['focus_idx']
            shift_focus_idx = inputs['focus_idx'] - 1   
            shift_focus_idx[shift_focus_idx < 0] = 0  
            shift_focus_idx[shift_focus_idx >= 1199] = 0 
            shift_focus_idx = shift_focus_idx[:,: -1]
            
            reweight_vector = torch.zeros(shift_focus_idx.size(0), shift_focus_idx.size(1))            
            rows, cols = torch.meshgrid(torch.arange(shift_focus_idx.size(0)), torch.arange(shift_focus_idx.size(1)))
            rows = rows.to(self.device)
            row_indices = rows[shift_focus_idx!=0].flatten()
            col_indices = shift_focus_idx.flatten()
            col_indices = col_indices[col_indices!=0]
            
            ## check col_indices out of bound
            if not (col_indices < shift_teacher_logits.size(1)).all():
                print("col_indices out of bound")
                import pdb; pdb.set_trace()
            
            if self.di_kwargs['focus_hard']:
                reweight_vector[row_indices.long(), col_indices.long()] = 1
                reweight_vector = reweight_vector.flatten().to(self.device)
            else:
                reweight_vector[row_indices.long(), col_indices.long()] = self.di_kwargs['focus_coeff'] - 1
                reweight_vector = reweight_vector.flatten().to(self.device) + 1
            undial_loss = undial_loss * reweight_vector
         
        return (weight_factor * undial_loss.mean()) + ( (1-weight_factor) * hf_loss)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Setting the model to evaluation mode
        self.model.eval()

        total_loss = 0
        total_examples = 0

        for batch in self.get_eval_dataloader(eval_dataset):
            # Move batch to device
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            batch['labels'] = batch['input_ids']

            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_examples += batch['input_ids'].size(0)

        # Calculate average loss
        avg_loss = total_loss / total_examples

        # Calculate perplexity
        perplexity = math.exp(avg_loss)
        print(f"Perplexity: {perplexity}")
        logs = {"eval_loss": avg_loss, "perplexity": perplexity}
        if self.callback_handler is not None:
            self.callback_handler.on_evaluate(self.args, self.state, self.control, logs)

        return {"eval_loss": avg_loss, "perplexity": perplexity}

class AlternatingUnlearningTrainer(transformers.Trainer):
    
    def __init__(self, *args, **kwargs):
        di_kwargs = kwargs.pop('di_kwargs')
        super().__init__(*args, **kwargs)
        self.di_kwargs = di_kwargs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unlearn_teacher_model = AutoModelForCausalLM.from_pretrained(di_kwargs['teacher_model']).to(self.device)

        self.interval_steps = 8  # Number of steps to use standard loss
        self.current_step = 0  # Track total steps
        self.use_undial_loss = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Alternates between standard HF loss and UNDIAL loss based on step counter.
        """
        # Increment step counter
        if not model.training: 
            loss, outputs = self._compute_standard_loss(model, inputs, return_outputs, num_items_in_batch)
            if return_outputs:
                return (loss, outputs)
            return loss


        # self.current_step += 1
        # # Determine which loss to use based on the step counter
        # if self.current_step % self.interval_steps == 0:
        #     print(f"Applying Undial loss type at step {self.current_step}")
        #     loss = self._compute_undial_loss(model, inputs)
        #     log_prefix = "[UNDIAL]"

        # else:
        #     # Use standard HF loss for the first `interval_steps` steps
        #     loss = self._compute_standard_loss(model, inputs, return_outputs)
        #     log_prefix = "[STANDARD]"


            
        self.current_step += 1
        # Determine which loss to use based on the step counter
        if self.current_step % self.interval_steps == 0:
            print(f"Switching loss type at step {self.current_step}")
            self.use_undial_loss = not self.use_undial_loss

        if self.use_undial_loss:
            # Use UNDIAL loss for the remaining step in the cycle
            loss = self._compute_undial_loss(model, inputs)
            log_prefix = "[UNDIAL]"
        else:
            # Use standard HF loss for the first `interval_steps` steps
            loss = self._compute_standard_loss(model, inputs, return_outputs)
            log_prefix = "[STANDARD]"
            
        # Log which loss we're using
        if self.current_step % 100 == 0:
            print(f"{log_prefix} Step: {self.current_step}, Loss: {loss.item():.4f}")
            # print("Updating teacher model")
            # self.unlearn_teacher_model = model
            
        # if return_outputs:
        #     return (loss, outputs)
        return loss

    def _compute_standard_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def _compute_undial_loss(self, model, inputs, num_items_in_batch=-1, return_outputs=False):
        outputs = model(input_ids = inputs['input_ids'].to(self.device), attention_mask = inputs['attention_mask'].to(self.device))
        logits = outputs.logits
        input_ids = inputs['input_ids'].to(self.device)
        shift_labels = input_ids[..., 1:].contiguous().to(self.device)
        shift_logits = logits[..., :-1, :].contiguous().to(self.device)       

        with torch.no_grad():
            teacher_logits = self.unlearn_teacher_model(input_ids = input_ids, attention_mask = inputs['attention_mask']).logits
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        
        mask = torch.zeros_like(shift_logits).to(self.device)
        mask[torch.arange(mask.shape[0]).view(-1, 1, 1), torch.arange(mask.shape[1]).view(1, -1, 1), shift_labels.unsqueeze(-1)] = 1
        pre_softmax = shift_teacher_logits - mask * self.di_kwargs['di_strength']
        soft_label = F.softmax(pre_softmax, dim=-1)
        
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), soft_label.view(-1, soft_label.size(-1)))
        
        if self.di_kwargs['focus']:
            # focus_idx = inputs['focus_idx']
            shift_focus_idx = inputs['focus_idx'] - 1   
            shift_focus_idx[shift_focus_idx < 0] = 0  
            shift_focus_idx[shift_focus_idx >= 1535] = 0 
            shift_focus_idx = shift_focus_idx[:,: -1]
            
            reweight_vector = torch.zeros(shift_focus_idx.size(0), shift_focus_idx.size(1))            
            rows, cols = torch.meshgrid(torch.arange(shift_focus_idx.size(0)), torch.arange(shift_focus_idx.size(1)))
            rows = rows.to(self.device)
            row_indices = rows[shift_focus_idx!=0].flatten()
            col_indices = shift_focus_idx.flatten()
            col_indices = col_indices[col_indices!=0]
            
            ## check col_indices out of bound
            if not (col_indices < shift_teacher_logits.size(1)).all():
                print("col_indices out of bound")
                import pdb; pdb.set_trace()
            
            if self.di_kwargs['focus_hard']:
                reweight_vector[row_indices.long(), col_indices.long()] = 1
                reweight_vector = reweight_vector.flatten().to(self.device)
            else:
                reweight_vector[row_indices.long(), col_indices.long()] = self.di_kwargs['focus_coeff'] - 1
                reweight_vector = reweight_vector.flatten().to(self.device) + 1
            loss = loss * reweight_vector
        return loss.mean()
    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))