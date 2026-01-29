import argparse, torch, sys, os, time, math, wandb, json
from tqdm import tqdm
from utils.data_utils import TPAttackDataset, GretelDataset, DischSummDataset
from utils.other_utils import torch_save
from utils.ft_utils import (get_train_args, 
    GradientAscentTrainer,
    DITrainer,
    AlternatingUnlearningTrainer,
    RegularizedTrainer,
    MyDataCollator,
    ATDataCollator,
    load_peft_model,
    get_train_args_di,
    get_train_args_di_custom,
    EarlyStoppingCallback
)
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, set_seed
from dotenv import load_dotenv
# os.environ["WANDB_DISABLED"] = "false"


def get_args_parser():
    parser = argparse.ArgumentParser('llm_unlearning', add_help=False)
    parser.add_argument('-v', '--verbose', action='store_true')
    
    ## Path
    parser.add_argument('--output_folder', default='outputs', type=str)
    parser.add_argument('--model_folder', default='./model', type=str)
    parser.add_argument("--extract_challenge_data_path", default = "./data/extract_challenge/train_dataset.npy", type=str)
    parser.add_argument("--filtered_extract_challenge_data_path", default = "./data/extract_challenge/filtered.npy", type=str)
    parser.add_argument('-m', "--model_name_or_path", default = "meta-llama/LLama-3.2-1B", type=str) #EleutherAI/gpt-neox-20b, EleutherAI/gpt-j-6b, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B, EleutherAI/gpt-neo-125m
    parser.add_argument('--teacher_model', default = "meta-llama/LLama-3.2-1B", type=str) 
    parser.add_argument("--cache_dir", default='./data', type=str)

    ## Methods and their params
    parser.add_argument('--method', default='raw_gpt', choices=['GPT', 'UL', 'TA', 'CD', 'DI', 'RT', 'AT'], type=str) ## UL for unlikelihood training, TA for task arithemic, CD for Contrastive Decoding, DI (ours) for deliberate imagination
    ## CD
    parser.add_argument('--contrastive_coef', type=float, default=0.1)
    parser.add_argument('--strat', choices=['relu', 'relu2', 'relu3', 'relu_offset'], default='relu2') ## Varients of CD
    parser.add_argument('--relu_threshold', type=float, default=0)
    ## TA
    parser.add_argument("--weight_subtraction_coef", default=0.25, type=float)
    ## DP
    parser.add_argument("--DP", default=False, type=bool)
    parser.add_argument("--DP_coef", default=0, type=float)
    ## DI (ours) params
    parser.add_argument("--num_epochs_di", default=50, type=int)
    parser.add_argument("--lr_di", default=6e-4, type=float)
    parser.add_argument("--di_strength", default=5, type=float)
    ## Focus params
    parser.add_argument("--focus", default=False, type=bool)
    parser.add_argument("--focus_dataset", default=False, type=bool)
    parser.add_argument("--focus_coeff", default=10, type=float)
    parser.add_argument("--focus_type", default='entity', type=str)
    parser.add_argument("--focus_hard", default=False, type=bool)

    ## Train params
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--peft", choices = ['ft', 'lora', 'adapter'], default='ft')
    parser.add_argument("--rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--train_num", default=15000, type=int)
    parser.add_argument("--gradient_accu", default=1, type=int)
    parser.add_argument("--lr_sc", default="", choices=['cosine', 'polynomial'], type=str)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--early_stop", default=False, type=bool)
    parser.add_argument("--early_stop_criteria", default=1.03, type=float) ## A number > 1 that indicates the percentage of ppl change
    
    ## Eval params
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--eval_window_size", default=40, type=int) # The window calculating the extraction likehood during evaluation
    parser.add_argument("--eval_num", default=500, type=int)
    parser.add_argument("--oracle_model", default='EleutherAI/gpt-j-6b', type=str)
    
    ## Generation params
    parser.add_argument("--do_sample", default=True, type=bool)
    parser.add_argument("--top_p", default=0.95, type=int)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--relu3_topk", default=500, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument("--max_length", default=60, type=int)
    parser.add_argument("--cd_num_token", default=1000, type=int)

    # Custom dataset name
    parser.add_argument("--dataset_name", default="undial-DS-3knames", type=str)
    
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    set_seed(42)
    
    ## Set paths
    log_path, log_sample_path = f'./{args.output_folder}/log.txt', f'./{args.output_folder}/log_sample.txt'
    print("Log path: ", log_path, log_sample_path)
    model_save_folder= f"{args.method}_{args.peft}_{args.dataset_name}_{args.model_name_or_path.replace('/', '-')}_{args.num_epochs}_{args.lr}_{args.train_batch_size}"
    if args.method == 'DI':
        model_save_folder += f"_{args.num_epochs_di}_{args.lr_di}_{args.di_strength}_{args.weight_decay}_{args.warmup_steps}_{args.gradient_accu}_{args.early_stop}_{args.early_stop_criteria}"
        # Set the FT'd model automatically (We don't want to apply Unlearning in the base LLaMa)
        args.model_name_or_path = 'llama-3.2-1b-document-classifier'
        args.teacher_model = 'llama-3.2-1b-document-classifier'
        
        if args.max_length > 200:
            print("CHANGE MAX_LENGTH")
            raise Exception
    if args.focus:
        model_save_folder += f"_focus_{args.focus_coeff}_type_{args.focus_type}_hard_{args.focus_hard}"
    model_save_path = f"{args.model_folder}/{model_save_folder}"
    if not os.path.exists(args.output_folder): 
        os.mkdir(args.output_folder)
    if not os.path.exists(args.model_folder): 
        os.mkdir(args.model_folder)
        

    # RUN_NAME = f'Undial_{args.model_name_or_path.replace("/","-")}_{args.dataset_name}' if not args.focus else f'Undial-{args.dataset_name}-focus'
    RUN_NAME = 'Test_Run'

    print(f"Loading model {args.model_name_or_path}, method {args.method}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    peft_flag = True if (args.peft == 'lora' or args.peft == 'adapter') else False


    if args.method in ['AT', 'RT']:
        if args.max_length < 200:
            print("CHANGE MAX_LENGTH")
            raise Exception

        print("Loading data...")
        # train_dataset = GretelDataset(args, eval_window_size = args.eval_window_size, tokenizer = tokenizer, split='train', max_length=args.max_length)
        # validation_set = GretelDataset(args, eval_window_size = args.eval_window_size, tokenizer = tokenizer, split='validation', max_length=args.max_length)
        train_dataset = DischSummDataset(args, eval_window_size = args.eval_window_size, tokenizer = tokenizer, split='train', max_length=args.max_length)
        validation_set = DischSummDataset(args, eval_window_size = args.eval_window_size, tokenizer = tokenizer, split='validation', max_length=args.max_length)
        print("Data loaded.")
        
    print("Unlearning")
    
    # If the model already exists, we skip training
    if not os.path.exists(model_save_path) and args.method != 'raw_gpt': 
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
        # if args.peft == 'lora':
        #     model = load_peft_model(args, model, args.peft)
        
        if args.method == 'UL':
            trainer = GradientAscentTrainer(model=model, args=train_args, train_dataset=train_dataset, tokenizer=tokenizer, data_collator=data_collator)
            trainer.train()
            torch_save(model, model_save_path)
        
        elif args.method == 'TA':
            trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset, tokenizer=tokenizer, data_collator=data_collator)
            trainer.train()
            torch_save(model, model_save_path)
        
        elif args.method == 'CD':
            trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset, tokenizer=tokenizer, data_collator=data_collator)
            trainer.train()
            torch_save(model, model_save_path)
        
        elif args.method == "DI":
            print("Training DI...")
            train_dataset = TPAttackDataset(args, eval_window_size = args.eval_window_size, tokenizer = tokenizer, split='train',
                max_length=args.max_length, dataset_name=args.dataset_name)
            DI_data_collator = MyDataCollator(tokenizer)
            di_kwargs = {'di_strength': args.di_strength, "focus": args.focus, "focus_coeff": args.focus_coeff, 'focus_hard': args.focus_hard, 'teacher_model': args.teacher_model}
            di_train_args = get_train_args_di_custom(args)
            if args.early_stop:
                trainer = DITrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, eval_dataset=validation_set, tokenizer=tokenizer, data_collator=DI_data_collator)
                eval_results = trainer.evaluate(validation_set)
                initial_loss = eval_results.get("eval_loss")
                initial_perplexity = math.exp(initial_loss)
                
                trainer = DITrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, eval_dataset=validation_set, callbacks=[EarlyStoppingCallback(initial_perplexity = initial_perplexity, ppl_change=args.early_stop_criteria)], tokenizer=tokenizer, data_collator=DI_data_collator)
                trainer.train()
                for callback in trainer.callback_handler.callbacks:
                    if isinstance(callback, EarlyStoppingCallback):
                        early_stopping_callback = callback
                        break
                # early_stopping_callback = trainer.callback_handler.callbacks[0]
                early_stop_epoch = early_stopping_callback.early_stop_epoch
                if early_stop_epoch is None:
                    early_stop_epoch = args.num_epochs_di
                args.early_stop_epoch = early_stop_epoch
            else:
                print("Batch size:", di_train_args.train_batch_size)
                torch.cuda.empty_cache()
                for name, param in model.named_parameters():
                    if 'lora' in name.lower():
                        param.requires_grad = True
                def print_trainable_parameters(model):
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"Trainable parameters: {trainable_params} / {total_params} "
                        f"({100 * trainable_params / total_params:.2f}%)")
                print_trainable_parameters(model)
                print(train_dataset)
                trainer = DITrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, tokenizer=tokenizer, data_collator=DI_data_collator)
                torch.cuda.empty_cache()
                trainer.train()
            # torch_save(model, model_save_path, peft=peft_flag)
            trainer.save_model(RUN_NAME)
            training_args_dict = trainer.args.to_dict()  # Convert arguments to a dictionary
            training_args_dict["di_kwargs"] = di_kwargs
            with open(os.path.join(RUN_NAME, 'training_args.json'), 'w') as f:
                json.dump(training_args_dict, f, indent=2)
       
        elif args.method == "RT":
            print("Using Custom Regularized FT with UnDial...")
            WANDB_KEY = os.getenv("WANDB_KEY")
            wandb.login(key=WANDB_KEY)
            data_collator = ATDataCollator(tokenizer)
            di_kwargs = {'di_strength': args.di_strength, "focus": args.focus, "focus_coeff": args.focus_coeff, 'focus_hard': args.focus_hard, 'teacher_model': args.teacher_model}
            print(di_kwargs)
            di_train_args = get_train_args_di_custom(args, report="wandb")
            wandb.init(
                project="Unlearning-runs",
                name=RUN_NAME,
                config=di_train_args
            )
            if args.early_stop:
                trainer = RegularizedTrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, eval_dataset=validation_set, tokenizer=tokenizer, data_collator=data_collator)
                eval_results = trainer.evaluate(validation_set)
                initial_loss = eval_results.get("eval_loss")
                initial_perplexity = math.exp(initial_loss)
                
                trainer = RegularizedTrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, eval_dataset=validation_set, callbacks=[EarlyStoppingCallback(initial_perplexity = initial_perplexity, ppl_change=args.early_stop_criteria)], tokenizer=tokenizer, data_collator=data_collator)
                trainer.train()
                for callback in trainer.callback_handler.callbacks:
                    if isinstance(callback, EarlyStoppingCallback):
                        early_stopping_callback = callback
                        break
                # early_stopping_callback = trainer.callback_handler.callbacks[0]
                early_stop_epoch = early_stopping_callback.early_stop_epoch
                if early_stop_epoch is None:
                    early_stop_epoch = args.num_epochs_di
                args.early_stop_epoch = early_stop_epoch
            else:
                # trainer = DITrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, eval_dataset=validation_set, tokenizer=tokenizer, data_collator=data_collator)
                print("Not using early stopping")
                torch.cuda.empty_cache()
                trainer = RegularizedTrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, tokenizer=tokenizer, data_collator=data_collator)
                trainer.train()
            torch_save(model, model_save_path, peft=peft_flag)
            if RUN_NAME is not None:
                trainer.save_model(RUN_NAME)
                training_args_dict = trainer.args.to_dict()  # Convert arguments to a dictionary
                training_args_dict["di_kwargs"] = di_kwargs
                with open(os.path.join(RUN_NAME, 'training_args.json'), 'w') as f:
                    json.dump(training_args_dict, f, indent=2)
            else:
                trainer.save_model("temp_save")

        elif args.method == "AT":
            print("Using Custom Alternative Training with UnDial...")
            WANDB_KEY = os.getenv("WANDB_KEY")
            wandb.login(key=WANDB_KEY)
            data_collator = ATDataCollator(tokenizer)
            di_kwargs = {'di_strength': args.di_strength, "focus": args.focus, "focus_coeff": args.focus_coeff, 'focus_hard': args.focus_hard, 'teacher_model': args.teacher_model}
            print(di_kwargs)
            di_train_args = get_train_args_di_custom(args, report="wandb")
            wandb.init(
                project="Unlearning-runs",
                name=RUN_NAME,
                config=di_train_args
            )
            if args.early_stop:
                trainer = AlternatingUnlearningTrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, eval_dataset=validation_set, tokenizer=tokenizer, data_collator=data_collator)
                eval_results = trainer.evaluate(validation_set)
                initial_loss = eval_results.get("eval_loss")
                initial_perplexity = math.exp(initial_loss)
                
                trainer = AlternatingUnlearningTrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, eval_dataset=validation_set, callbacks=[EarlyStoppingCallback(initial_perplexity = initial_perplexity, ppl_change=args.early_stop_criteria)], tokenizer=tokenizer, data_collator=data_collator)
                trainer.train()
                for callback in trainer.callback_handler.callbacks:
                    if isinstance(callback, EarlyStoppingCallback):
                        early_stopping_callback = callback
                        break
                # early_stopping_callback = trainer.callback_handler.callbacks[0]
                early_stop_epoch = early_stopping_callback.early_stop_epoch
                if early_stop_epoch is None:
                    early_stop_epoch = args.num_epochs_di
                args.early_stop_epoch = early_stop_epoch
            else:
                # trainer = DITrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, eval_dataset=validation_set, tokenizer=tokenizer, data_collator=data_collator)
                print("Not using early stopping")
                print(di_train_args.train_batch_size)
                torch.cuda.empty_cache()
                trainer = AlternatingUnlearningTrainer(model=model, di_kwargs = di_kwargs, args=di_train_args, train_dataset=train_dataset, tokenizer=tokenizer, data_collator=data_collator)
                trainer.train()
            # torch_save(model, model_save_path, peft=peft_flag)
            if RUN_NAME is not None:
                trainer.save_model(RUN_NAME)
                training_args_dict = trainer.args.to_dict()  # Convert arguments to a dictionary
                training_args_dict["di_kwargs"] = di_kwargs
                with open(os.path.join(RUN_NAME, 'training_args.json'), 'w') as f:
                    json.dump(training_args_dict, f, indent=2)
            else:
                trainer.save_model("temp_save")
        del model
        torch.cuda.empty_cache()
    print("Unlearning Done")