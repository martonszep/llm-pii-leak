import outlines, os
from huggingface_hub import login
import pandas as pd
from dotenv import load_dotenv
from utils.TumorClassification import TumorClassificationSimple
from datasets import load_from_disk
from tqdm import tqdm
from transformers import set_seed

def create_batch_prompts(samples_text, input_language, task_description, schema):
  prompts = []
  for sample in samples_text:
    prompt = f"""You are a helpful assistant. Your task is to extract the following tumor related information from a {input_language} bone and soft tissue tumor pathology report:
    {task_description}
    Besides classifying the tumor based on the given criteria, you should also retrieve the text sequences that contain the tumor information. The tumor information should be classified according to the following schema:
    {schema}
    Retrieve and extract the relevant information from the following report:
    {sample}"""
    prompts.append(prompt)
  return prompts

def batch_inference(model_name, dataset, batch_size=16):

  model = outlines.models.transformers(
    model_name,
    device="auto"
  )
  generator = outlines.generate.json(model, TumorClassificationSimple)
    
  results = []

  num_samples = len(dataset['test'])
  num_batches = (num_samples + batch_size - 1) // batch_size
    
  input_language = "German"
  task_description = open('prompt_eng_new.txt').read()
    
  for i in tqdm(range(num_batches), desc="Processing batches"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, num_samples)
    

    batch_samples = dataset['test'][start_idx:end_idx]
    batch_prompts = create_batch_prompts(
      batch_samples['text'],
      input_language,
      task_description,
      TumorClassificationSimple.model_json_schema()
    )
    
    try:
      batch_predictions = generator(batch_prompts)
      
      for sample, prediction in zip(batch_samples['label'], batch_predictions):
        results.append({
          'label': sample,
          'prediction': prediction.model_dump_json()
        })
          
    except Exception as e:
      print(f"Error processing batch {i}: {str(e)}")
      # Add None predictions for failed samples
      for sample in batch_samples['label']:
        results.append({
          'label': sample,
          'prediction': None
        })
    
  return results

if __name__ == "__main__":
  SEED = 2442
  set_seed(SEED)
  load_dotenv()
  HF_KEY = os.getenv("HF_KEY")
  login(token=HF_KEY)
  model_name = 'ortho-ft-5dim-real'
  dataset = load_from_disk('private-dataset/ortho-real-strat-5cols')
  results = batch_inference(model_name, dataset)
  
  df = pd.DataFrame(results)
  df.to_csv(f'{model_name.replace("/","-")}_results_{SEED}.csv', index=False, sep=';')
  print(f"Results saved to {model_name.replace('/','-')}_results.csv")