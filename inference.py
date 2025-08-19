import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import random
import numpy as np
import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import ujson as json
from tqdm import tqdm
import numpy as np
from scipy import stats

def random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

random_seed(42)

model_name = "path_to_checkpoint/LLaMA-Factory/output/llama3-3B_mm_MF_time2"
generator = pipeline(
    "text-generation", 
    model_name, 
    torch_dtype="auto", 
    device_map="auto",
)
generator.tokenizer.padding_side="left"

def batch_generate(prompts):
    batch = []
    for prompt in prompts:
        batch.append([{"role": "user", "content": prompt}])
    
    results = generator(batch, max_new_tokens=768, batch_size=btz)
    results = [result[0]["generated_text"][-1]['content'] for result in results]
    # print(results)
    # xxx
    results = [r.split("</think>")[-1].strip() for r in results]
    # print(results)
    return results 

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    assert len(y_true) == len(y_pred), "Inconsistent length of input list"
    
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    
    spearman_corr, _ = stats.spearmanr(y_true, y_pred)
    
    return {
        "MAE": mae,
        "MSE": mse,
        "Spearman": spearman_corr
    }

def main():
    test_data = json.load(open('path_to_test_data/dataset_MF_time/SMP_test_mm_MF_time.json','r'))
    # Prepare the input data for batch inference
    inputs = [
        item['input']  # Concatenate instruction and input
        for item in test_data
    ]
    post_ids = [
        "post"+str(item['pid'])  # Concatenate instruction and input
        for item in test_data
    ]

    btz = 64
    results = []
    for i in tqdm(range(int(len(test_data)/btz)+1)):
        batch_inputs = inputs[i*btz:(i+1)*btz]
        batch_outputs = batch_generate(batch_inputs)
        # print(batch_outputs)
        # xxx
        for j in batch_outputs:
            try:
                results.append(float(j.split("assistant")[-1].strip()))
            except:
                print("wrong in parse!")
                results.append(5)
    
    print(len(post_ids))
    print(len(results))

    gt = []
    for item in test_data:
        gt.append(float(item["output"]))
    metrics = calculate_metrics(results, gt)
    print(metrics)
    
if __name__ == "__main__":
    main()