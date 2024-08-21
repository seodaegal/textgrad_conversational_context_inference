import argparse
import concurrent
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import random

load_dotenv(override=True)

import sys
sys.path.append('/root/textgrad')
import textgrad as tg
from textgrad.tasks import load_task
from textgrad.tasks.bbung_task import BBUNG

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
def eval_sample(x, y, model):
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    # y 값을 int로 변환
    y = int(y) if isinstance(y, np.float64) else y
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    return response
def eval_dataset(test_set, eval_fn, model, max_samples: int=None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            
            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list 
def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]
    
    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)
set_seed(12)
llm_api_eval = tg.get_engine(engine_name="gpt-4o")
llm_api_test = tg.get_engine(engine_name="gpt-3.5-turbo-0125")
tg.set_backward_engine(llm_api_eval, override=True)

# Load the data and the evaluation function
train_set, val_set, test_set, eval_fn = load_task("BBUNG_BBUNG", evaluation_api=llm_api_eval)
print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
STARTING_SYSTEM_PROMPT = train_set.get_task_description()

print(train_set.__getitem__(0).keys())