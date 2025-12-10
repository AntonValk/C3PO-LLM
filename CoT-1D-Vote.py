import os
import json
import random
import numpy as np
from collections import Counter
from pathlib import Path

np.random.seed(42)

# FAILED: GEMMA, 'llama/llama_8b_31',

MODELS_QWEN = ['qwen/qwen_1b', 'qwen/qwen_32b', 'qwen/qwen_72b']
MODELS_MISTRAL = ['mixtral/mixtral_8x7b', 'mixtral/mixtral_8x22b']
MODELS_LLAMA = ['llama/llama_1b_32', 'llama/llama_3b_32', 'llama/llama_70b_33', 'llama/llama_405b_31']
MODELS_DEEPSEEK = ['deepseek/deepseek_v3']
MODELS_GPT = ['openai_gpt/gpt35_turbo', 'openai_gpt/gpt4o_mini', 'openai_gpt/o3_mini']

# MODEL_DICT = {'qwen': MODELS_QWEN, 'mistral': MODELS_MISTRAL, 'llama': MODELS_LLAMA, 'deepseek': MODELS_DEEPSEEK}

def run_CoT_1D_Vote(dataset, weak_llm, strong_llm, sample_budget, threshold, N_cal):
    with open(f'output/{weak_llm}_{dataset}_fs_cot_40.json', 'r', encoding='utf-8') as f:
        data_w = json.load(f)
    with open(f'costs/{weak_llm}_{dataset}_fs_cot_40_costs.json', 'r', encoding='utf-8') as f:
        cost_w = json.load(f)

    try:
        with open(f'output/{strong_llm}_{dataset}_fs_cot_40.json', 'r', encoding='utf-8') as f:
            data_s = json.load(f)
        with open(f'costs/{strong_llm}_{dataset}_fs_cot_40_costs.json', 'r', encoding='utf-8') as f:
            cost_s = json.load(f)
    except:
        with open(f'output/{strong_llm}_{dataset}_zs_cot_3.json', 'r', encoding='utf-8') as f:
            data_s = json.load(f)
        with open(f'costs/{strong_llm}_{dataset}_fs_cot_3_costs.json', 'r', encoding='utf-8') as f:
            cost_s = json.load(f)

    correct = 0
    cost = []
    corr = []
    for i in np.arange(2*N_cal, len(data_w["responses"])):
        true_answer = data_w["responses"][i]["data_entry"]["answer"]
        if dataset in ['bigbench_date', 'bigbench_disambiguationQA', 'bigbench_geometric_shapes', 'bigbench_movie_recommendation', 'bigbench_penguins', 'bigbench_ruin_names', 'bigbench_snarks', 'bigbench_temporal_sequences']:
            true_answer = true_answer.replace('(', '').replace(')', '')

        try:
            ans_idx_w = np.random.choice(np.arange(len(data_w["responses"][i]["decoded_answers"])), sample_budget)

            sampled_w = [data_w["responses"][i]["decoded_answers"][int(j)] for j in ans_idx_w]
            counts_w = Counter(sampled_w)
            answer_w = counts_w.most_common(1)[0][0]
            confidence_w = counts_w[answer_w] / sample_budget

            cost_i = cost_w['cost'][i]['ip_cost'] + np.sum([cost_w['cost'][i]['op_cost'][int(j)] for j in ans_idx_w])

            final_answer = answer_w
        except:
            final_answer = None
            cost_i = 0.
            confidence_w = 0.

        if confidence_w < threshold:
            try:
                ans_idx_s = np.random.choice(np.arange(len(data_s["responses"][i]["decoded_answers"])), sample_budget)

                sampled_s = [data_s["responses"][i]["decoded_answers"][int(j)] for j in ans_idx_s]
                counts_s = Counter(sampled_s)
                answer_s = counts_s.most_common(1)[0][0]

                cost_i += cost_s['cost'][i]['ip_cost'] + np.sum([cost_s['cost'][i]['op_cost'][int(j)] for j in ans_idx_s])

                final_answer = answer_s
            except:
                final_answer = None

        cost.append(cost_i)

        if dataset in ['bigbench_date', 'bigbench_disambiguationQA', 'bigbench_geometric_shapes', 'bigbench_movie_recommendation', 'bigbench_penguins', 'bigbench_ruin_names', 'bigbench_snarks', 'bigbench_temporal_sequences'] and final_answer is not None:
            final_answer = final_answer.replace('(', '').replace(')', '')

        if final_answer == true_answer:
            correct += 1
            corr.append(1.0)
        else:
            corr.append(0)

    acc = np.mean(corr)
    cost_avg = np.mean(cost)
    print('------------------------')
    print('Weak LLM: ', weak_llm)
    print('Strong LLM: ', strong_llm)
    print("Threshold: ", threshold)
    print("Accuracy: ", acc)
    print("Cost: ", cost_avg)

    return acc, cost_avg, corr

# Usage example

from scipy.stats import bootstrap
def bootstrap_accuracy_scipy(acc_list, 
                             n_resamples=1000, 
                             ci_level=0.90, 
                             random_seed=42):
    data = np.asarray(acc_list)
    orig_mean = data.mean()
    
    # SciPy's bootstrap expects a sequence of arrays (one per variable axis)
    res = bootstrap((data,), 
                    np.mean, 
                    confidence_level=ci_level,
                    n_resamples=n_resamples,
                    vectorized=False,
                    random_state=random_seed)
    
    # res.confidence_interval is a ConfidenceInterval(low, high)

    ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
    print(orig_mean, ci_low, ci_high)
    return orig_mean, ci_low, ci_high
    

if __name__ == "__main__":
    for dataset in ['aqua', 'bigbench_causal_judgement', 'bigbench_date', 'bigbench_disambiguationQA',
                    'bigbench_formal_fallacies', 'bigbench_geometric_shapes', 'bigbench_movie_recommendation',
                    'bigbench_penguins', 'bigbench_ruin_names', 'bigbench_snarks', 'bigbench_sports',
                    'bigbench_temporal_sequences', 'commonsenseQA', 'GSM8K', 'SVAMP', 'math_500']:

      families = ["GPT", "LLAMA", "QWEN"]
      N_cal = 50

      for family in families:

          print("Now running", dataset)

          if family == 'GPT':
              weak_llm = 'openai_gpt/gpt35_turbo'
              strong_llm = 'openai_gpt/o3_mini'
          elif family == 'QWEN':
              weak_llm = 'qwen/qwen_1b'
              strong_llm = 'qwen/qwen_72b'
          elif family == 'LLAMA':
              weak_llm = 'llama/llama_1b_32'
              strong_llm = 'llama/llama_405b_31'
          else:
              print('error!')

          weak = fr'{weak_llm}'.split("/")[-1]
          strong = fr'{strong_llm}'.split("/")[-1]
          sample_budget = 5
          Path(f"logs/{dataset}/MoT/{family}/boot").mkdir(parents=True, exist_ok=True)

          res = {}
          for threshold in np.arange(0.0, 0.999, 0.15):
              acc_, cost_, corr_ = run_CoT_1D_Vote(dataset, weak_llm, strong_llm, sample_budget, threshold, N_cal)
              if acc_ > 0 and acc_ < 1.0:
                  acc, lo, hi = bootstrap_accuracy_scipy(corr_)
              else:
                  acc = lo = hi = acc_
              res[cost_] = [acc, lo, hi]
          with open(f'logs/{dataset}/MoT/{family}/boot/{weak}_{strong}_budget_{sample_budget}.json', 'w+') as fp:
            json.dump(res, fp)