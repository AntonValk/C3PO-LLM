import os
import json, itertools
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

MODEL_DICT = {'qwen': MODELS_QWEN, 'mistral': MODELS_MISTRAL, 'llama': MODELS_LLAMA, 'deepseek': MODELS_DEEPSEEK}

def calculate_weight(answers):
    answers = [z for z in answers if z is not None]
    l = len(answers)

    answer__ = Counter(answers).most_common()

    unique_answers = [x_[0] for x_ in answer__]
    unique_weights = [x_[1] for x_ in answer__]
    unique_weights = unique_weights / np.sum(unique_weights)

    entropy = -np.sum(unique_weights * np.log(unique_weights))

    weight = 1 / l + (1 - 1/l)*(1 - entropy/np.log(l))
    return [weight] * l

def run_model_switch(dataset, llm_list, sample_budget, N_cal):
    data_all = {}
    cost_all = {}

    for llm in llm_list:
        try:
            with open(f'output/{llm}_{dataset}_fs_cot_40.json', 'r', encoding='utf-8') as f:
                data_ = json.load(f)
            with open(f'costs/{llm}_{dataset}_fs_cot_40_costs.json', 'r', encoding='utf-8') as f:
                cost_ = json.load(f)
        except:
            with open(f'output/{llm}_{dataset}_zs_cot_3.json', 'r', encoding='utf-8') as f:
                data_ = json.load(f)
            with open(f'costs/{llm}_{dataset}_fs_cot_3_costs.json', 'r', encoding='utf-8') as f:
                cost_ = json.load(f)

        data_all[llm] = data_
        cost_all[llm] = cost_

    correct = 0
    cost = []
    corr =[]
    for i in range(2*N_cal, len(data_all[llm_list[0]]["responses"])):
        early_stop = False
        true_answer = data_all[llm_list[0]]["responses"][i]["data_entry"]["answer"]
        if dataset in ['bigbench_date', 'bigbench_disambiguationQA', 'bigbench_geometric_shapes', 'bigbench_movie_recommendation', 'bigbench_penguins', 'bigbench_ruin_names', 'bigbench_snarks', 'bigbench_temporal_sequences']:
            true_answer = true_answer.replace('(', '').replace(')', '')

        cost_i = 0.

        all_ans_i = []
        all_weights_i = []

        for llm in llm_list:
            try:
                ans_idx = np.random.choice(np.arange(len(data_all[llm]["responses"][i]["decoded_answers"])), sample_budget)
                sampled_ans = [data_all[llm]["responses"][i]["decoded_answers"][int(j)] for j in ans_idx]
            except:
                continue

            if dataset in ['bigbench_date', 'bigbench_disambiguationQA', 'bigbench_geometric_shapes',
                           'bigbench_movie_recommendation', 'bigbench_penguins', 'bigbench_ruin_names',
                           'bigbench_snarks', 'bigbench_temporal_sequences']:
                sampled_ans = [z.replace('(', '').replace(')', '') for z in sampled_ans if z is not None]

            cost_i += cost_all[llm]['cost'][i]['ip_cost'] + np.sum([cost_all[llm]['cost'][i]['op_cost'][int(j)] for j in ans_idx])

            try:
                counts_ = Counter([z for z in sampled_ans if z is not None])
                answer_ = counts_.most_common(1)[0][0]
                if len(counts_) == 1:
                    final_answer = answer_
                    early_stop = True
                    break
            except:
                print('Go to next LLM')

            else:
                all_ans_i += [z for z in sampled_ans if z is not None]
                all_weights_i += calculate_weight([z for z in sampled_ans if z is not None])

        if not early_stop:
            z = sorted(zip(all_ans_i, all_weights_i), key=lambda x__: x__[0])
            weighted_results = []
            for k, g in itertools.groupby(z, key=lambda x__: x__[0]):
                weighted_results.append((k, sum([t[1] for t in g])))

            weighted_results.sort(key=lambda x__: x__[1])
            final_answer = weighted_results[-1][0]

        cost.append(cost_i)

        if final_answer == true_answer:
            correct += 1
            corr.append(1.0)
        else:
            corr.append(0)

    acc = np.mean(corr)
    cost_avg = np.mean(cost)
    print('------------------------')
    print("Accuracy: ", acc)
    print("Cost: ", cost_avg)

    return acc, cost_avg, corr

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
    

# Usage example
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
              all_models = ['openai_gpt/gpt35_turbo', 'openai_gpt/gpt4o_mini', 'openai_gpt/o3_mini']
          elif family == 'QWEN':
              all_models = ['qwen/qwen_1b', 'qwen/qwen_32b', 'qwen/qwen_72b']
          elif family == 'LLAMA':
              all_models = ['llama/llama_1b_32', 'llama/llama_3b_32', 'llama/llama_70b_33', 'llama/llama_405b_31']
          else:
              print('error!')

          for z in [5]:
            for j in [len(all_models)]:
              for llms in itertools.combinations(all_models, j):
                llm_list = list(llms)
                # print(llm_list)
                sample_budget = z
                Path(f"logs/{dataset}/Switch/{family}/boot").mkdir(parents=True, exist_ok=True)
                # model_string = fr'{"".join(llm_list)}'.split("/")[-1]
                acc_, cost_, corr_ = run_model_switch(dataset, llm_list, sample_budget, N_cal)
                if acc_ > 0 and acc_ < 1.0:
                    acc, lo, hi = bootstrap_accuracy_scipy(corr_)
                else:
                    acc = lo = hi = acc_

                llm_list = fr'{"".join(llm_list)}'.split("/")
                model_string = "-".join(llm_list)
                res = {}
                res[cost_] = [acc_, lo, hi]
                with open(f'logs/{dataset}/Switch/{family}/boot/{model_string}_budget_{sample_budget}.json', 'w+') as fp:
                  json.dump(res, fp)

