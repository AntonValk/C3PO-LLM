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

MODEL_DICT = {'qwen': MODELS_QWEN, 'mistral': MODELS_MISTRAL, 'llama': MODELS_LLAMA, 'deepseek': MODELS_DEEPSEEK}


def run_CoT_SC(dataset, llm, sampling_budget, N_cal):
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

    correct = 0
    cost = []
    corr = []
    for i in range(len(data_["responses"])):
        true_answer = data_["responses"][i]["data_entry"]["answer"]
        if dataset in ['bigbench_date', 'bigbench_disambiguationQA', 'bigbench_geometric_shapes',
                       'bigbench_movie_recommendation', 'bigbench_penguins', 'bigbench_ruin_names', 'bigbench_snarks',
                       'bigbench_temporal_sequences']:
            true_answer = true_answer.replace('(', '').replace(')', '')

        try:
            ans_idx_ = np.random.choice(np.arange(len(data_["responses"][i]["decoded_answers"])), sampling_budget)

            sampled_ = [data_["responses"][i]["decoded_answers"][int(j)] for j in ans_idx_]
            counts_ = Counter(sampled_)
            final_answer = counts_.most_common(1)[0][0]

            cost_i = cost_['cost'][i]['ip_cost'] + np.sum([cost_['cost'][i]['op_cost'][int(j)] for j in ans_idx_])

            cost.append(cost_i)

            if dataset in ['bigbench_date', 'bigbench_disambiguationQA', 'bigbench_geometric_shapes',
                           'bigbench_movie_recommendation', 'bigbench_penguins', 'bigbench_ruin_names',
                           'bigbench_snarks', 'bigbench_temporal_sequences'] and final_answer is not None:
                final_answer = final_answer.replace('(', '').replace(')', '')
        except:
            final_answer = None
            cost.append(0.)

        if final_answer == true_answer:
            correct += 1
            corr.append(1)
        else:
            corr.append(0)

        acc__ = np.mean(corr)
        cost__ = np.mean(cost)

    print('------------------------')
    print('LLM: ', llm, ' , Budget: ', sampling_budget)
    print("Accuracy: ", acc__)
    print("Cost: ", cost__)

    return acc__, cost__, corr


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
    # for dataset in ['aqua', 'bigbench_causal_judgement', 'bigbench_date', 'bigbench_disambiguationQA',
    #                 'bigbench_formal_fallacies', 'bigbench_geometric_shapes', 'bigbench_movie_recommendation',
    #                 'bigbench_penguins', 'bigbench_ruin_names', 'bigbench_snarks', 'bigbench_sports',
    #                 'bigbench_temporal_sequences', 'commonsenseQA', 'GSM8K', 'SVAMP', 'math_500']:

    for dataset in ['aqua']:

        # llm_families = [MODELS_LLAMA, MODELS_QWEN, MODELS_GPT]
        llm_families = [MODELS_QWEN]
        N_cal = 50

        for llm_family in llm_families:

            print('Now running on : ' + dataset)

            for llm in llm_family:
                res = {}
                for sampling_budget in [5, 10, 15, 20, 25, 30, 35, 40]:
                    acc_, cost_, corr = run_CoT_SC(dataset, llm, sampling_budget, N_cal)
                    # if acc_ > 0 and acc_ < 1.0:
                    #     acc, lo, hi = bootstrap_accuracy_scipy(corr)
                    # else:
                    #     acc = lo = hi = acc_
                    # res[cost_] = [acc_, lo, hi]
                    res[sampling_budget] = [acc_, cost_]

                Path(f"logs/{dataset}/CoT_SC/benchmark").mkdir(parents=True, exist_ok=True)

                llm_name = fr'{llm}'.split("/")[-1]

                with open(f'logs/{dataset}/CoT_SC/benchmark/{llm_name}.json', 'w+') as fp:
                    json.dump(res, fp)

