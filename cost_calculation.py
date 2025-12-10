import json
import random, os
import tiktoken
import numpy as np
from collections import Counter


def count_tokens(text, engine="gpt-3.5-turbo"):
    # try:
    #     encoding = tiktoken.encoding_for_model(engine)
    #     return len(encoding.encode(text))
    # except:
    #     # 3/4-th of each word is roughly one token as per OpenAI's documentation
    #     return int(round(1.25*len(text.split())))
    return int(round(1.25 * len(text.split())))


datasets = ['aqua', 'bigbench_causal_judgement', 'bigbench_date', 'bigbench_disambiguationQA',
            'bigbench_formal_fallacies', 'bigbench_geometric_shapes', 'bigbench_movie_recommendation',
            'bigbench_penguins', 'bigbench_ruin_names', 'bigbench_snarks', 'bigbench_sports',
            'bigbench_temporal_sequences', 'commonsenseQA', 'GSM8K', 'SVAMP', 'math_500', 'aime']

# datasets = ['aqua']

model_families = ['llama', 'qwen', 'mixtral', 'openai_gpt']
# model_families = ['qwen']


for dataset in datasets:
    for model_family in model_families:

        #################################################################################################
        if model_family == 'llama':
            model_dollar_costs = {
                        'llama/llama_1b_32': {'input_cost': 0.005, 'output_cost': 0.01},
                        'llama/llama_3b_32': {'input_cost': 0.01, 'output_cost': 0.02},
                        'llama/llama_70b_33': {'input_cost': 0.13, 'output_cost': 0.40},
                        'llama/llama_405b_31': {'input_cost': 1.00, 'output_cost': 3.00}
                    }
        ####################################################################################################
        if model_family == 'qwen':
            model_dollar_costs = {
                        'qwen/qwen_1b': {'input_cost': 0.02, 'output_cost': 0.06},
                        'qwen/qwen_32b': {'input_cost': 0.06, 'output_cost': 0.20},
                        'qwen/qwen_72b': {'input_cost': 0.13, 'output_cost': 0.40}
                    }
        ####################################################################################################
        if model_family == 'mixtral':
            model_dollar_costs = {
                        'mixtral/mixtral_8x7b': {'input_cost': 0.08, 'output_cost': 0.24},
                        'mixtral/mixtral_8x22b': {'input_cost': 0.40, 'output_cost': 1.20}
                    }
        #####################################################################################################
        if model_family == 'openai_gpt':
            model_dollar_costs = {
                        'openai_gpt/gpt35_turbo': {'input_cost': 0.50, 'output_cost': 1.50},
                        'openai_gpt/gpt4o_mini': {'input_cost': 0.15, 'output_cost': 0.60},
                        'openai_gpt/o3_mini': {'input_cost': 1.10, 'output_cost': 4.40}
                    }


        for model in list(model_dollar_costs.keys()):

            try:
                log_dir = 'costs/' + model_family + '/'
                if not os.path.exists(os.path.dirname(log_dir)):
                    os.makedirs(os.path.dirname(log_dir))

                try:
                    with open(f'output/{model}_{dataset}_fs_cot_40.json', 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        #############################
                        detailed_cost = []
                        for q_idx, r in enumerate(data['responses']):
                            ip_cost = 1e-6 * model_dollar_costs.get(model)['input_cost'] * \
                                      (count_tokens(r["request"]["message"]) + count_tokens(r["request"]["system_prompt"]))
                            try:
                                op_cost = [1e-6 * model_dollar_costs.get(model)['output_cost'] * \
                                           count_tokens(CoT["message"]) for CoT in r["response"]["responses"]]
                            except:
                                try:
                                    op_cost = np.random.choice(detailed_cost[q_idx-1]['op_cost'], len(r["response"]["responses"]))
                                except:
                                    op_cost = []

                            detailed_cost.append({'ip_cost': ip_cost, 'op_cost': op_cost})
                            # print({'ip_cost': ip_cost, 'op_cost': op_cost})

                        with open('costs/' + f'{model}_{dataset}_fs_cot_40_costs.json', 'w') as json_file:
                            json.dump({'cost': detailed_cost}, json_file, indent=4)
                except:
                    with open(f'output/{model}_{dataset}_zs_cot_3.json', 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    #############################
                    detailed_cost = []
                    for q_idx, r in enumerate(data['responses']):
                        ip_cost = 1e-6 * model_dollar_costs.get(model)['input_cost'] * \
                                  (count_tokens(r["request"]["message"]) + count_tokens(r["request"]["system_prompt"]))
                        try:
                            op_cost = [1e-6 * model_dollar_costs.get(model)['output_cost'] * \
                                       count_tokens(CoT["message"]) for CoT in r["response"]["responses"]]
                        except:
                            try:
                                op_cost = np.random.choice(detailed_cost[q_idx - 1]['op_cost'],
                                                           len(r["response"]["responses"]))
                            except:
                                op_cost = []

                        detailed_cost.append({'ip_cost': ip_cost, 'op_cost': op_cost})
                        # print({'ip_cost': ip_cost, 'op_cost': op_cost})

                    with open('costs/' + f'{model}_{dataset}_fs_cot_3_costs.json', 'w') as json_file:
                        json.dump({'cost': detailed_cost}, json_file, indent=4)

                # print(dataset, model_family, model, ' cost calculation done!' )
            except:
                print(dataset, model_family, model, ' cost calculation is not done!')
                print('Go to the next configuration!')





