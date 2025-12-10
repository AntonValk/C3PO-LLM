import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict

np.random.seed(42)


class BruteForceCascadeOptimizer:
    def __init__(self,
                 dataset_name,
                 model_families,
                 model_costs,
                 cost_budget=2.0,
                 num_responses=40,
                 calibration_size=50,
                 threshold_steps=15,
                 alpha=0.1):
        self.dataset = dataset_name
        self.model_families = model_families
        self.model_costs = model_costs
        self.cost_budget = cost_budget
        self.num_responses = num_responses
        self.N_cal = calibration_size
        self.N_ss = calibration_size  # will set after loading questions
        self.threshold_steps = threshold_steps
        self.alpha = alpha

        # load all models' responses & costs
        self.models = self._load_models()
        # sort by cost
        self.sorted_models = sorted(self.models.keys(),
                                    key=lambda m: self.models[m]['cost'])
        self.oracle = self.sorted_models[-1]

        # load data entries
        self._load_questions()

        # split indices
        self.ss_indices = np.arange(0, self.N_ss)
        self.cal_indices = np.arange(self.N_ss, self.N_ss + self.N_cal)

    def _get_model_list(self, family):
        MODEL_DICT = {
            'qwen': ['qwen/qwen_1b', 'qwen/qwen_32b', 'qwen/qwen_72b'],
            'llama': ['llama/llama_1b_32', 'llama/llama_3b_32', 'llama/llama_70b_33', 'llama/llama_405b_31'],
            'gpt': ['openai_gpt/gpt35_turbo', 'openai_gpt/gpt4o_mini', 'openai_gpt/o3_mini']
        }
        return MODEL_DICT.get(family, [])

    def _default_cost(self, model):
        # parse trailing number
        parts = model.split('_')[-1]
        num = ''.join([c for c in parts if c.isdigit() or c == '.'])
        return float(num) if num else 1.0

    def _load_models(self):
        models = {}
        for fam in self.model_families:
            for model in self._get_model_list(fam):
                try:
                    path = f'output/{model}_{self.dataset}_fs_cot_40.json'
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    path_ = f'costs/{model}_{self.dataset}_fs_cot_40_costs.json'
                    with open(path_, 'r', encoding='utf-8') as f_:
                        dollar_cost_ = json.load(f_)
                        dollar_cost = [z for z in dollar_cost_['cost']]
                except:
                    path = f'output/{model}_{self.dataset}_zs_cot_3.json'
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    path_ = f'costs/{model}_{self.dataset}_fs_cot_3_costs.json'
                    with open(path_, 'r', encoding='utf-8') as f_:
                        dollar_cost_ = json.load(f_)
                        dollar_cost = [z for z in dollar_cost_['cost']]

                cost = self.model_costs.get(model,
                                            self._default_cost(model))
                # each entry is list of decoded_answers
                resp_lists = [r['decoded_answers'] for r in data['responses']]
                models[model] = {'cost': cost, 'responses': resp_lists, 'dollar_cost': dollar_cost}
        return models

    def _load_questions(self):
        # just need to load answers for SS vs Cal splits
        # assume all models share same 'data_entry's
        first = self.sorted_models[0]
        path = f'output/{first}_{self.dataset}_fs_cot_40.json'
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.questions = [r['data_entry'] for r in data['responses']]
        # total questions = SS + Cal
        # self.N_ss = len(self.questions) - self.N_cal

    def get_true_answer(self, q_idx):
        # data_entry contains 'answer' field
        true_answer = self.questions[q_idx]['answer']
        if self.dataset in ['bigbench_date', 'bigbench_disambiguationQA', 'bigbench_geometric_shapes',
                            'bigbench_movie_recommendation', 'bigbench_penguins', 'bigbench_ruin_names',
                            'bigbench_snarks', 'bigbench_temporal_sequences']:
            true_answer = true_answer.replace('(', '').replace(')', '')

        return true_answer

    def _get_model_answer_and_conf(self, model, q_idx):
        try:
            resps = self.models[model]['responses'][q_idx]
            api_cost = self.models[model]['dollar_cost'][q_idx]
            sampled_idx = np.random.choice(np.arange(len(resps)), self.num_responses)
            sampled = [resps[int(j)] for j in sampled_idx]
            if self.dataset in ['bigbench_date', 'bigbench_disambiguationQA', 'bigbench_geometric_shapes',
                                'bigbench_movie_recommendation', 'bigbench_penguins', 'bigbench_ruin_names',
                                'bigbench_snarks', 'bigbench_temporal_sequences']:
                sampled = [z.replace('(', '').replace(')', '') for z in sampled if z is not None]

            cnt = Counter(sampled)
            ans, freq = cnt.most_common(1)[0]
            norm_prob = len(sampled)
            incurred_cost = api_cost['ip_cost'] + np.sum([api_cost['op_cost'][int(j)] for j in sampled_idx])
        except:
            ans, freq, norm_prob, incurred_cost = None, 0., self.num_responses, 0.

        return ans, freq / norm_prob, incurred_cost

    def _compute_loss(self, thresholds, indices, split='train'):
        # zero–one loss on SS
        mistakes = 0
        corr = []
        for i in indices:
            # simulate cascade exit
            answer = None
            for m in self.sorted_models:
                ans, conf, _ = self._get_model_answer_and_conf(m, i)
                try:
                    if conf >= thresholds[m]:
                        answer = ans
                        break
                except:
                    answer = ans
                    break
            if answer is None:
                # fallback to oracle
                answer, _, _ = self._get_model_answer_and_conf(self.oracle, i)
            # compare to oracle's actual output
            if split == 'train':
                true_ans, _, _ = self._get_model_answer_and_conf(self.oracle, i)
            elif split == 'test':
                true_ans = self.get_true_answer(i)
            if answer != true_ans:
                mistakes += 1
                corr.append(0)
            else:
                corr.append(1)
        return mistakes / len(indices), corr

    def _compute_costs(self, thresholds, indices):
        # cumulative cost per query on Cal
        costs = []
        true_costs = []
        exits = []
        for i in indices:
            cum_cost = 0.0
            cum_cost_true = 0.0
            exited = False
            for model_id, m in enumerate(self.sorted_models):
                try:
                    if thresholds[m] > 1:
                        continue

                    cum_cost += self.models[m]['cost']
                    _, conf, api_cost_ = self._get_model_answer_and_conf(m, i)
                    cum_cost_true += api_cost_
                    if conf >= thresholds[m]:
                        exits.append(model_id)
                        exited = True
                        break
                except:
                    cum_cost += self.models[self.sorted_models[0]]['cost']
                    _, _, api_cost_ = self._get_model_answer_and_conf(self.sorted_models[0], i)
                    cum_cost_true += api_cost_
                    exits.append(model_id)
                    exited = True
                    break

            if not exited:
                exits.append(len(self.sorted_models) - 1)
                exited = True
            costs.append(cum_cost)
            true_costs.append(cum_cost_true)
        return np.array(costs), np.array(true_costs)  # , exits

    def brute_force_search(self):
        # grid = None
        # build uniform grid for each model except oracle
        # if self.threshold_steps == 0: # naively always exit at final model (should match SC-CoT)
        # grid = np.array([1.1])
        # else:
        grid = np.linspace(0.0, 1.1, self.threshold_steps)
        # small_grid = np.linspace(0.0, 1.1, 3)
        model_grids = {m: grid for m in self.sorted_models[:-1]}
        # model_grids[self.sorted_models[0]] = small_grid
        # model_grids[self.sorted_models[1]] = small_grid
        model_grids[self.oracle] = np.array([0.0])

        # total combinations for progress bar
        num_coords = len(self.sorted_models) - 1
        total_combos = self.threshold_steps ** num_coords

        best_tau = None
        best_loss = float('inf')

        # iterate with tqdm
        prod_iter = itertools.product(
            *[model_grids[m] for m in self.sorted_models]
        )
        for combo in tqdm(prod_iter, total=total_combos, desc="Brute‐forcing thresholds"):
            thresholds = {m: combo[i] for i, m in enumerate(self.sorted_models)}

            # 1) evaluate loss on SS
            loss, _ = self._compute_loss(thresholds, self.ss_indices)

            # 2) evaluate costs on Cal
            costs, _ = self._compute_costs(thresholds, self.cal_indices)
            # quantile check4
            k = int(np.ceil((len(costs) + 1) * (1 - self.alpha))) - 1
            q = np.sort(costs)[k]

            if q <= self.cost_budget and loss < best_loss:
                best_loss = loss
                best_tau = thresholds.copy()

        self.thresholds = best_tau
        print(f"\nBest SS‐loss: {best_loss:.4f}")
        print("Best thresholds:", best_tau)

    def evaluate(self, split='test'):
        if split == 'test':
            idx = np.arange(self.N_ss + self.N_cal, len(self.questions))
        else:
            idx = np.arange(0, self.N_ss)
        loss, corr = self._compute_loss(self.thresholds, idx, split=split)
        costs, true_costs = self._compute_costs(self.thresholds, idx)
        avg_cost = costs.mean()
        true_avg_cost = true_costs.mean()
        print(f"{split} accuracy: {1 - loss:.2%}, avg cost: {avg_cost:.2f}")
        return 1 - loss, avg_cost, true_avg_cost, corr


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
    # for dataset in ['aqua', 'bigbench_causal_judgement', 'bigbench_date', 'bigbench_disambiguationQA',
    #                 'bigbench_formal_fallacies', 'bigbench_geometric_shapes', 'bigbench_movie_recommendation',
    #                 'bigbench_penguins', 'bigbench_ruin_names', 'bigbench_snarks', 'bigbench_sports',
    #                 'bigbench_temporal_sequences', 'commonsenseQA', 'GSM8K', 'SVAMP', 'math_500']:
    for dataset in ['GSM8K', 'SVAMP']:

        test_dataset = 'math_500'

        family = "LLAMA"  # "LLAMA" #"GPT" #"QWEN" #

        for z in [5]:
            sample_budget = z
            res = {}

            if family == 'GPT':
                model_fam = ['gpt']
                model_costs = {
                    'openai_gpt/gpt35_turbo': 1.50,
                    'openai_gpt/gpt4o_mini': 0.60,
                    'openai_gpt/o3_mini': 4.40
                }
                # budget_range = np.linspace(0, 7, 7)
                budget_range = np.linspace(0, 9, 20)

            elif family == 'QWEN':
                model_fam = ['qwen']
                model_costs = {
                    'qwen/qwen_1b': 0.06,
                    'qwen/qwen_32b': 0.20,
                    'qwen/qwen_72b': 0.40,
                }
                # budget_range = np.linspace(0, 0.75, 6)
                budget_range = np.linspace(0, 1, 10)

            elif family == 'LLAMA':
                model_fam = ['llama']
                model_costs = {
                    'llama/llama_1b_32': 0.01,
                    'llama/llama_3b_32': 0.02,
                    'llama/llama_70b_33': 0.40,
                    'llama/llama_405b_31': 3.00
                }
                # budget_range = np.linspace(0, 5, 6)
                budget_range = np.linspace(0, 5, 10)

            else:
                print('error!')

            for i in budget_range:
                print("Training on ", dataset)
                optimizer = BruteForceCascadeOptimizer(
                    dataset_name=dataset,
                    model_families=model_fam,
                    model_costs=model_costs,
                    cost_budget=i,
                    calibration_size=50,
                    num_responses=sample_budget,
                    threshold_steps=z + 1,
                    alpha=0.1
                )

                optimizer.brute_force_search()

                print("Testing on ", test_dataset)
                optimizer_ = BruteForceCascadeOptimizer(
                    dataset_name=test_dataset,
                    model_families=model_fam,
                    model_costs=model_costs,
                    cost_budget=i,
                    calibration_size=50,
                    num_responses=sample_budget,
                    threshold_steps=z + 1,
                    alpha=0.1
                )
                optimizer_.thresholds = optimizer.thresholds

                acc, cost, true_dollar_cost, corr = optimizer_.evaluate('test')

                if acc > 0 and acc < 1.0:
                    acc, lo, hi = bootstrap_accuracy_scipy(corr)
                else:
                    lo = hi = acc

                print('res', acc, cost, true_dollar_cost)
                res[true_dollar_cost] = [acc, lo, hi]

            Path(f"logs/{dataset}_{test_dataset}/Ours_shift/{family}/boot").mkdir(parents=True, exist_ok=True)
            llm_list = optimizer._get_model_list(model_fam[0])
            llm_list = fr'{"".join(llm_list)}'.split("/")
            model_string = "-".join(llm_list)

            with open(f'logs/{dataset}_{test_dataset}/Ours_shift/{family}/boot/{model_string}_budget_{z}.json', 'w+') as fp:
                json.dump(res, fp)
