import os
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path

np.random.seed(42)


class FrugalRegressionCascade:
    def __init__(self,
                 dataset_name,
                 model_families,
                 model_costs,
                 cost_budget=2.0,
                 calibration_size=50,
                 num_responses=1,
                 threshold=0.5):
        self.dataset = dataset_name
        self.model_families = model_families
        self.model_costs = model_costs
        self.cost_budget = cost_budget
        self.num_responses = num_responses
        self.N_cal = calibration_size
        self.threshold = threshold

        # DistilBERT embedder
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased").eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)

        # Load all LLM outputs + costs
        self.models = self._load_models()
        self.sorted_models = sorted(self.models.keys(),
                                    key=lambda m: self.models[m]['cost'])
        # Data entries (with true answers)
        self._load_questions()

        # Split indices
        total = len(self.questions)
        self.ss_indices = np.arange(0, 2 * self.N_cal)
        self.test_indices = np.arange(2 * self.N_cal, total)

    def _get_model_list(self, family):
        return {
            'llama': ['llama/llama_1b_32', 'llama/llama_3b_32',
                      'llama/llama_70b_33', 'llama/llama_405b_31'],
            'qwen': ['qwen/qwen_1b', 'qwen/qwen_32b', 'qwen/qwen_72b'],
            'gpt': ['openai_gpt/gpt35_turbo', 'openai_gpt/gpt4o_mini', 'openai_gpt/o3_mini']
        }.get(family, [])

    def _default_cost(self, model):
        parts = model.split('_')[-1]
        num = ''.join(c for c in parts if (c.isdigit() or c == '.'))
        return float(num) if num else 1.0

    def _load_models(self):
        models = {}
        for fam in self.model_families:
            for model in self._get_model_list(fam):
                try:
                    # load answers
                    with open(f'output/{model}_{self.dataset}_fs_cot_40.json', encoding='utf-8') as f:
                        data = json.load(f)
                    # load dollar‐cost trace
                    with open(f'costs/{model}_{self.dataset}_fs_cot_40_costs.json', encoding='utf-8') as f:
                        cost_info = json.load(f)['cost']
                except:
                    # load answers
                    with open(f'output/{model}_{self.dataset}_zs_cot_3.json', encoding='utf-8') as f:
                        data = json.load(f)
                    # load dollar‐cost trace
                    with open(f'costs/{model}_{self.dataset}_fs_cot_3_costs.json', encoding='utf-8') as f:
                        cost_info = json.load(f)['cost']

                # keep only decoded answers list per query
                resp_lists = [r['decoded_answers'] for r in data['responses']]
                full_ans_list = [r['response']['responses'] if 'responses' in r['response'] else []
                                 for r in data['responses']]

                models[model] = {
                    'cost': self.model_costs.get(model, self._default_cost(model)),
                    'responses': resp_lists,
                    'dollar_cost': cost_info,
                    'full_answers': full_ans_list
                }
        return models

    def _load_questions(self):
        # assume all models share same data_entry
        first = self.sorted_models[0]
        with open(f'output/{first}_{self.dataset}_fs_cot_40.json', encoding='utf-8') as f:
            data = json.load(f)
        # extract the ground‐truth answer from each response's data_entry
        self.questions = [r['data_entry'] for r in data['responses']]

    def get_true_answer(self, q_idx):
        # data_entry contains 'answer' field
        true_answer = self.questions[q_idx]['answer']
        if self.dataset in ['bigbench_date', 'bigbench_disambiguationQA', 'bigbench_geometric_shapes',
                            'bigbench_movie_recommendation', 'bigbench_penguins', 'bigbench_ruin_names',
                            'bigbench_snarks', 'bigbench_temporal_sequences']:
            true_answer = true_answer.replace('(', '').replace(')', '')

        return true_answer

    def _get_model_full_answer(self, model, q_idx):
        try:
            resps = self.models[model]['full_answers'][q_idx]  # [q_idx]['message']
            a = np.random.choice(np.arange(len(resps)))
            sampled = resps[a]
            ans = self.models[model]['responses'][q_idx][a]
            pred = sampled['message']
        except:
            pred, ans, a = "", None, None
        return pred, ans, a

    def _embed(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt',
                                padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        # use [CLS] token representation
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def _build_dataset_(self, indices):
        X, y = [], []
        for i in indices:
            true = self.get_true_answer(i)
            for model in self.sorted_models[:-1]:
                pred, ans, _ = self._get_model_full_answer(model, i)
                label = int(ans == true)
                # embed question+predicted answer
                q = self.questions[i].get('question',
                                          self.questions[i].get('input', ''))
                if not pred:
                    vec = self._embed([q + " "])[0]
                else:
                    vec = self._embed([q + " " + pred])[0]
                X.append(vec)
                y.append(label)
        return np.vstack(X), np.array(y)

    def _build_dataset(self, indices):
        while True:
            feat, label_correctness = self._build_dataset_(indices)
            if np.mean(label_correctness) > 0. and np.mean(label_correctness) < 1.:
                break
        return feat, label_correctness

    def train(self):
        # Self‐supervision training
        X_ss, y_ss = self._build_dataset(self.ss_indices)
        self.model = LogisticRegression(max_iter=1000).fit(X_ss, y_ss)

        # # Select threshold on calibration set
        # X_cal, y_cal = self._build_dataset(self.test_indices)
        # probs = self.model.predict_proba(X_cal)[:,1]

    def evaluate(self, split='test'):
        answer_list = []
        ground_truth = []
        if split == 'test':
            idx = self.test_indices
        else:
            idx = self.ss_indices

        accs, costs = [], []
        for i in tqdm(idx, desc=f"Eval on {split}"):
            cum_cost = 0.0
            # cascade through models
            for model in self.sorted_models:
                pred = None
                max_try_ = 10
                jjj = 0
                while not pred and jjj < max_try_:
                    pred, ans, a = self._get_model_full_answer(model, i)
                    jjj += 1
                # cum_cost += self.models[model]['cost']
                q = self.questions[i].get('question',
                                          self.questions[i].get('input', ''))
                if not pred:
                    cum_cost += 0.
                    prob = 0.
                else:
                    api_cost = self.models[model]['dollar_cost'][i]
                    cum_cost += api_cost['ip_cost'] + api_cost['op_cost'][a]
                    prob = self.model.predict_proba(
                        self._embed([q + " " + pred]))[0, 1]

                if prob >= self.threshold:
                    break

            true = self.get_true_answer(i)
            accs.append(int(ans == true))
            costs.append(cum_cost)
            answer_list.append(ans)
            ground_truth.append(true)

        acc = np.mean(accs)
        avg_cost = np.mean(costs)
        print(f"{split.capitalize()}  Accuracy: {acc:.3%},  Avg Cost: {avg_cost:.5f}")
        return acc, avg_cost, answer_list, ground_truth


# if __name__ == "__main__":
#     for i in np.linspace(0, 1.1, 6):
#         dataset = "aqua"
#         model_fam = ["llama"]
#         model_costs = {
#             'llama/llama_1b_32': 0.01,
#             'llama/llama_3b_32': 0.02,
#             'llama/llama_70b_33': 0.40,
#             'llama/llama_405b_31': 3.00
#         }

#         ensemble = EnsembleFrugalCascade(
#             n_ens=20,
#             dataset_name=dataset,
#             model_families=model_fam,
#             model_costs=model_costs,
#             cost_budget=1.5,
#             calibration_size=50,
#             num_responses=20,
#             threshold=i
#         )

#         ensemble.train_all()
#         acc, cost = ensemble.evaluate_ensemble()
#         res = {}
#         res[cost] = acc

#         Path(f"logs/{dataset}/FrugalGPT").mkdir(parents=True, exist_ok=True)
#         with open(f'logs/{dataset}/FrugalGPT/threshold_{i}.json', 'w+') as fp:
#             json.dump(res, fp)


# if __name__ == "__main__":
#     dataset = 'SVAMP'
#     for i in np.linspace(0, 1.1, 5):
#         cascade = FrugalRegressionCascade(
#             dataset_name   = dataset,
#             model_families = ["llama"],
#             model_costs    = {
#                 'llama/llama_1b_32':   0.01,
#                 'llama/llama_3b_32':   0.02,
#                 'llama/llama_70b_33':  0.40,
#                 'llama/llama_405b_31': 3.00
#             },
#             cost_budget      = 1.0,
#             calibration_size = 50,
#             num_responses    = 20,
#             threshold        = i
#         )
#         cascade.train()
#         acc, cost = cascade.evaluate('test')
#         print(f"Accuracy: {acc:.3%},  Avg Cost: {cost:.5f}")
#         res = {}
#         res[cost] = acc

#         Path(f"logs/{dataset}/FrugalGPT").mkdir(parents=True, exist_ok=True)
#         with open(f'logs/{dataset}/FrugalGPT/threshold_{i}.json', 'w+') as fp:
#             json.dump(res, fp)

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
    # --- ensemble configuration ---
    # for dataset in ['aqua', 'bigbench_causal_judgement', 'bigbench_date', 'bigbench_disambiguationQA',
    #                 'bigbench_formal_fallacies', 'bigbench_geometric_shapes', 'bigbench_movie_recommendation',
    #                 'bigbench_penguins', 'bigbench_ruin_names', 'bigbench_snarks', 'bigbench_sports',
    #                 'bigbench_temporal_sequences', 'commonsenseQA', 'GSM8K', 'SVAMP', 'math_500']:
    for dataset in ['GSM8K', 'SVAMP']:

        test_dataset = 'math_500'

        family = "LLAMA"  # "LLAMA" #"GPT" #"QWEN" #

        if family == 'GPT':
            model_fam = ['gpt']
            model_costs = {
                'openai_gpt/gpt35_turbo': 1.50,
                'openai_gpt/gpt4o_mini': 0.60,
                'openai_gpt/o3_mini': 4.40
            }

        elif family == 'QWEN':
            model_fam = ['qwen']
            model_costs = {
                'qwen/qwen_1b': 0.06,
                'qwen/qwen_32b': 0.20,
                'qwen/qwen_72b': 0.40,
            }

        elif family == 'LLAMA':
            model_fam = ['llama']
            model_costs = {
                'llama/llama_1b_32': 0.01,
                'llama/llama_3b_32': 0.02,
                'llama/llama_70b_33': 0.40,
                'llama/llama_405b_31': 3.00
            }

        else:
            print('error!')

        n_ens = 5
        res = {}
        for thr in np.linspace(0, 1, 5):
            print("Training on ", dataset)
            cost_budget = 1.0
            calibration_size = 50
            num_responses = 5
            threshold = thr

            # initialize storage for each question index
            # we’ll fill these once we know the exact test indices
            ensemble_preds = []
            ensemble_costs = 0

            cascade = FrugalRegressionCascade(
                dataset_name=dataset,
                model_families=model_fam,
                model_costs=model_costs,
                cost_budget=cost_budget,
                calibration_size=calibration_size,
                num_responses=num_responses,
                threshold=threshold
            )
            cascade.train()

            print("Testing on ", test_dataset)
            cascade_ = FrugalRegressionCascade(
                dataset_name=test_dataset,
                model_families=model_fam,
                model_costs=model_costs,
                cost_budget=cost_budget,
                calibration_size=calibration_size,
                num_responses=num_responses,
                threshold=threshold
            )
            cascade_.model = cascade.model

            for run_id in range(n_ens):
                print(f"\n=== Ensemble member {run_id + 1}/{n_ens} ===")
                _, avg_cost, answer_list, ground_truth = cascade_.evaluate()
                ensemble_preds.append(answer_list)
                ensemble_costs += avg_cost
            maj_answer = []
            correct = 0
            corr = []
            for i in range(len(ensemble_preds[0])):
                answers = []
                for j in range(n_ens):
                    answers.append(ensemble_preds[j][i])
                maj_ans, _ = Counter(answers).most_common(1)[0]
                maj_answer.append(maj_ans)
                if maj_ans == ground_truth[i]:
                    correct += 1
                    corr.append(1)
                else:
                    corr.append(0)

            if np.mean(corr) > 0 and np.mean(corr) < 1.0:
                acc, lo, hi = bootstrap_accuracy_scipy(corr)
            else:
                lo = hi = np.mean(corr)

            overall_acc = correct / len(ground_truth)
            print(f"\n=== Ensemble Results ===")
            print(f"Accuracy (majority vote): {overall_acc:.2%}")
            print(f"Average total cost per query: {ensemble_costs:.4f}")
            res[ensemble_costs] = [acc, lo, hi]

        Path(f"logs/{dataset}_{test_dataset}/FrugalGPT_shift/{family}/boot").mkdir(parents=True, exist_ok=True)
        llm_list = cascade._get_model_list(model_fam[0])
        llm_list = fr'{"".join(llm_list)}'.split("/")
        model_string = "-".join(llm_list)
        with open(f'logs/{dataset}_{test_dataset}/FrugalGPT_shift/{family}/boot/{model_string}_budget_{n_ens}.json', 'w+') as fp:
            json.dump(res, fp)
