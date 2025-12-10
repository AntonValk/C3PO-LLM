import os
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

np.random.seed(42)
torch.manual_seed(42)

# class TinyDQN(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2)    # two actions: EXIT (0), QUERY_NEXT (1)
#         )
#     def forward(self, x):
#         return self.net(x)
    
# import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)   # 3 actions
        )
    def forward(self, x):
        return self.net(x)


class FrugalRLFCascade:
    def __init__(self,
                 dataset_name,
                 model_families,
                 model_costs,
                 cost_budget=2.0,
                 calibration_size=50,
                 num_responses=1):
        self.dataset        = dataset_name
        self.model_families = model_families
        self.model_costs    = model_costs
        self.budget         = cost_budget
        self.N_cal          = calibration_size
        self.num_responses  = num_responses

        # embedder
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.encoder   = AutoModel.from_pretrained("distilbert-base-uncased").eval()
        self.device    = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)

        # load LLM outputs & costs
        self.models = self._load_models()
        self.sorted_models = sorted(self.models.keys(),
                                    key=lambda m: self.models[m]['cost'])
        self._load_questions()
        # build DQN
        # dummy = self._embed(["hello world"])[0]
        # self.dqn = TinyDQN(len(dummy)).to(self.device)
        # self.optimizer = optim.Adam(self.dqn.parameters(), lr=1e-3)
        # self.loss_fn = nn.MSELoss()

        bert_dim = self.encoder.config.hidden_size
        self.embedding_dim = 768                           # DistilBERT [CLS] size
        self.extra_feats   = 2                             # consistency & n_samples
        self.num_models    = len(self.sorted_models)       # M
        self.state_dim     = self.embedding_dim + self.extra_feats + self.num_models
        self.dqn = DQN(self.state_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.MSELoss()
        self.gamma = 0.99   


    def _get_model_list(self, family):
        return {
            'llama': ['llama/llama_1b_32', 'llama/llama_3b_32', 'llama/llama_70b_33', 'llama/llama_405b_31'],
            'qwen': ['qwen/qwen_1b', 'qwen/qwen_32b', 'qwen/qwen_72b'],
            'gpt': ['openai_gpt/gpt35_turbo', 'openai_gpt/gpt4o_mini', 'openai_gpt/o3_mini']
        }.get(family, [])

    def _load_models(self):
        models = {}
        for fam in self.model_families:
            for m in self._get_model_list(fam):
                try:
                    with open(f'output/{m}_{self.dataset}_fs_cot_40.json', encoding='utf-8') as f:
                        data = json.load(f)
                    with open(f'costs/{m}_{self.dataset}_fs_cot_40_costs.json', encoding='utf-8') as f:
                        costs = json.load(f)['cost']
                except:
                    with open(f'output/{m}_{self.dataset}_zs_cot_3.json', encoding='utf-8') as f:
                        data = json.load(f)
                    with open(f'costs/{m}_{self.dataset}_fs_cot_3_costs.json', encoding='utf-8') as f:
                        costs = json.load(f)['cost']

                if self.dataset in ['bigbench_date', 'bigbench_disambiguationQA', 'bigbench_geometric_shapes',
                                    'bigbench_movie_recommendation', 'bigbench_penguins', 'bigbench_ruin_names',
                                    'bigbench_snarks', 'bigbench_temporal_sequences']:

                    models[m] = {
                        'responses': [[z.replace('(', '').replace(')', '') if z is not None else z for z in r['decoded_answers']] for r in data['responses']],
                        'full': [r['response']['responses'] if 'responses' in r['response'] else []
                                 for r in data['responses']],
                        'dollar_cost': costs,
                        'cost': self.model_costs[m] #.get(m, float(m.split('_')[-1]))
                    }
                else:
                    models[m] = {
                        'responses': [r['decoded_answers'] for r in data['responses']],
                        'full': [r['response']['responses'] if 'responses' in r['response'] else []
                                 for r in data['responses']],
                        'dollar_cost': costs,
                        'cost': self.model_costs[m]  # .get(m, float(m.split('_')[-1]))
                    }
        return models

    def _load_questions(self):
        first = self.sorted_models[0]
        with open(f'output/{first}_{self.dataset}_fs_cot_40.json', encoding='utf-8') as f:
            data = json.load(f)
        self.questions = [r['data_entry'] for r in data['responses']]

    def get_true_answer(self, q_idx):
        # data_entry contains 'answer' field
        true_answer = self.questions[q_idx]['answer']
        if self.dataset in ['bigbench_date', 'bigbench_disambiguationQA', 'bigbench_geometric_shapes',
                            'bigbench_movie_recommendation', 'bigbench_penguins', 'bigbench_ruin_names',
                            'bigbench_snarks', 'bigbench_temporal_sequences']:
            true_answer = true_answer.replace('(', '').replace(')', '')

        return true_answer

    def _get_model_full_answer(self, model, i):
        try:
            choices = self.models[model]['full'][i]
            idx = np.random.randint(len(choices))
            msg = choices[idx]['message']
            ans = self.models[model]['responses'][i][idx]
        except:
            msg, ans, idx = "", None, None
        return msg, ans, idx

    def _embed(self, texts):
        inp = self.tokenizer(texts, return_tensors='pt',
                             padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            out = self.encoder(**inp)
        return out.last_hidden_state[:,0,:].cpu().numpy()

    def train(self, epochs: int = 1):
        """
        Train a small 2-layer DQN to decide among:
        0 = EXIT,
        1 = MOVE TO NEXT MODEL,
        2 = RE-QUERY CURRENT MODEL
        under the cost budget.
        """
        bert_dim = self.encoder.config.hidden_size
        for ep in range(epochs):
            epoch_losses = []
            loss = torch.tensor(0.0, device=self.device)   # define it up front
            # iterate over your self-supervision split
            for q_idx in range(0, 2 * self.N_cal):
                remaining_budget = self.budget
                model_ptr = 0
                sample_history = []
                done = False

                true_label = self.get_true_answer(q_idx)

                while not done:
                    model_name = self.sorted_models[model_ptr]
                    model_cost = self.models[model_name]['cost']

                    # draw one new sample from current model
                    msg, ans, _ = self._get_model_full_answer(model_name, q_idx)
                    remaining_budget -= model_cost
                    sample_history.append(ans)

                    # compute self-consistency features
                    mode_count = Counter(sample_history).most_common(1)[0][1]
                    consistency = mode_count / len(sample_history)
                    n_samples = len(sample_history)

                    # build state vector = [CLS_emb, consistency, n_samples, model_id]
                    model_one_hot = np.zeros(len(self.sorted_models), dtype=float)
                    model_one_hot[model_ptr] = 1.0
                    emb = self._embed([msg])[0]  # shape (bert_dim,)
                    state = torch.tensor(
                        np.concatenate([emb, [consistency, n_samples], model_one_hot]),
                        dtype=torch.float32,
                        device=self.device
                    ).unsqueeze(0)
                    # shape (1, state_dim)

                    # get Q-values for the 3 actions
                    qvals = self.dqn(state)  # (1, 3)

                    # mask out illegal actions if budget insufficient
                    if remaining_budget < 0:
                        # only EXIT is legal
                        qvals[0,1:] = -1e9

                    # pick greedy action (you can inject e-greedy here if you like)
                    action = qvals.argmax(dim=1).item()

                    # prepare a bootstrap target copy
                    target_q = qvals.clone().detach()

                    if action == 0:  # EXIT
                        # choose final answer = mode(sample_history)
                        final_ans = Counter(sample_history).most_common(1)[0][0]
                        reward = 1.0 if final_ans == true_label else 0.0

                        # immediate-only: override Q_target at EXIT slot
                        target_q[0, 0] = reward

                        # one-step MSE update
                        loss = self.loss_fn(qvals, target_q)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        done = True

                    elif action == 1:  # MOVE TO NEXT MODEL
                        # reset sampling history, advance to next model
                        model_ptr += 1
                        sample_history = []
                        # if we have run out of models, force EXIT next loop
                        if model_ptr >= len(self.sorted_models):
                            done = True
                        # no immediate update (could add small negative cost penalty)

                    else:  # action == 2, RE-QUERY CURRENT MODEL
                        # simply loop again, drawing another sample
                        # no immediate update

                        # if you like, you could give a small negative reward here to
                        # discourage endless re-querying: e.g., reward = -0.01
                        pass

                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"[Epoch {ep+1}/{epochs}] avg training loss: {avg_loss:.4f}")

    def evaluate(self, split='test'):
        idx =  np.arange(2 * self.N_cal, len(self.questions)) if split == 'test' else  range(0, 2 * self.N_cal)
        all_preds, all_costs = [], []
    
        last_model_idx = len(self.sorted_models) - 1

        for i in tqdm(idx, desc=f"Eval on {split}"):
            cum_cost    = 0.0
            model_ptr   = 0
            history     = []
            n_drawn     = 0
            consistency = 0.0
    
            # we'll track the last most_common_ans so that if we break on cost, we still have one
            most_common_ans = None
    
            done = False
            while not done:
                model_name = self.sorted_models[model_ptr]
    
                # 1) Sample one new answer
                msg, ans, sample_idx = self._get_model_full_answer(model_name, i)
                history.append(ans)
                n_drawn += 1
    
                # 2) Accrue cost
                if not sample_idx:
                    cum_cost += 0.
                else:
                    cost_info = self.models[model_name]['dollar_cost'][i]
                    step_cost = cost_info['ip_cost'] + cost_info['op_cost'][sample_idx]
                    cum_cost += step_cost
    
                # Immediately exit if we've exceeded budget
                if cum_cost > self.budget:
                    # use current most common answer (or fallback to this latest ans)
                    freq = Counter(history)
                    most_common_ans, _ = freq.most_common(1)[0]
                    break
    
                # 3) Update consistency
                freq = Counter(history)
                most_common_ans, count = freq.most_common(1)[0]
                consistency = count / n_drawn
    
                # 4) Forced exit caps
                if model_ptr < last_model_idx and n_drawn >= 10:
                    # escalate
                    history, n_drawn, consistency = [], 0, 0.0
                    model_ptr += 1
                    continue
                if model_ptr == last_model_idx and n_drawn >= 5:
                    break
    
                # 5) Build state for DQN
                q   = self.questions[i].get('question', self.questions[i].get('input',''))
                emb = self._embed([f"{q} {most_common_ans}"])[0]
                one_hot = np.zeros(len(self.sorted_models), dtype=float)
                one_hot[model_ptr] = 1.0
                state = np.concatenate([emb, [consistency, n_drawn], one_hot])
                state_t = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
    
                # 6) Ask DQN for action: 0=re-query, 1=escalate, 2=stop
                with torch.no_grad():
                    q_vals = self.dqn(state_t)      # shape (1,3)
                action = q_vals.argmax(dim=1).item()
    
                if action == 0:
                    continue
                elif action == 1:
                    history, n_drawn, consistency = [], 0, 0.0
                    model_ptr += 1
                    if model_ptr > last_model_idx:
                        break
                else:  # action == 2
                    break
    
            # Record results
            all_preds.append(most_common_ans if most_common_ans is not None else ans)
            all_costs.append(cum_cost)
    
        truths = [self.get_true_answer(i) for i in idx]
        acc = np.mean([p == t for p, t in zip(all_preds, truths)])
        avg_cost = np.mean(all_costs)
        print(f"{split}  Accuracy: {acc:.3%},  Avg Cost: {avg_cost:.5f}")
        return acc, avg_cost, [p == t for p, t in zip(all_preds, truths)]


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
    

if __name__=="__main__":
    # for dataset in ['aqua', 'bigbench_causal_judgement', 'bigbench_date', 'bigbench_disambiguationQA',
    #                 'bigbench_formal_fallacies', 'bigbench_geometric_shapes', 'bigbench_movie_recommendation',
    #                 'bigbench_penguins', 'bigbench_ruin_names', 'bigbench_snarks', 'bigbench_sports',
    #                 'bigbench_temporal_sequences', 'commonsenseQA', 'GSM8K', 'SVAMP', 'math_500']:
    for dataset in ['math_500']:
        family = "LLAMA" #"QWEN" #"GPT" #

        if family == 'GPT':
            model_fam = ['gpt']
            model_costs = {
                'openai_gpt/gpt35_turbo': 1.50,
                'openai_gpt/gpt4o_mini': 0.60,
                'openai_gpt/o3_mini': 4.40
            }
            budget_range = np.linspace(0, 7, 7)

        elif family == 'QWEN':
            model_fam = ['qwen']
            model_costs = {
                'qwen/qwen_1b': 0.06,
                'qwen/qwen_32b': 0.20,
                'qwen/qwen_72b': 0.40,
            }
            budget_range = np.linspace(0, 0.75, 6)

        elif family == 'LLAMA':
            model_fam = ['llama']
            model_costs = {
                'llama/llama_1b_32': 0.01,
                'llama/llama_3b_32': 0.02,
                'llama/llama_70b_33': 0.40,
                'llama/llama_405b_31': 3.00
            }
            budget_range = np.linspace(0, 5, 6)

        else:
            print('error!')

        print("Now running", dataset)

        res = {}
        num_responses = 5
        for thresh in budget_range:
          cascade = FrugalRLFCascade(
              dataset_name   = dataset,
              model_families = model_fam,
              model_costs=model_costs,
              cost_budget      = thresh,
              calibration_size = 50,
              num_responses    = num_responses
          )
          cascade.train(epochs=3)
          acc, true_dollar_cost, corr_ = cascade.evaluate()
          if acc > 0 and acc < 1.0:
              acc, lo, hi = bootstrap_accuracy_scipy(corr_)
          else:
              lo = hi = acc

          res[true_dollar_cost] = [acc, lo, hi]

        Path(f"logs/{dataset}/TREACLE/{family}/boot").mkdir(parents=True, exist_ok=True)
        llm_list = cascade._get_model_list(model_fam[0])
        llm_list = fr'{"".join(llm_list)}'.split("/")
        model_string = "-".join(llm_list)

        with open(f'logs/{dataset}/TREACLE/{family}/boot/{model_string}_budget_{num_responses}.json', 'w+') as fp:
            json.dump(res, fp)

    #Path(f"logs/{dataset}/TREACLE/").mkdir(parents=True, exist_ok=True)
    #with open(f'logs/{dataset}/TREACLE/budget_{thresh}.json', 'w+') as fp:
    #    json.dump(res, fp)

