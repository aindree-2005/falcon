import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
import os
import math

warnings.filterwarnings("ignore")

# set seed for reproducible results
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# common global parameters
FED_ROUNDS = 25
LOCAL_EPOCHS = 5
BATCH_SIZE = 256
BASE_LR = 1.5e-3
WEIGHT_DECAY = 1e-4

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# path logic to find input data
primary_path = "../data/eicu/eicu_dafl_ready.csv"
fallback_path = "/kaggle/input/dafl-falcon/eicu_dafl_ready.csv"
dataset_path = primary_path if os.path.exists(primary_path) else (fallback_path if os.path.exists(fallback_path) else "data/eicu/eicu_dafl_ready.csv")

print(f"loading csv from {dataset_path}...")
df = pd.read_csv(dataset_path)
print(f"data shape: {df.shape}")

# list of hospitals and metadata column filtering
selected_hospitals = [167, 420, 199, 458, 252, 165, 148, 281, 449, 283]
target_hospitals = [167, 199, 252, 420, 458]
feature_cols = df.columns.difference(["patientunitstayid", "hospitalid", "death", "ventilation", "sepsis"])

# partition client data into dictionary
clients_raw = {}
for h in selected_hospitals:
    hospital_df = df[df["hospitalid"] == h]
    X = hospital_df[feature_cols].values
    y = hospital_df["death"].values
    clients_raw[h] = (X, y)

print(f"split data into {len(clients_raw)} hospital clients")

# evaluation function to compute metrics and search best threshold
def evaluate_comprehensive(model, X, y, batch_size=512):
    model.eval()
    all_logits = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            logits = model(bx).squeeze().cpu().numpy()
            if np.ndim(logits) == 0:
                logits = np.array([logits])
            all_logits.append(logits)

    logits = np.concatenate(all_logits)
    probs = 1 / (1 + np.exp(-logits))

    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
    auprc = average_precision_score(y, probs) if len(np.unique(y)) > 1 else 0.0

    precisions, recalls, thresholds = precision_recall_curve(y, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    preds = (probs >= best_threshold).astype(int)
    f1 = f1_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)

    return {
        "AUROC": float(auc),
        "AUPRC": float(auprc),
        "F1": float(f1),
        "Precision": float(prec),
        "Recall": float(rec),
        "Best_Threshold": float(best_threshold)
    }

# bootstrap confidence intervals on full test set
def evaluate_point_and_bootstrap_ci(model, X, y, fixed_threshold, batch_size=512, n_boot=100, seed=SEED):
    model.eval()
    all_logits = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            logits = model(bx).squeeze().cpu().numpy()
            if np.ndim(logits) == 0:
                logits = np.array([logits])
            all_logits.append(logits)

    logits = np.concatenate(all_logits)
    probs = 1 / (1 + np.exp(-logits))

    # evaluate complete dataset point estimate
    point = {}
    point["AUROC"] = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.5
    point["AUPRC"] = float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else 0.0

    preds_full = (probs >= fixed_threshold).astype(int)
    point["F1"] = float(f1_score(y, preds_full, zero_division=0))
    point["Precision"] = float(precision_score(y, preds_full, zero_division=0))
    point["Recall"] = float(recall_score(y, preds_full, zero_division=0))

    # run bootstrap resampling loops
    rng = np.random.RandomState(seed)
    boot = {'AUROC': [], 'AUPRC': [], 'F1': [], 'Precision': [], 'Recall': []}

    for b in range(n_boot):
        boot_seed = rng.randint(0, 2**31 - 1)
        y_b, p_b = resample(y, probs, replace=True, random_state=boot_seed, stratify=y)

        if len(np.unique(y_b)) < 2:
            continue

        boot['AUROC'].append(roc_auc_score(y_b, p_b))
        boot['AUPRC'].append(average_precision_score(y_b, p_b))

        preds_b = (p_b >= fixed_threshold).astype(int)
        boot['F1'].append(f1_score(y_b, preds_b, zero_division=0))
        boot['Precision'].append(precision_score(y_b, preds_b, zero_division=0))
        boot['Recall'].append(recall_score(y_b, preds_b, zero_division=0))

    summary = {}
    for metric in point.keys():
        values = boot[metric]
        summary[metric] = {
            'point': point[metric],
            'lower_ci': float(np.percentile(values, 2.5)) if len(values) > 0 else point[metric],
            'upper_ci': float(np.percentile(values, 97.5)) if len(values) > 0 else point[metric]
        }
    return summary

# simple mortality prediction neural net
class MortalityModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.base(x))

# helper function to return dataloader
def get_loader(X, y, batch_size=BATCH_SIZE):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), batch_size=batch_size, shuffle=True)

# helper function to derive decaying learning rate
def get_lr(round_idx):
    eta_min = BASE_LR * 0.1
    return eta_min + 0.5 * (BASE_LR - eta_min) * (1 + math.cos(math.pi * round_idx / FED_ROUNDS))

# local training routine for fedavg
def train_local_fedavg(model, X, y, current_round, epochs=LOCAL_EPOCHS):
    model.train()
    loader = get_loader(X, y)
    y_t = torch.tensor(y, dtype=torch.float32)
    pos = y_t.sum()
    neg = len(y_t) - pos
    pos_weight = neg / (pos + 1e-8)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=get_lr(current_round), weight_decay=WEIGHT_DECAY)

    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx).squeeze(), by)
            loss.backward()
            optimizer.step()

    return model.state_dict()

# local training routine for fedprox
def train_local_fedprox(model, X, y, global_weights, current_round, mu=1e-4, epochs=LOCAL_EPOCHS):
    model.train()
    loader = get_loader(X, y)
    y_t = torch.tensor(y, dtype=torch.float32)
    pos = y_t.sum()
    neg = len(y_t) - pos
    pos_weight = neg / (pos + 1e-8)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=get_lr(current_round), weight_decay=WEIGHT_DECAY)
    global_params = {k: v.clone().detach().to(device) for k, v in global_weights.items()}

    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx).squeeze(), by)

            proximal_term = sum(torch.sum((param - global_params[name]) ** 2) for name, param in model.named_parameters() if name in global_params)
            loss = loss + (mu / 2.0) * proximal_term

            loss.backward()
            optimizer.step()

    return model.state_dict()

# local training routine for feddyn
def train_local_feddyn(model, X, y, client_grad_state, prev_global_weights, current_round, alpha_coeff=0.01, epochs=LOCAL_EPOCHS):
    model.train()
    loader = get_loader(X, y)
    y_t = torch.tensor(y, dtype=torch.float32)
    pos = y_t.sum()
    neg = len(y_t) - pos
    pos_weight = neg / (pos + 1e-8)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=get_lr(current_round), weight_decay=WEIGHT_DECAY)
    theta_prev_round = {k: v.clone().detach().to(device) for k, v in prev_global_weights.items()}

    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx).squeeze(), by)

            linear_term = sum(torch.sum(client_grad_state[name].to(device) * param) for name, param in model.named_parameters() if name in client_grad_state)
            quad_term = sum(torch.sum((param - theta_prev_round[name]) ** 2) for name, param in model.named_parameters() if name in theta_prev_round)

            loss = loss - linear_term + (alpha_coeff / 2.0) * quad_term
            loss.backward()
            optimizer.step()

    # compute dynamic local gradient state update
    new_grad_state = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            old = client_grad_state.get(name, torch.zeros_like(param).to(device))
            new_grad_state[name] = (old - alpha_coeff * (param.detach() - theta_prev_round[name])).cpu()

    return model.state_dict(), new_grad_state

# main benchmark training loops
algorithms = ['FedAvg', 'FedProx', 'FedDyn']
all_benchmark_records = []

for algo in algorithms:
    print(f"\n--- Running algorithm: {algo} ---")

    for target_hospital in target_hospitals:
        print(f"\ntarget hospital fold: {target_hospital} ({algo})")

        X_target_raw, y_target = clients_raw[target_hospital]

        # 70/30 train/test split on target hospital
        X_val_raw, X_test_raw, y_val, y_test = train_test_split(
            X_target_raw, y_target, test_size=0.3, random_state=SEED, stratify=y_target
        )

        # scale features using only source hospital training data
        source_hospital_ids = [h for h in clients_raw.keys() if h != target_hospital]
        X_source_concat = np.concatenate([clients_raw[h][0] for h in source_hospital_ids], axis=0)

        source_scaler = StandardScaler()
        source_scaler.fit(X_source_concat)

        source_clients = {}
        for h in source_hospital_ids:
            X_raw, y_raw = clients_raw[h]
            source_clients[h] = (source_scaler.transform(X_raw), y_raw)

        X_val = source_scaler.transform(X_val_raw)
        X_test = source_scaler.transform(X_test_raw)

        input_dim = X_val.shape[1]
        global_model = MortalityModel(input_dim).to(device)

        # initialize client and server tracking for feddyn
        feddyn_client_grads = {h: {name: torch.zeros_like(param).cpu() for name, param in global_model.named_parameters()} for h in source_clients}
        feddyn_server_h = {name: torch.zeros_like(param).to(device) for name, param in global_model.named_parameters()}
        FEDDYN_ALPHA = 0.01

        best_val_auprc = -1.0
        best_weights = None
        locked_test_threshold = 0.5

        for r in range(FED_ROUNDS):
            prev_global_state = copy.deepcopy(global_model.state_dict())

            client_weights = []
            client_sizes = []

            for hospital, (X, y) in source_clients.items():
                local_model = MortalityModel(input_dim).to(device)
                local_model.load_state_dict(prev_global_state)

                if algo == 'FedAvg':
                    weights = train_local_fedavg(local_model, X, y, current_round=r)
                elif algo == 'FedProx':
                    weights = train_local_fedprox(local_model, X, y, global_weights=prev_global_state, current_round=r, mu=1e-4)
                elif algo == 'FedDyn':
                    weights, feddyn_client_grads[hospital] = train_local_feddyn(
                        local_model, X, y,
                        client_grad_state=feddyn_client_grads[hospital],
                        prev_global_weights=prev_global_state,
                        current_round=r,
                        alpha_coeff=FEDDYN_ALPHA
                    )

                client_weights.append(weights)
                client_sizes.append(len(X))

            # aggregate local weights based on baseline algorithm
            if algo in ('FedAvg', 'FedProx'):
                total_samples = sum(client_sizes)
                sample_weights = [s / total_samples for s in client_sizes]

                global_dict = global_model.state_dict()
                for key in global_dict.keys():
                    tensors = [w[key] for w in client_weights]
                    if tensors[0].dtype.is_floating_point:
                        global_dict[key] = sum(tensors[i].float().to(device) * sample_weights[i] for i in range(len(tensors)))
                    else:
                        global_dict[key] = tensors[0].to(device)

                global_model.load_state_dict(global_dict)

            elif algo == 'FedDyn':
                m = len(client_weights)
                new_global_dict = {}
                global_dict_keys = global_model.state_dict().keys()

                for key in global_dict_keys:
                    tensors = [w[key].float().to(device) for w in client_weights]

                    if not tensors[0].dtype.is_floating_point:
                        new_global_dict[key] = tensors[0]
                        continue

                    avg_theta = sum(tensors) / m
                    delta_sum = sum(t - prev_global_state[key].to(device) for t in tensors)

                    feddyn_server_h[key] = feddyn_server_h[key] - (FEDDYN_ALPHA / m) * delta_sum
                    new_global_dict[key] = avg_theta - (1.0 / FEDDYN_ALPHA) * feddyn_server_h[key]

                global_model.load_state_dict(new_global_dict)

            # check validation performance
            val_metrics = evaluate_comprehensive(global_model, X_val, y_val)
            print(f"round {r:2d} | val auroc: {val_metrics['AUROC']:.4f} | val auprc: {val_metrics['AUPRC']:.4f} | val f1: {val_metrics['F1']:.4f}")

            all_benchmark_records.append({
                "Algorithm": algo,
                "Target_Hospital": target_hospital,
                "Round": r,
                "Type": "Validation",
                **val_metrics
            })

            # track best model weights on validation set
            if val_metrics["AUPRC"] > best_val_auprc:
                best_val_auprc = val_metrics["AUPRC"]
                best_weights = copy.deepcopy(global_model.state_dict())
                locked_test_threshold = val_metrics["Best_Threshold"]

        # evaluate best weights on held-out test split
        global_model.load_state_dict(best_weights)

        print(f"evaluating test metrics for target hospital {target_hospital}...")
        test_metrics = evaluate_point_and_bootstrap_ci(
            global_model, X_test, y_test, fixed_threshold=locked_test_threshold, n_boot=100, seed=SEED
        )

        print(f"--> test auroc:     {test_metrics['AUROC']['point']:.4f} (95% ci: {test_metrics['AUROC']['lower_ci']:.4f} - {test_metrics['AUROC']['upper_ci']:.4f})")
        print(f"--> test auprc:     {test_metrics['AUPRC']['point']:.4f} (95% ci: {test_metrics['AUPRC']['lower_ci']:.4f} - {test_metrics['AUPRC']['upper_ci']:.4f})")
        print(f"--> test precision: {test_metrics['Precision']['point']:.4f} (95% ci: {test_metrics['Precision']['lower_ci']:.4f} - {test_metrics['Precision']['upper_ci']:.4f})")
        print(f"--> test recall:    {test_metrics['Recall']['point']:.4f} (95% ci: {test_metrics['Recall']['lower_ci']:.4f} - {test_metrics['Recall']['upper_ci']:.4f})")
        print(f"--> test f1:        {test_metrics['F1']['point']:.4f} (95% ci: {test_metrics['F1']['lower_ci']:.4f} - {test_metrics['F1']['upper_ci']:.4f})\n")

        all_benchmark_records.append({
            "Algorithm": algo,
            "Target_Hospital": target_hospital,
            "Round": "FINAL_TEST",
            "Type": "Test",
            "AUROC": test_metrics['AUROC']['point'],
            "AUROC_Lower": test_metrics['AUROC']['lower_ci'],
            "AUROC_Upper": test_metrics['AUROC']['upper_ci'],
            "AUPRC": test_metrics['AUPRC']['point'],
            "AUPRC_Lower": test_metrics['AUPRC']['lower_ci'],
            "AUPRC_Upper": test_metrics['AUPRC']['upper_ci'],
            "Precision": test_metrics['Precision']['point'],
            "Precision_Lower": test_metrics['Precision']['lower_ci'],
            "Precision_Upper": test_metrics['Precision']['upper_ci'],
            "Recall": test_metrics['Recall']['point'],
            "Recall_Lower": test_metrics['Recall']['lower_ci'],
            "Recall_Upper": test_metrics['Recall']['upper_ci'],
            "F1": test_metrics['F1']['point'],
            "F1_Lower": test_metrics['F1']['lower_ci'],
            "F1_Upper": test_metrics['F1']['upper_ci'],
            "Best_Threshold": locked_test_threshold,
        })

# export final metrics
df_results = pd.DataFrame(all_benchmark_records)
df_results.to_csv("federated_benchmark_results.csv", index=False)
print("\nfinished training! saved results to 'federated_benchmark_results.csv'")