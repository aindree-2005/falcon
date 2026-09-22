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

# set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# common global parameters
FED_ROUNDS = 25
LOCAL_EPOCHS = 5
BATCH_SIZE = 256
BASE_LR = 1.5e-3
WEIGHT_DECAY = 1e-4

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# file paths check
primary_path = "../data/eicu/eicu_dafl_ready.csv"
fallback_path = "/kaggle/input/dafl-falcon/eicu_dafl_ready.csv"
dataset_path = primary_path if os.path.exists(primary_path) else (fallback_path if os.path.exists(fallback_path) else "data/eicu/eicu_dafl_ready.csv")

print(f"loading data from {dataset_path}...")
df = pd.read_csv(dataset_path)
print(f"data shape: {df.shape}")

# setup hospital splits and features
selected_hospitals = [167, 420, 199, 458, 252, 165, 148, 281, 449, 283]
target_hospitals = [167, 199, 252, 420, 458]
feature_cols = df.columns.difference(["patientunitstayid", "hospitalid", "death", "ventilation", "sepsis"])

# partition clients into dictionary
clients_raw = {}
for h in selected_hospitals:
    hospital_df = df[df["hospitalid"] == h]
    X = hospital_df[feature_cols].values
    y = hospital_df["death"].values
    clients_raw[h] = (X, y)

print(f"created {len(clients_raw)} client partitions")

# evaluation function to compute metrics and threshold
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

# bootstrap evaluation on test set
def evaluate_bootstrapped(model, X, y, fixed_threshold, batch_size=512, seeds=100):
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
    
    results = {'AUROC': [], 'AUPRC': [], 'F1': [], 'Precision': [], 'Recall': []}
    
    for seed in range(seeds):
        y_b, p_b = resample(y, probs, replace=True, random_state=seed)
        if len(np.unique(y_b)) < 2:
            continue
            
        auc = roc_auc_score(y_b, p_b)
        auprc = average_precision_score(y_b, p_b)
        
        preds = (p_b >= fixed_threshold).astype(int)
        f1 = f1_score(y_b, preds, zero_division=0)
        prec = precision_score(y_b, preds, zero_division=0)
        rec = recall_score(y_b, preds, zero_division=0)
        
        results['AUROC'].append(auc)
        results['AUPRC'].append(auprc)
        results['F1'].append(f1)
        results['Precision'].append(prec)
        results['Recall'].append(rec)
        
    summary = {}
    for metric, values in results.items():
        summary[metric] = {
            'mean': np.mean(values),
            'lower_ci': np.percentile(values, 2.5),
            'upper_ci': np.percentile(values, 97.5)
        }
    return summary

# mortality classification model
class MortalityModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.base(x))

# helper function to compute cosine learning rate step
def get_lr(round_idx):
    eta_min = BASE_LR * 0.1
    return eta_min + 0.5 * (BASE_LR - eta_min) * (1 + math.cos(math.pi * round_idx / FED_ROUNDS))

# local train step using AdamW + Cosine LR Schedule
def train_local_sgd(model, X, y, current_round, epochs=LOCAL_EPOCHS):
    model.train()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
    y_t = torch.tensor(y, dtype=torch.float32)
    pos_weight = (len(y_t) - y_t.sum()) / (y_t.sum() + 1e-8)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=get_lr(current_round), weight_decay=WEIGHT_DECAY)

    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx).squeeze(), by)
            loss.backward()
            optimizer.step()

    return {k: v.cpu() for k, v in model.state_dict().items()}

# FedREP mask generation
def get_consensus_mask(client_updates, sparsity_ratio=0.1):
    masks = {}
    keys = list(client_updates.values())[0].keys()
    
    for k in keys:
        if 'weight' not in k and 'bias' not in k:
            masks[k] = torch.ones_like(list(client_updates.values())[0][k])
            continue
            
        union_mask = torch.zeros_like(list(client_updates.values())[0][k])
        for h, update in client_updates.items():
            tensor = update[k]
            num_elements = tensor.numel()
            if num_elements == 0: continue
                
            k_top = max(1, int(num_elements * sparsity_ratio))
            _, indices = torch.topk(torch.abs(tensor.flatten()), k_top)
            local_mask = torch.zeros_like(tensor.flatten())
            local_mask[indices] = 1.0
            union_mask = torch.max(union_mask, local_mask.view_as(tensor))
            
        masks[k] = union_mask
    return masks

# FedREP trimmed mean aggregation
def trimmed_mean(tensors_list, trim_ratio=0.1):
    keys = tensors_list[0].keys()
    num_clients = len(tensors_list)
    trim_count = int(num_clients * trim_ratio)
    
    aggregated = {}
    for k in keys:
        if not tensors_list[0][k].dtype.is_floating_point:
            aggregated[k] = tensors_list[0][k]
            continue
            
        stacked = torch.stack([t[k] for t in tensors_list])
        if trim_count > 0 and num_clients > 2 * trim_count:
            sorted_tensor, _ = torch.sort(stacked, dim=0)
            trimmed_tensor = sorted_tensor[trim_count:-trim_count]
            aggregated[k] = torch.mean(trimmed_tensor, dim=0)
        else:
            aggregated[k] = torch.mean(stacked, dim=0)
            
    return aggregated

# main training loop for FEDPER and FedREP
algorithms = ['FEDPER', 'FedREP']
all_benchmark_records = []

for algo in algorithms:
    print(f"\n--- Running algorithm: {algo} ---")
    
    for target_hospital in target_hospitals:
        print(f"\ntarget hospital fold: {target_hospital} ({algo})")

        X_tgt_raw, y_tgt = clients_raw[target_hospital]

        # Personalized baseline evaluation split: 60% Train, 20% Val, 20% Test
        X_train_tgt, X_temp, y_train_tgt, y_temp = train_test_split(X_tgt_raw, y_tgt, test_size=0.4, random_state=SEED, stratify=y_tgt)
        X_val_tgt, X_test_tgt, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)
        
        tgt_scaler = StandardScaler()
        X_train_tgt = tgt_scaler.fit_transform(X_train_tgt)
        X_val = tgt_scaler.transform(X_val_tgt)
        X_test = tgt_scaler.transform(X_test_tgt)

        training_clients = {target_hospital: (X_train_tgt, y_train_tgt)}
        for k, (X_raw, y_raw) in clients_raw.items():
            if k != target_hospital:
                training_clients[k] = (StandardScaler().fit_transform(X_raw), y_raw)

        input_dim = X_val.shape[1]
        
        global_model = MortalityModel(input_dim)
        
        if algo == 'FEDPER':
            global_base_state = {k: v for k, v in global_model.state_dict().items() if 'base' in k}
            local_heads = {h: {k: v for k, v in MortalityModel(input_dim).state_dict().items() if 'head' in k} for h in training_clients.keys()}
        
        elif algo == 'FedREP':
            global_state = global_model.state_dict()
            u_k = {h: {k: torch.zeros_like(v) for k, v in global_state.items()} for h in training_clients.keys()}

        best_val_auprc = -1.0
        best_weights_tgt = None
        locked_test_threshold = 0.5

        for r in range(FED_ROUNDS):
            if algo == 'FEDPER':
                client_base_weights = []
                client_sizes = []
                
                for hospital, (X, y) in training_clients.items():
                    local_model = MortalityModel(input_dim).to(device)
                    local_model.load_state_dict({**global_base_state, **local_heads[hospital]})
                    
                    trained_state = train_local_sgd(local_model, X, y, current_round=r)
                    
                    local_heads[hospital] = {k: v for k, v in trained_state.items() if 'head' in k}
                    client_base_weights.append({k: v for k, v in trained_state.items() if 'base' in k})
                    client_sizes.append(len(X))
                
                total_samples = sum(client_sizes)
                for key in global_base_state.keys():
                    tensors = [w[key] for w in client_base_weights]
                    if tensors[0].dtype.is_floating_point:
                        global_base_state[key] = sum(tensors[i].float() * (client_sizes[i] / total_samples) for i in range(len(tensors)))
                    else:
                        global_base_state[key] = tensors[0]
                
                eval_model = MortalityModel(input_dim).to(device)
                eval_model.load_state_dict({**global_base_state, **local_heads[target_hospital]})
                
            elif algo == 'FedREP':
                client_updates = {}
                
                for hospital, (X, y) in training_clients.items():
                    local_model = MortalityModel(input_dim).to(device)
                    local_model.load_state_dict(global_state)
                    
                    trained_state = train_local_sgd(local_model, X, y, current_round=r)
                    
                    g_k = {}
                    for k in trained_state.keys():
                        g_k[k] = u_k[hospital][k] + (global_state[k] - trained_state[k])
                    client_updates[hospital] = g_k
                
                I_t = get_consensus_mask(client_updates, sparsity_ratio=0.1)
                
                for hospital in training_clients.keys():
                    for k in client_updates[hospital].keys():
                        mask = I_t[k]
                        masked_g = client_updates[hospital][k] * mask
                        u_k[hospital][k] = client_updates[hospital][k] - masked_g
                        client_updates[hospital][k] = masked_g
                
                aggregated_g = trimmed_mean(list(client_updates.values()), trim_ratio=0.1)
                
                for k in global_state.keys():
                    global_state[k] -= aggregated_g[k]
                    
                eval_model = MortalityModel(input_dim).to(device)
                eval_model.load_state_dict(global_state)

            val_metrics = evaluate_comprehensive(eval_model, X_val, y_val)
            print(f"round {r:2d} | val auroc: {val_metrics['AUROC']:.4f} | val auprc: {val_metrics['AUPRC']:.4f} | val f1: {val_metrics['F1']:.4f}")

            all_benchmark_records.append({
                "Algorithm": algo,
                "Target_Hospital": target_hospital,
                "Round": r,
                "Type": "Validation",
                **val_metrics
            })

            if val_metrics["AUPRC"] > best_val_auprc:
                best_val_auprc = val_metrics["AUPRC"]
                best_weights_tgt = copy.deepcopy(eval_model.state_dict())
                locked_test_threshold = val_metrics["Best_Threshold"] 

        eval_model.load_state_dict(best_weights_tgt)
        
        print(f"evaluating test set for target hospital {target_hospital}...")
        test_metrics = evaluate_bootstrapped(eval_model, X_test, y_test, fixed_threshold=locked_test_threshold, seeds=100)
        
        print(f"--> test auroc:     {test_metrics['AUROC']['mean']:.4f} (95% ci: {test_metrics['AUROC']['lower_ci']:.4f} - {test_metrics['AUROC']['upper_ci']:.4f})")
        print(f"--> test auprc:     {test_metrics['AUPRC']['mean']:.4f} (95% ci: {test_metrics['AUPRC']['lower_ci']:.4f} - {test_metrics['AUPRC']['upper_ci']:.4f})")
        print(f"--> test precision: {test_metrics['Precision']['mean']:.4f} (95% ci: {test_metrics['Precision']['lower_ci']:.4f} - {test_metrics['Precision']['upper_ci']:.4f})")
        print(f"--> test recall:    {test_metrics['Recall']['mean']:.4f} (95% ci: {test_metrics['Recall']['lower_ci']:.4f} - {test_metrics['Recall']['upper_ci']:.4f})")
        print(f"--> test f1:        {test_metrics['F1']['mean']:.4f} (95% ci: {test_metrics['F1']['lower_ci']:.4f} - {test_metrics['F1']['upper_ci']:.4f})\n")

        all_benchmark_records.append({
            "Algorithm": algo, 
            "Target_Hospital": target_hospital, 
            "Round": "FINAL_TEST", 
            "Type": "Test",
            "AUROC_Mean": test_metrics['AUROC']['mean'], 
            "AUROC_Lower": test_metrics['AUROC']['lower_ci'], 
            "AUROC_Upper": test_metrics['AUROC']['upper_ci'],
            "AUPRC_Mean": test_metrics['AUPRC']['mean'], 
            "AUPRC_Lower": test_metrics['AUPRC']['lower_ci'], 
            "AUPRC_Upper": test_metrics['AUPRC']['upper_ci'],
            "Precision_Mean": test_metrics['Precision']['mean'], 
            "Precision_Lower": test_metrics['Precision']['lower_ci'], 
            "Precision_Upper": test_metrics['Precision']['upper_ci'],
            "Recall_Mean": test_metrics['Recall']['mean'], 
            "Recall_Lower": test_metrics['Recall']['lower_ci'], 
            "Recall_Upper": test_metrics['Recall']['upper_ci'],
            "F1_Mean": test_metrics['F1']['mean'], 
            "F1_Lower": test_metrics['F1']['lower_ci'], 
            "F1_Upper": test_metrics['F1']['upper_ci'],
        })

# save output metrics to csv
df_results = pd.DataFrame(all_benchmark_records)
df_results.to_csv("federated_benchmark_results_baseline2.csv", index=False)
print("\nfinished! saved output to 'federated_benchmark_results_baseline2.csv'")