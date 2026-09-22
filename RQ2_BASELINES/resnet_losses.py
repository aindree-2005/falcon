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
if device.type == "cpu":
    torch.set_num_threads(os.cpu_count())

# file paths check with clean names
primary_path = "/kaggle/input/datasets/aindreechatterjee/dafl-falcon/eicu_dafl_ready.csv"
fallback_path = "/kaggle/input/dafl-falcon/eicu_dafl_ready.csv"
dataset_path = primary_path if os.path.exists(primary_path) else (fallback_path if os.path.exists(fallback_path) else "data/eicu/eicu_dafl_ready.csv")

print(f"loading dataset from {dataset_path}...")
df = pd.read_csv(dataset_path)
print(f"data shape: {df.shape}")

# setup hospital splits and features
selected_hospitals = [167, 420, 199, 458, 252, 165, 148, 281, 449, 283]
target_hospitals = [167, 199, 252, 420, 458]
feature_cols = df.columns.difference(["patientunitstayid", "hospitalid", "death", "ventilation", "sepsis"])

# partition client data
clients_raw = {}
for h in selected_hospitals:
    hospital_df = df[df["hospitalid"] == h]
    X = hospital_df[feature_cols].values
    y = hospital_df["death"].values
    clients_raw[h] = (X, y)

print(f"created {len(clients_raw)} hospital client partitions")

# evaluation function to compute metrics and threshold
def evaluate_comprehensive(model, X, y, batch_size=1024):
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
def evaluate_point_and_bootstrap_ci(model, X, y, fixed_threshold, batch_size=1024, n_boot=100, seed=SEED):
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

    point = {}
    point["AUROC"] = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.5
    point["AUPRC"] = float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else 0.0

    preds_full = (probs >= fixed_threshold).astype(int)
    point["F1"] = float(f1_score(y, preds_full, zero_division=0))
    point["Precision"] = float(precision_score(y, preds_full, zero_division=0))
    point["Recall"] = float(recall_score(y, preds_full, zero_division=0))

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

# focal loss definition
class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# supervised contrastive loss definition
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections, labels):
        device = projections.device
        batch_size = projections.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(projections, projections.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        mask_sum = mask.sum(1)
        valid_mask = mask_sum > 0

        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_mask] / mask_sum[valid_mask]
        return -mean_log_prob_pos.mean()

# pre-activation resnet block
class PreActResNetBlockLN(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.ln1(x)
        out = self.relu(out)
        out = self.linear1(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return x + out

# tabular resnet model with token pooling and contrastive projection head
class TabularResNetMortalityModel(nn.Module):
    def __init__(self, input_dim, d_token=16, hidden_dim=256, num_blocks=2):
        super().__init__()

        self.W_token = nn.Parameter(torch.randn(input_dim, d_token) * 0.02)
        self.b_token = nn.Parameter(torch.zeros(input_dim, d_token))
        self.emb_dropout = nn.Dropout(0.2)

        self.initial = nn.Linear(d_token, hidden_dim)
        self.blocks = nn.Sequential(
            *[PreActResNetBlockLN(hidden_dim) for _ in range(num_blocks)]
        )
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.final_relu = nn.ReLU()

        self.base = nn.Sequential(
            self.initial,
            self.blocks,
            self.final_ln,
            self.final_relu
        )

        self.head = nn.Linear(hidden_dim, 1)

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)
        )

    def _pooled_tokens(self, x):
        tokens = x.unsqueeze(-1) * self.W_token + self.b_token
        tokens = self.emb_dropout(tokens)
        return tokens.mean(dim=1)

    def forward(self, x):
        pooled = self._pooled_tokens(x)
        latents = self.base(pooled)
        return self.head(latents)

    def forward_with_latents(self, x):
        pooled = self._pooled_tokens(x)
        latents = self.base(pooled)
        logits = self.head(latents)
        projections = self.projector(latents)
        return logits, projections

# compute cosine learning rate decay per round
def get_lr(round_idx):
    eta_min = BASE_LR * 0.1
    return eta_min + 0.5 * (BASE_LR - eta_min) * (1 + math.cos(math.pi * round_idx / FED_ROUNDS))

# local train step with joint focal and supcon loss
def train_local_sgd(model, X, y, current_round, epochs=LOCAL_EPOCHS):
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    y_t = torch.tensor(y, dtype=torch.float32)
    pos_ratio = (y_t.sum() / len(y_t)).item()
    alpha_val = 1.0 - pos_ratio

    focal_criterion = BinaryFocalLossWithLogits(alpha=alpha_val, gamma=2.0).to(device)
    supcon_criterion = SupConLoss(temperature=0.07).to(device)
    lam = 0.1

    optimizer = optim.AdamW(model.parameters(), lr=get_lr(current_round), weight_decay=WEIGHT_DECAY)

    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()

            logits, projections = model.forward_with_latents(bx)

            loss_focal = focal_criterion(logits.squeeze(), by)

            projections_norm = nn.functional.normalize(projections, p=2, dim=1)
            loss_supcon = supcon_criterion(projections_norm, by)

            loss = loss_focal + lam * loss_supcon

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return {k: v.cpu() for k, v in model.state_dict().items()}

# main benchmark training loop
algorithms = ['FedAvg']
all_benchmark_records = []

for algo in algorithms:
    print(f"\n--- Running algorithm: {algo} (Pooled Tokenized ResNet + SupCon + Focal Loss) ---")

    for target_hospital in target_hospitals:
        print(f"\ntarget hospital fold: {target_hospital} ({algo})")

        # scale features using source hospital data only
        source_hospital_ids = [h for h in clients_raw.keys() if h != target_hospital]
        X_source_concat = np.concatenate([clients_raw[h][0] for h in source_hospital_ids], axis=0)
        source_scaler = StandardScaler()
        source_scaler.fit(X_source_concat)

        training_clients = {}
        for h in source_hospital_ids:
            X_raw, y_raw = clients_raw[h]
            training_clients[h] = (source_scaler.transform(X_raw), y_raw)

        # 70/30 train/test split on target hospital
        X_tgt_raw, y_tgt = clients_raw[target_hospital]
        X_val_raw, X_test_raw, y_val, y_test = train_test_split(
            X_tgt_raw, y_tgt, test_size=0.3, random_state=SEED, stratify=y_tgt
        )
        X_val = source_scaler.transform(X_val_raw)
        X_test = source_scaler.transform(X_test_raw)

        input_dim = X_val.shape[1]

        global_model = TabularResNetMortalityModel(input_dim).to(device)
        global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}

        best_val_auprc = -1.0
        best_weights_tgt = None
        locked_test_threshold = 0.5

        for r in range(FED_ROUNDS):
            client_weights = []
            client_sizes = []

            for hospital, (X, y) in training_clients.items():
                local_model = TabularResNetMortalityModel(input_dim).to(device)
                local_model.load_state_dict(global_state)

                trained_state = train_local_sgd(local_model, X, y, current_round=r)

                client_weights.append(trained_state)
                client_sizes.append(len(X))

            # aggregate local models via FedAvg
            total_samples = sum(client_sizes)
            for key in global_state.keys():
                tensors = [w[key] for w in client_weights]
                if tensors[0].dtype.is_floating_point:
                    global_state[key] = sum(tensors[i].float() * (client_sizes[i] / total_samples) for i in range(len(tensors)))
                else:
                    global_state[key] = tensors[0]

            eval_model = TabularResNetMortalityModel(input_dim).to(device)
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
        test_metrics = evaluate_point_and_bootstrap_ci(
            eval_model, X_test, y_test, fixed_threshold=locked_test_threshold, n_boot=100, seed=SEED
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

# export final output metrics with a sensible filename
output_csv = "resnet_supcon_focal_results.csv"
df_results = pd.DataFrame(all_benchmark_records)
df_results.to_csv(output_csv, index=False)

test_rows = df_results[df_results["Type"] == "Test"]
print("\n--- Final Summary ---")
for metric in ["AUROC", "AUPRC", "Precision", "Recall", "F1"]:
    vals = test_rows[metric].values
    print(f"  {metric:10s}: {vals.mean():.4f} +/- {vals.std():.4f}")

print(f"\nfinished training! saved output to '{output_csv}'")