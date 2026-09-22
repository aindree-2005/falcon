import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
from torch.utils.data import TensorDataset, DataLoader
import copy
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

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ====================================================================================
# 1. HARDWARE & DATA LOADING
# ====================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">> Utilizing computation device: {device}", flush=True)

primary_path = "/kaggle/input/datasets/aindreechatterjee/dafl-falcon/eicu_dafl_ready.csv"
fallback_path = "/kaggle/input/dafl-falcon/eicu_dafl_ready.csv"
dataset_path = primary_path if os.path.exists(primary_path) else (fallback_path if os.path.exists(fallback_path) else "data/eicu/eicu_dafl_ready.csv")

print(f">> Loading '{dataset_path}'...", flush=True)
df = pd.read_csv(dataset_path)

selected_hospitals = [167, 420, 199, 458, 252, 165, 148, 281, 449, 283]
target_hospitals = [167, 199, 252, 420, 458]

# ABLATION FIX: Removed "ventilation" and "sepsis" from meta_cols so they are 
# included in the feature set alongside demographics, labs, and medicines.
meta_cols = ["patientunitstayid", "hospitalid", "death"]

# Identified 12 Lab Columns from Preprocessing Pipeline
lab_cols = [
    'o2sat', 'pao2', 'paco2', 'ph', 'albumin_lab', 'bands', 
    'bun', 'hct', 'inr', 'lactate', 'platelets', 'wbc'
]

# Total Input Feature Set (Now includes Sepsis and Ventilation)
feature_cols = df.columns.difference(meta_cols)

# Non-Lab Feature Indices (Used strictly for SWD & Prototype Calculation)
non_lab_cols = [c for c in feature_cols if c not in lab_cols]
non_lab_indices = [list(feature_cols).index(c) for c in non_lab_cols]

clients_raw = {}
for h in selected_hospitals:
    hospital_df = df[df["hospitalid"] == h]
    X = hospital_df[feature_cols].values
    y = hospital_df["death"].values
    clients_raw[h] = (X, y)

print(f">> Data loaded. Partitions created for {len(clients_raw)} hospitals.", flush=True)
print(f">> Total features: {len(feature_cols)} | Non-Lab features for SWD/Prototypes: {len(non_lab_cols)}")

# ====================================================================================
# 2. METRICS EVALUATION UTILS
# ====================================================================================
def evaluate_comprehensive(model, X, y, batch_size=1024):
    model.eval()
    all_logits = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (bx,) in loader:
            logits, _ = model(bx.to(device))
            all_logits.append(logits.squeeze().cpu().numpy())

    logits = np.concatenate(all_logits)
    if np.ndim(logits) == 0: logits = np.array([logits])
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
        "AUROC": float(auc), "AUPRC": float(auprc), "F1": float(f1),
        "Precision": float(prec), "Recall": float(rec), "Best_Threshold": float(best_threshold)
    }

def evaluate_point_and_bootstrap_ci(model, X, y, fixed_threshold, n_boot=100, seed=SEED):
    model.eval()
    all_logits = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=1024, shuffle=False)

    with torch.no_grad():
        for (bx,) in loader:
            logits, _ = model(bx.to(device))
            all_logits.append(logits.squeeze().cpu().numpy())

    logits = np.concatenate(all_logits)
    if np.ndim(logits) == 0: logits = np.array([logits])
    probs = 1 / (1 + np.exp(-logits))

    point = {
        "AUROC": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.5,
        "AUPRC": float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else 0.0
    }
    preds_full = (probs >= fixed_threshold).astype(int)
    point["F1"] = float(f1_score(y, preds_full, zero_division=0))
    point["Precision"] = float(precision_score(y, preds_full, zero_division=0))
    point["Recall"] = float(recall_score(y, preds_full, zero_division=0))

    rng = np.random.RandomState(seed)
    boot = {'AUROC': [], 'AUPRC': [], 'F1': [], 'Precision': [], 'Recall': []}

    for b in range(n_boot):
        boot_seed = rng.randint(0, 2**31 - 1)
        y_b, p_b = resample(y, probs, replace=True, random_state=boot_seed, stratify=y)
        if len(np.unique(y_b)) < 2: continue

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

# ====================================================================================
# 3. SWD UTILS & LOSS FUNCTIONS
# ====================================================================================
def sliced_wasserstein_distance(Z_i, Z_target, num_projections=128, p=2, max_samples=1024, device='cuda'):
    n_i, n_t = Z_i.shape[0], Z_target.shape[0]
    min_n = min(n_i, n_t, max_samples)
    
    Z_i_sample = Z_i[torch.randperm(n_i, device=device)[:min_n]] if n_i > min_n else Z_i
    Z_t_sample = Z_target[torch.randperm(n_t, device=device)[:min_n]] if n_t > min_n else Z_target
        
    dim = Z_i.shape[1]
    projections = torch.randn(dim, num_projections, device=device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    
    proj_i, _ = torch.sort(torch.matmul(Z_i_sample, projections), dim=0)
    proj_t, _ = torch.sort(torch.matmul(Z_t_sample, projections), dim=0)
    
    return torch.mean(torch.abs(proj_i - proj_t) ** p).item()

class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * self.bce(logits, targets)).mean()

def true_supcon_loss(features, labels, temperature=0.2):
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    features = torch.nn.functional.normalize(features, dim=1)
    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(features.shape[0]).view(-1, 1).to(device), 0)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    mask_sum = mask.sum(1)
    valid = mask_sum > 0
    if not valid.any(): return torch.tensor(0.0, device=device)
    return (-(mask * log_prob).sum(1)[valid] / mask_sum[valid]).mean()

# ====================================================================================
# 4. TABULAR RESNET ARCHITECTURE
# ====================================================================================
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
        res = x
        out = self.ln1(x)
        out = self.relu(out)
        out = self.linear1(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return res + out

class TabularResNetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, d_token=16, num_blocks=2, dropout=0.2):
        super().__init__()
        self.W_token = nn.Parameter(torch.randn(input_dim, d_token) * 0.02)
        self.b_token = nn.Parameter(torch.zeros(input_dim, d_token))
        self.emb_dropout = nn.Dropout(0.2)

        layers = [nn.Linear(d_token, hidden_dim)]
        for _ in range(num_blocks):
            layers.append(PreActResNetBlockLN(hidden_dim, dropout=dropout))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        
        self.base = nn.Sequential(*layers)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.W_token + self.b_token
        tokens = self.emb_dropout(tokens)
        pooled = tokens.mean(dim=1)
        return self.base(pooled)

    def extract_non_lab_latents(self, x, non_lab_indices):
        x_sub = x.clone()
        lab_mask = torch.ones_like(x_sub)
        lab_mask[:, non_lab_indices] = 1.0
        all_indices = set(range(x.shape[1]))
        lab_only_indices = list(all_indices - set(non_lab_indices))
        x_sub[:, lab_only_indices] = 0.0
        
        tokens = x_sub.unsqueeze(-1) * self.W_token + self.b_token
        tokens = self.emb_dropout(tokens)
        pooled = tokens.mean(dim=1)
        return self.base(pooled)

class FALCONUltimateModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = TabularResNetEncoder(input_dim, hidden_dim, d_token=16, num_blocks=2, dropout=0.2)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.encoder(x)
        logits = self.classifier(h)
        return logits, h

# ====================================================================================
# 5. META-CNN AGGREGATOR
# ====================================================================================
class CalibratedConvAggregator(nn.Module):
    def __init__(self, num_clients, tau=2.0, max_weight_cap=0.35, alpha_smooth=0.10):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 1, kernel_size=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        self.quality_scale = nn.Parameter(torch.tensor(1.0))
        self.lambda_sim = nn.Parameter(torch.tensor(1.0))
        self.tau = tau
        self.max_weight_cap = max_weight_cap
        self.alpha_smooth = alpha_smooth

    def forward(self, stacked_updates, client_confidences, client_similarities):
        x = stacked_updates.unsqueeze(1) 
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.conv2(x1)
        conv_scores = self.pool(x2).view(-1)
        
        sim_term = self.lambda_sim * client_similarities
        qual_term = self.quality_scale * client_confidences
        fused_logits = conv_scores + qual_term + sim_term
        scaled_logits = fused_logits / self.tau
        raw_weights = torch.softmax(scaled_logits, dim=0)
        
        K = stacked_updates.size(0)
        smoothed_weights = (1.0 - self.alpha_smooth) * raw_weights + self.alpha_smooth * (1.0 / K)
        
        if self.max_weight_cap is not None and self.max_weight_cap < 1.0:
            clipped_weights = torch.clamp(smoothed_weights, max=self.max_weight_cap)
            final_weights = clipped_weights / clipped_weights.sum()
        else:
            final_weights = smoothed_weights
            
        return final_weights

def flatten_weights(state_dict, prefix=""): 
    return torch.cat([v.flatten() for k, v in state_dict.items() if v.dtype.is_floating_point and k.startswith(prefix)])

def update_state_dict_from_flat(flat_tensor, target_state_dict, prefix=""):
    new_state_dict = dict(target_state_dict)
    idx = 0
    for k, v in target_state_dict.items():
        if v.dtype.is_floating_point and k.startswith(prefix):
            numel = v.numel()
            new_state_dict[k] = flat_tensor[idx:idx+numel].view(v.shape)
            idx += numel
    return new_state_dict

def cosine_lr(base_lr, round_idx, total_rounds, min_frac=0.1):
    progress = round_idx / max(total_rounds - 1, 1)
    return base_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + np.cos(np.pi * progress)))

# ====================================================================================
# 6. LOCAL CLIENT TRAINING
# ====================================================================================
def train_local_falcon(model, X, y, epochs=4, lr=2e-3, global_weights=None, global_proto=None, non_lab_indices=None):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    y_t = torch.tensor(y, dtype=torch.float32)
    pos = y_t.sum().item(); neg = len(y_t) - pos
    criterion = BinaryFocalLossWithLogits(alpha=neg/(pos+neg+1e-8), gamma=2.0).to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), y_t), batch_size=256, shuffle=True)

    global_params = {k: v.clone().detach().to(device) for k, v in global_weights.items()} if global_weights else {}

    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, lat = model(bx)
            
            lf = criterion(logits.squeeze(), by)
            ls = 0.05 * true_supcon_loss(lat, by, temperature=0.2)
            
            if global_proto is not None and non_lab_indices is not None:
                non_lab_lat = model.encoder.extract_non_lab_latents(bx, non_lab_indices)
                la = 0.05 * torch.nn.functional.mse_loss(non_lab_lat.mean(dim=0), global_proto)
            else:
                la = 0.0
                
            lp = 0.0
            if global_params:
                for name, param in model.named_parameters():
                    if name in global_params:
                        lp += torch.sum((param - global_params[name]) ** 2)
            lp = (1e-4 / 2.0) * lp

            (lf + la + ls + lp).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    return model.state_dict()

# ====================================================================================
# 7. MAIN FEDERATED EXPERIMENT LOOP
# ====================================================================================
all_results = {}
algo = 'FALCON-Ablation-AllFeatures'

print(f"\n==========================================================================", flush=True)
print(f">> RUNNING EXPERIMENTS FOR: {algo}", flush=True)
print(f">> Objective: Full Feature Ablation (Demographics+Labs+Medicines+Sepsis+Ventilation)", flush=True)
print(f"==========================================================================", flush=True)

final_results = {}
all_benchmark_records = []
rounds = 25
base_lr = 1.5e-3

for target_hospital in target_hospitals:
    print(f"\n--- Evaluating Target Hospital: {target_hospital} ---", flush=True)

    source_hospital_ids = [h for h in clients_raw.keys() if h != target_hospital]
    
    X_source_concat = np.concatenate([clients_raw[h][0] for h in source_hospital_ids], axis=0)
    source_scaler = StandardScaler()
    source_scaler.fit(X_source_concat)

    training_clients = {}
    source_val_x_list = []
    source_val_y_list = []
    
    for h in source_hospital_ids:
        X_raw, y_raw = clients_raw[h]
        X_scaled = source_scaler.transform(X_raw)
        
        X_tr, X_v, y_tr, y_v = train_test_split(X_scaled, y_raw, test_size=0.2, random_state=42, stratify=y_raw)
        training_clients[h] = {"X_train": X_tr, "y_train": y_tr, "X_val": X_v, "y_val": y_v}
        source_val_x_list.append(X_v)
        source_val_y_list.append(y_v)

    X_global_ref = np.concatenate(source_val_x_list, axis=0)
    y_global_ref = np.concatenate(source_val_y_list, axis=0)
    X_global_ref_t = torch.tensor(X_global_ref, dtype=torch.float32).to(device)
    y_global_ref_t = torch.tensor(y_global_ref, dtype=torch.float32).to(device)

    X_tgt_raw, y_tgt = clients_raw[target_hospital]
    X_tgt_val_raw, X_tgt_test_raw, y_tgt_val, y_tgt_test = train_test_split(
        X_tgt_raw, y_tgt, test_size=0.2, random_state=42, stratify=y_tgt
    )
    X_tgt_val = source_scaler.transform(X_tgt_val_raw)
    X_tgt_test = source_scaler.transform(X_tgt_test_raw)
    
    input_dim = X_tgt_val.shape[1]
    
    global_model = FALCONUltimateModel(input_dim, hidden_dim=64).to(device)
    conv_aggregator = CalibratedConvAggregator(num_clients=len(training_clients)).to(device)
    aggregator_optimizer = optim.AdamW(conv_aggregator.parameters(), lr=1e-4, weight_decay=1e-2)

    best_val_auprc = -1
    best_weights = None
    global_prototype = None
    BETA_PARAM = 2.0 

    for r in range(rounds):
        round_lr = cosine_lr(base_lr, r, rounds)
        client_weights = []
        client_sizes = []
        client_flat_enc_updates = []
        client_confidences = []
        client_full_latents = []
        client_mean_protos = []

        global_enc_flat = flatten_weights(global_model.state_dict(), prefix='encoder')

        global_model.eval()
        with torch.no_grad():
            ref_latents = []
            for i in range(0, len(X_global_ref_t), 1024):
                l = global_model.encoder.extract_non_lab_latents(X_global_ref_t[i:i+1024], non_lab_indices)
                ref_latents.append(l)
            Z_ref = torch.cat(ref_latents) 

        global_proto_t = torch.tensor(global_prototype, dtype=torch.float32).to(device) if global_prototype is not None else None
        
        for hospital, client_data in training_clients.items():
            local_model = FALCONUltimateModel(input_dim, hidden_dim=64).to(device)
            local_model.load_state_dict(global_model.state_dict())

            weights = train_local_falcon(
                local_model, client_data["X_train"], client_data["y_train"], 
                epochs=4, lr=round_lr, 
                global_weights=global_model.state_dict(),
                global_proto=global_proto_t,
                non_lab_indices=non_lab_indices
            )

            local_model.eval()
            with torch.no_grad():
                X_local_t = torch.tensor(client_data["X_train"], dtype=torch.float32)
                l_latents = []
                for i in range(0, len(X_local_t), 1024):
                    l = local_model.encoder.extract_non_lab_latents(X_local_t[i:i+1024].to(device), non_lab_indices)
                    l_latents.append(l)
                latents_full = torch.cat(l_latents)
                
            client_full_latents.append(latents_full)
            client_mean_protos.append(latents_full.mean(dim=0).cpu().numpy())

            local_model.load_state_dict(weights)
            v_metrics = evaluate_comprehensive(local_model, client_data["X_val"], client_data["y_val"])
            client_confidences.append(v_metrics['AUPRC'])

            client_weights.append(weights)
            client_sizes.append(len(client_data["X_train"]))
            
            update = flatten_weights(weights, prefix='encoder').to(device) - global_enc_flat
            client_flat_enc_updates.append(update)

        swd_distances = []
        for z_i in client_full_latents:
            swd = sliced_wasserstein_distance(z_i, Z_ref, num_projections=128, device=device)
            swd_distances.append(swd)
            
        swd_arr = np.array(swd_distances)
        min_swd, max_swd = np.min(swd_arr), np.max(swd_arr)
        swd_norm = (swd_arr - min_swd) / (max_swd - min_swd + 1e-8)
        client_similarities = np.exp(-BETA_PARAM * swd_norm).tolist()

        global_prototype = np.sum([p * s for p, s in zip(client_mean_protos, client_sizes)], axis=0) / np.sum(client_sizes)

        stacked_enc_updates = torch.stack(client_flat_enc_updates)
        conf_t = torch.tensor(client_confidences, dtype=torch.float32).to(device)
        sim_t = torch.tensor(client_similarities, dtype=torch.float32).to(device)

        conv_aggregator.train()
        aggregator_optimizer.zero_grad()
            
        enc_scores = conv_aggregator(stacked_enc_updates, conf_t, sim_t)
        stacked_enc_weights = torch.stack([flatten_weights(w, prefix='encoder').to(device) for w in client_weights])
        aggregated_enc_flat = torch.sum(stacked_enc_weights * enc_scores.unsqueeze(1), dim=0)

        sizes_t = torch.tensor(client_sizes, dtype=torch.float32).to(device)
        cls_scores = conf_t * (sizes_t / sizes_t.sum())
        cls_scores = cls_scores / (cls_scores.sum() + 1e-8)
        
        stacked_cls_weights = torch.stack([flatten_weights(w, prefix='classifier').to(device) for w in client_weights])
        aggregated_cls_flat = torch.sum(stacked_cls_weights * cls_scores.unsqueeze(1), dim=0)

        global_dict = update_state_dict_from_flat(aggregated_enc_flat, global_model.state_dict(), prefix='encoder')
        global_dict = update_state_dict_from_flat(aggregated_cls_flat, global_dict, prefix='classifier')
        
        buffer_keys = {k for k, _ in global_model.named_buffers()}
        meta_state_dict = {k: (v.detach() if k in buffer_keys else v) for k, v in global_dict.items()}

        global_model.eval()
        
        perm_idx = torch.randperm(len(X_global_ref_t), device=device)[:1024]
        X_meta, y_meta = X_global_ref_t[perm_idx], y_global_ref_t[perm_idx]
            
        val_logits, _ = functional_call(global_model, meta_state_dict, (X_meta,), tie_weights=False)
        
        pos_c = y_meta.sum().item(); neg_c = len(y_meta) - pos_c
        meta_criterion = BinaryFocalLossWithLogits(alpha=neg_c / (pos_c + neg_c + 1e-8), gamma=2.0).to(device)
        
        meta_loss_before = meta_criterion(val_logits.squeeze(), y_meta)
        meta_loss_before.backward()
        
        torch.nn.utils.clip_grad_norm_(conv_aggregator.parameters(), max_norm=0.5)
        aggregator_optimizer.step()

        old_global_dict = global_model.state_dict()
        final_global_dict = update_state_dict_from_flat(aggregated_enc_flat.detach(), old_global_dict, prefix='encoder')
        final_global_dict = update_state_dict_from_flat(aggregated_cls_flat.detach(), final_global_dict, prefix='classifier')
        global_model.load_state_dict(final_global_dict)

        val_metrics = evaluate_comprehensive(global_model, X_tgt_val, y_tgt_val)
        print(f"Round {r:2d} (lr={round_lr:.5f}) | Target Val AUROC: {val_metrics['AUROC']:.4f} | Target Val AUPRC: {val_metrics['AUPRC']:.4f}", flush=True)

        all_benchmark_records.append({
            "Algorithm": algo,
            "Target_Hospital": target_hospital,
            "Round": r,
            "Type": "Validation",
            **val_metrics
        })

        if val_metrics["AUPRC"] > best_val_auprc:
            best_val_auprc = val_metrics["AUPRC"]
            best_weights = copy.deepcopy(global_model.state_dict())
            locked_test_threshold = val_metrics["Best_Threshold"]

    global_model.load_state_dict(best_weights)
    print(f"\n  >> Running 100-seed bootstrap evaluation on TEST set for Target {target_hospital}...", flush=True)
    
    test_metrics = evaluate_point_and_bootstrap_ci(
        global_model, X_tgt_test, y_tgt_test, fixed_threshold=locked_test_threshold, n_boot=100, seed=SEED
    )

    print(f"  --> TEST AUROC:     {test_metrics['AUROC']['point']:.4f} (95% CI: {test_metrics['AUROC']['lower_ci']:.4f} to {test_metrics['AUROC']['upper_ci']:.4f})", flush=True)
    print(f"  --> TEST AUPRC:     {test_metrics['AUPRC']['point']:.4f} (95% CI: {test_metrics['AUPRC']['lower_ci']:.4f} to {test_metrics['AUPRC']['upper_ci']:.4f})", flush=True)
    print(f"  --> TEST Precision: {test_metrics['Precision']['point']:.4f} (95% CI: {test_metrics['Precision']['lower_ci']:.4f} to {test_metrics['Precision']['upper_ci']:.4f})", flush=True)
    print(f"  --> TEST Recall:    {test_metrics['Recall']['point']:.4f} (95% CI: {test_metrics['Recall']['lower_ci']:.4f} to {test_metrics['Recall']['upper_ci']:.4f})", flush=True)
    print(f"  --> TEST F1-Score:  {test_metrics['F1']['point']:.4f} (95% CI: {test_metrics['F1']['lower_ci']:.4f} to {test_metrics['F1']['upper_ci']:.4f})\n", flush=True)

    all_benchmark_records.append({
        "Algorithm": algo, "Target_Hospital": target_hospital, "Round": "FINAL_TEST", "Type": "Test",
        "AUROC": test_metrics['AUROC']['point'], "AUROC_Lower": test_metrics['AUROC']['lower_ci'], "AUROC_Upper": test_metrics['AUROC']['upper_ci'],
        "AUPRC": test_metrics['AUPRC']['point'], "AUPRC_Lower": test_metrics['AUPRC']['lower_ci'], "AUPRC_Upper": test_metrics['AUPRC']['upper_ci'],
        "Precision": test_metrics['Precision']['point'], "Precision_Lower": test_metrics['Precision']['lower_ci'], "Precision_Upper": test_metrics['Precision']['upper_ci'],
        "Recall": test_metrics['Recall']['point'], "Recall_Lower": test_metrics['Recall']['lower_ci'], "Recall_Upper": test_metrics['Recall']['upper_ci'],
        "F1": test_metrics['F1']['point'], "F1_Lower": test_metrics['F1']['lower_ci'], "F1_Upper": test_metrics['F1']['upper_ci'],
        "Best_Threshold": locked_test_threshold,
    })

df_results = pd.DataFrame(all_benchmark_records)
df_results.to_csv("falcon_ablation_allfeatures_benchmark.csv", index=False)

test_rows = df_results[df_results["Type"] == "Test"]
print("\n==========================================================================", flush=True)
print(f">> FINAL SUMMARY RESULTS: {algo} (Strict Zero-Leakage LOHO)", flush=True)
print("==========================================================================", flush=True)
for metric in ["AUROC", "AUPRC", "Precision", "Recall", "F1"]:
    vals = test_rows[metric].values
    print(f"  {metric:10s}: {vals.mean():.4f} +/- {vals.std():.4f}", flush=True)

print("\n>> Benchmark completed successfully. Saved to 'falcon_ablation_allfeatures_benchmark.csv'.", flush=True)