import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
import random
import os
import math

warnings.filterwarnings("ignore")

# set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# hospital settings
SELECTED_HOSPITALS = [167, 420, 199, 458, 252, 165, 148, 281, 449, 283]
TARGET_HOSPITALS = [167, 199, 252, 420, 458]

# parameters based on common benchmark specs
INPUT_WINDOW = 48 * 60  
NUM_BINS = 24       
TOP_K_DIAGNOSES = 200      
BATCH_SIZE = 256
LOCAL_EPOCHS = 5
FED_ROUNDS = 25        
LEARNING_RATE = 1.5e-3
WEIGHT_DECAY = 1e-4

# paths setup
DATA_DIR = "/kaggle/input/datasets/aindreechatterjee/ablation-dafl"
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# preprocess data and construct vectors
def preprocess_diagnosis_data():
    print("loading eicu csv...")
    
    ready_csv_path = os.path.join(DATA_DIR, "eicu_dafl_ready.csv")
    if not os.path.exists(ready_csv_path):
        ready_csv_path = "data/eicu/eicu_dafl_ready.csv"
    
    # filter main data for selected hospitals
    df_ready = pd.read_csv(ready_csv_path)
    df_ready = df_ready[df_ready["hospitalid"].isin(SELECTED_HOSPITALS)].copy()
    df_ready = df_ready.reset_index(drop=True)
    
    valid_stays = set(df_ready["patientunitstayid"].values)
    
    # check possible diagnosis file locations
    diag_path = os.path.join(DATA_DIR, "diagnosis.csv")
    if not os.path.exists(diag_path):
        diag_path = os.path.join(DATA_DIR, "diagnosis.csv.gz")
    if not os.path.exists(diag_path): 
        diag_path = "/kaggle/input/datasets/aindreechatterjee/dafl-falcon/diagnosis.csv.gz"
    if not os.path.exists(diag_path): 
        diag_path = "data/eicu/diagnosis.csv.gz"

    print(f"reading diagnosis file from {diag_path}...")
    diagnosis = pd.read_csv(diag_path)
    diagnosis = diagnosis[diagnosis["patientunitstayid"].isin(valid_stays)]
    diagnosis = diagnosis[diagnosis["diagnosisoffset"] <= INPUT_WINDOW].copy()

    # parse diagnosis string levels
    diagnosis["diagnosisstring"] = diagnosis["diagnosisstring"].str.lower()
    parts = diagnosis["diagnosisstring"].str.split("|", expand=True)

    diagnosis["level2"] = parts[1].fillna("none").str.strip()
    diagnosis["level3"] = parts[2].fillna("none").str.strip()

    # select most frequent categories
    top_l2 = diagnosis["level2"].value_counts().head(TOP_K_DIAGNOSES // 2).index
    top_l3 = diagnosis["level3"].value_counts().head(TOP_K_DIAGNOSES).index

    # create tokens
    diagnosis["tok_l2"] = diagnosis["level2"].apply(lambda x: f"l2_{x}" if x in top_l2 else "l2_other")
    diagnosis["tok_l3"] = diagnosis["level3"].apply(lambda x: f"l3_{x}" if x in top_l3 else "l3_other")
    diagnosis["tok_pri"] = "pri_other"

    # map tokens to numbers
    all_tokens = diagnosis["tok_l2"].unique().tolist() + diagnosis["tok_l3"].unique().tolist() + diagnosis["tok_pri"].unique().tolist()
    token_to_idx = {t: i + 1 for i, t in enumerate(sorted(set(all_tokens)))}
    vocab_size = len(token_to_idx) + 1 

    diagnosis["idx_l2"] = diagnosis["tok_l2"].map(token_to_idx)
    diagnosis["idx_l3"] = diagnosis["tok_l3"].map(token_to_idx)
    diagnosis["idx_pri"] = diagnosis["tok_pri"].map(token_to_idx)
    diagnosis["hour_bin"] = (diagnosis["diagnosisoffset"] // 120).clip(0, NUM_BINS - 1)

    # prepare sequence tensor
    patient_ids = df_ready["patientunitstayid"].values
    patient_map = {pid: i for i, pid in enumerate(patient_ids)}
    X_seq = np.zeros((len(df_ready), NUM_BINS, vocab_size), dtype=np.float32)

    print("populating sequence matrix...")
    for row in diagnosis.itertuples(index=False):
        p_idx = patient_map.get(row.patientunitstayid)
        if p_idx is None: continue
        t = int(row.hour_bin)
        X_seq[p_idx, t, int(row.idx_l2)] = 1.0
        X_seq[p_idx, t, int(row.idx_l3)] = 1.0
        X_seq[p_idx, t, int(row.idx_pri)] = 1.0

    # pull demographic features
    demographic_cols = [c for c in df_ready.columns if c not in ['patientunitstayid', 'hospitalid', 'death', 'ventilation', 'sepsis']]
    X_static = df_ready[demographic_cols].values.astype(np.float32)

    y = df_ready["death"].values.astype(int)
    hospitals = df_ready["hospitalid"].values

    print(f"vocab size = {vocab_size}, total patients = {len(df_ready)}")
    return X_seq, X_static, y, patient_ids, hospitals, vocab_size

# custom attention layer
class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads=3):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(hidden_dim * 2 * n_heads, hidden_dim * 2)

    def forward(self, x):
        contexts = []
        for head in self.heads:
            w = torch.softmax(head(x), dim=1)
            contexts.append((w * x).sum(dim=1))
        cat = torch.cat(contexts, dim=-1)
        return self.out_proj(cat)

# main prediction model
class DiagnosisPredictor(nn.Module):
    def __init__(self, vocab_dim, static_dim, hidden_dim=128, emb_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vocab_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.gru = nn.GRU(
            64, hidden_dim, num_layers=2,
            batch_first=True, bidirectional=True,
            dropout=0.25
        )
        self.attention = MultiHeadTemporalAttention(hidden_dim, n_heads=3)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 2 + static_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(self, x_seq, x_static, return_embeddings=False):
        x = self.proj(x_seq)
        gru_out, _ = self.gru(x)
        context = self.attention(gru_out)
        combined = torch.cat([context, x_static], dim=1)
        emb = self.bottleneck(combined)
        
        if return_embeddings:
            return emb
        return self.classifier(emb).squeeze(-1)

# train local model for a client
def train_local_sgd(model, X_seq, X_stat, y, pos_weight_val, current_round):
    model.train()
    
    loader = DataLoader(
        TensorDataset(torch.tensor(X_seq, dtype=torch.float32), 
                      torch.tensor(X_stat, dtype=torch.float32), 
                      torch.tensor(y, dtype=torch.float32)), 
        batch_size=BATCH_SIZE, shuffle=True
    )
    
    # cosine decay step per round
    eta_min = LEARNING_RATE * 0.1
    current_lr = eta_min + 0.5 * (LEARNING_RATE - eta_min) * (1 + math.cos(math.pi * current_round / FED_ROUNDS))
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=WEIGHT_DECAY)
    
    for _ in range(LOCAL_EPOCHS):
        for bx, bs, by in loader:
            bx, bs, by = bx.to(device), bs.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx, bs)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
    return {k: v.cpu() for k, v in model.state_dict().items()}, len(y)

# main loho federated loop
def run_loho_extraction():
    X_seq_all, X_static_all, y_all, patient_ids, hospitals_all, vocab_size = preprocess_diagnosis_data()
    static_dim = X_static_all.shape[1]
    
    # save patient ids array
    np.save(os.path.join(OUTPUT_DIR, "diagnosis_patient_ids.npy"), patient_ids)
    
    for target_hospital in TARGET_HOSPITALS:
        print(f"\n--- Running LOHO target hospital: {target_hospital} ---")
        
        source_mask = hospitals_all != target_hospital
        
        # scale features based on source split
        X_static_scaled = X_static_all.copy()
        scaler = StandardScaler()
        X_static_scaled[source_mask] = scaler.fit_transform(X_static_scaled[source_mask])
        X_static_scaled[~source_mask] = scaler.transform(X_static_scaled[~source_mask])
        
        # separate client datasets
        source_hospital_ids = [h for h in SELECTED_HOSPITALS if h != target_hospital]
        training_clients = {}
        for h in source_hospital_ids:
            h_mask = hospitals_all == h
            training_clients[h] = (X_seq_all[h_mask], X_static_scaled[h_mask], y_all[h_mask])
            
        y_source = y_all[source_mask]
        pos_weight = (len(y_source) - y_source.sum()) / max(y_source.sum(), 1)
        
        global_model = DiagnosisPredictor(vocab_dim=vocab_size, static_dim=static_dim).to(device)
        global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
        
        # federated training rounds
        for r in range(FED_ROUNDS):
            client_weights = []
            client_sizes = []
            
            for hospital, (X_seq, X_stat, y) in training_clients.items():
                local_model = DiagnosisPredictor(vocab_dim=vocab_size, static_dim=static_dim).to(device)
                local_model.load_state_dict(global_state)
                
                trained_state, n_samples = train_local_sgd(local_model, X_seq, X_stat, y, pos_weight, current_round=r)
                client_weights.append(trained_state)
                client_sizes.append(n_samples)
                
            # aggregate weights (FedAvg)
            total_samples = sum(client_sizes)
            for key in global_state.keys():
                tensors = [w[key] for w in client_weights]
                if tensors[0].dtype.is_floating_point:
                    global_state[key] = sum(tensors[i].float() * (client_sizes[i] / total_samples) for i in range(len(tensors)))
                else:
                    global_state[key] = tensors[0]
                    
            print(f"Finished round {r+1}/{FED_ROUNDS}")

        print(f"Extracting embeddings for fold {target_hospital}...")
        eval_model = DiagnosisPredictor(vocab_dim=vocab_size, static_dim=static_dim).to(device)
        eval_model.load_state_dict(global_state)
        eval_model.eval()
        
        # run inference to save final embeddings
        all_emb = []
        extract_loader = DataLoader(
            TensorDataset(torch.tensor(X_seq_all, dtype=torch.float32), torch.tensor(X_static_scaled, dtype=torch.float32)),
            batch_size=256, shuffle=False
        )
        
        with torch.no_grad():
            for bx, bs in extract_loader:
                emb = eval_model(bx.to(device), bs.to(device), return_embeddings=True).cpu().numpy()
                all_emb.append(emb)
                
        all_emb = np.vstack(all_emb)
        emb_path = os.path.join(OUTPUT_DIR, f"diagnosis_embeddings_LOHO_target_{target_hospital}.npy")
        np.save(emb_path, all_emb)
        print(f"Saved embeddings to {emb_path}")

    print("\nDone extracting all embeddings!")

if __name__ == "__main__":
    run_loho_extraction()