import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# hospital settings
SELECTED_HOSPITALS = [167, 420, 199, 458, 252, 165, 148, 281, 449, 283]
TARGET_HOSPITALS   = [167, 199, 252, 420, 458]

# parameters based on common benchmark specs
INPUT_WINDOW    = 48 * 60  
NUM_BINS        = 24       
TOP_K_TREATMENTS= 150      
BATCH_SIZE      = 256
LOCAL_EPOCHS    = 5
FED_ROUNDS      = 25       
LEARNING_RATE   = 1.5e-3
WEIGHT_DECAY     = 1e-4

# paths setup
DATA_DIR = "/kaggle/input/datasets/aindreechatterjee/ablation-dafl"
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# preprocess treatment data and create features
def preprocess_treatment_data():
    print("loading ready csv...")
    
    ready_csv_path = os.path.join(DATA_DIR, "eicu_dafl_ready.csv")
    if not os.path.exists(ready_csv_path):
        ready_csv_path = "data/eicu/eicu_dafl_ready.csv"
    
    # filter main data for selected hospitals
    df_ready = pd.read_csv(ready_csv_path)
    df_ready = df_ready[df_ready["hospitalid"].isin(SELECTED_HOSPITALS)].copy()
    df_ready = df_ready.reset_index(drop=True)
    
    valid_stays = set(df_ready["patientunitstayid"].values)
    
    # locate treatment dataset file
    treat_path = os.path.join(DATA_DIR, "treatment.csv")
    if not os.path.exists(treat_path):
        treat_path = os.path.join(DATA_DIR, "treatment.csv.gz")
    if not os.path.exists(treat_path): 
        treat_path = "/kaggle/input/datasets/aindreechatterjee/dafl-falcon/treatment.csv.gz"
    if not os.path.exists(treat_path): 
        treat_path = "data/eicu/treatment.csv.gz"

    print(f"reading treatments from {treat_path}...")
    treatment = pd.read_csv(treat_path)
    treatment = treatment[treatment["patientunitstayid"].isin(valid_stays)]
    treatment = treatment[treatment["treatmentoffset"] <= INPUT_WINDOW].copy()
    
    # parse treatment strings
    treatment["level3"] = treatment["treatmentstring"].str.split("|").str[2].fillna("none").str.strip().str.lower()
    top_t = treatment["level3"].value_counts().head(TOP_K_TREATMENTS).index
    treatment["token"] = treatment["level3"].apply(lambda x: x if x in top_t else "other")
    
    # map tokens to indices
    token_to_idx = {token: i + 1 for i, token in enumerate(sorted(treatment["token"].unique()))}
    treatment["token_idx"] = treatment["token"].map(token_to_idx)
    vocab_size = len(token_to_idx) + 1
    
    treatment["hour_bin"] = (treatment["treatmentoffset"] // 120).clip(0, NUM_BINS - 1)
    
    patient_ids = df_ready["patientunitstayid"].values
    patient_map = {pid: i for i, pid in enumerate(patient_ids)}
    
    # create sequence array
    X_seq = np.zeros((len(df_ready), NUM_BINS, vocab_size), dtype=np.float32)
    print("populating treatment sequence matrix...")
    for row in treatment.itertuples(index=False):
        p_idx = patient_map.get(row.patientunitstayid)
        if p_idx is not None:
            X_seq[p_idx, int(row.hour_bin), int(row.token_idx)] = 1.0
            
    # pull demographic columns
    demographic_cols = [c for c in df_ready.columns if c not in ['patientunitstayid', 'hospitalid', 'death', 'ventilation', 'sepsis']]
    X_static = df_ready[demographic_cols].values.astype(np.float32)

    y = df_ready["death"].values.astype(int)
    hospitals = df_ready["hospitalid"].values

    print(f"vocab size = {vocab_size}, total patients = {len(df_ready)}")
    return X_seq, X_static, y, patient_ids, hospitals, vocab_size

# custom focal loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return torch.mean(self.alpha * (1 - pt)**self.gamma * BCE_loss)

# main model architecture
class MortalityPredictor(nn.Module):
    def __init__(self, vocab_dim, static_dim, hidden_dim=128, emb_dim=64):
        super().__init__()
        self.gru = nn.GRU(vocab_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 2 + static_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_seq, x_static, return_embeddings=False):
        g_out, _ = self.gru(x_seq)
        w = torch.softmax(self.attn(g_out), dim=1)
        c = torch.sum(w * g_out, dim=1)
        
        combined = torch.cat([c, x_static], dim=1)
        emb = self.bottleneck(combined)
        
        if return_embeddings:
            return emb
        return self.classifier(emb)

# train model locally for a single client
def train_local_sgd(model, X_seq, X_stat, y, current_round):
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss()
    
    for _ in range(LOCAL_EPOCHS):
        for bx, bs, by in loader:
            bx, bs, by = bx.to(device), bs.to(device), by.to(device).unsqueeze(1)
            optimizer.zero_grad()
            out = model(bx, bs)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    return {k: v.cpu() for k, v in model.state_dict().items()}, len(y)

# main loho federated loop
def run_loho_extraction():
    X_seq_all, X_static_all, y_all, patient_ids, hospitals_all, vocab_size = preprocess_treatment_data()
    static_dim = X_static_all.shape[1]
    
    # save patient array
    np.save(os.path.join(OUTPUT_DIR, "treatment_patient_ids.npy"), patient_ids)
    
    for target_hospital in TARGET_HOSPITALS:
        print(f"\n--- Running LOHO treatment target hospital: {target_hospital} ---")
        
        source_mask = hospitals_all != target_hospital
        
        # scale features on source data only
        X_static_scaled = X_static_all.copy()
        scaler = StandardScaler()
        X_static_scaled[source_mask] = scaler.fit_transform(X_static_scaled[source_mask])
        X_static_scaled[~source_mask] = scaler.transform(X_static_scaled[~source_mask])
        
        # setup client partitions
        source_hospital_ids = [h for h in SELECTED_HOSPITALS if h != target_hospital]
        training_clients = {}
        for h in source_hospital_ids:
            h_mask = hospitals_all == h
            training_clients[h] = (X_seq_all[h_mask], X_static_scaled[h_mask], y_all[h_mask])
            
        global_model = MortalityPredictor(vocab_size, static_dim).to(device)
        global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
        
        # start federated rounds
        for r in range(FED_ROUNDS):
            client_weights = []
            client_sizes = []
            
            for hospital, (X_seq, X_stat, y) in training_clients.items():
                local_model = MortalityPredictor(vocab_size, static_dim).to(device)
                local_model.load_state_dict(global_state)
                
                trained_state, n_samples = train_local_sgd(local_model, X_seq, X_stat, y, current_round=r)
                client_weights.append(trained_state)
                client_sizes.append(n_samples)
                
            # aggregate parameters
            total_samples = sum(client_sizes)
            for key in global_state.keys():
                tensors = [w[key] for w in client_weights]
                if tensors[0].dtype.is_floating_point:
                    global_state[key] = sum(tensors[i].float() * (client_sizes[i] / total_samples) for i in range(len(tensors)))
                else:
                    global_state[key] = tensors[0]
                    
            print(f"Finished round {r+1}/{FED_ROUNDS}")

        print(f"Extracting embeddings for fold {target_hospital}...")
        eval_model = MortalityPredictor(vocab_size, static_dim).to(device)
        eval_model.load_state_dict(global_state)
        eval_model.eval()
        
        # extract final representations
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
        emb_path = os.path.join(OUTPUT_DIR, f"treatment_embeddings_LOHO_target_{target_hospital}.npy")
        np.save(emb_path, all_emb)
        print(f"Saved treatment embeddings to {emb_path}")

    print("\nDone extracting all treatment embeddings!")

if __name__ == "__main__":
    run_loho_extraction()