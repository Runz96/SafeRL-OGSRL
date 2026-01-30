# DDPG with CQL (No Guardian Version)

import os
import json
import math
import random
import copy
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm

# ─── HYPERPARAMETERS ───────────────────────────────────────────────────────────
GAMMA           = 0.99
TAU             = 0.001
BATCH_SIZE      = 32
NUM_EPOCHS      = 20000
CLIP_REWARD     = True
HIDDEN_SIZE     = 128

# CQL-Specific
NUM_CANDIDATES  = 12
NOISE_STD       = 0.1
CQL_ALPHA       = 1

# Logging and Checkpointing Intervals
SAVE_INTERVAL   = 100
EVAL_INTERVAL   = 1000

CHECKPOINT_DIR  = "../checkpoints/ddpg_cql_ts"
LOG_DIR         = "../logs/ddpg_cql_ts"

# ─── DEVICE SETUP ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─── SEED UTILITIES ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPG with CQL (no guardian) on Sepsis")
    parser.add_argument("--seed-index", "-i", type=int, default=0, help="Index into the seed list")
    parser.add_argument("--seeds-file", "-s", type=str, default="../seeds.json", help="Path to JSON seed file")
    return parser.parse_args()

def load_seed(seeds_file, idx):
    with open(seeds_file, 'r') as f:
        data = json.load(f)
    seeds = data.get('seeds', [])
    if not seeds:
        raise ValueError(f"No 'seeds' key in {seeds_file}")
    if idx < 0 or idx >= len(seeds):
        raise IndexError(f"seed-index {idx} out of range (0–{len(seeds)-1})")
    return seeds[idx]

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
# ─── SAMPLING METHOD ───────────────────────────────────────────────────────────
def sample_trajectory_level(df, traj_col='traj', action_cols=['a:max_dose_vaso', 'a:input_4hourly'],
                            n_bins=5, strategy='under_over', K=1.0):
    df_copy = df.copy()
    for action_col in action_cols:
        bin_col = f"{action_col}_bin"
        df_copy[bin_col] = pd.cut(df_copy[action_col], bins=n_bins, labels=False, include_lowest=True)
    bin_cols = [f"{col}_bin" for col in action_cols]
    df_copy['action_bin'] = df_copy[bin_cols].astype(str).agg('_'.join, axis=1)
    traj_action_bins = df_copy.groupby(traj_col)['action_bin'].agg(lambda x: x.mode().iloc[0]).reset_index()
    action_counts = traj_action_bins['action_bin'].value_counts().reset_index()
    action_counts.columns = ['action_bin', 'counts']
    mean_count = action_counts['counts'].mean()
    sampled_traj_ids = []
    for _, row in action_counts.iterrows():
        action_filter = traj_action_bins['action_bin'] == row['action_bin']
        action_trajs = traj_action_bins[action_filter]
        if strategy == 'under_over':
            sampled_trajs = action_trajs.sample(int(mean_count), replace=True)
        elif strategy == 'undersampling' and row['counts'] > K * mean_count:
            sampled_trajs = action_trajs.sample(int(K * mean_count), replace=False)
        elif strategy == 'oversampling' and row['counts'] < K * mean_count:
            sampled_trajs = action_trajs.sample(int(K * mean_count), replace=True)
        else:
            sampled_trajs = action_trajs
        sampled_traj_ids.extend(sampled_trajs[traj_col].tolist())
    return df[df[traj_col].isin(sampled_traj_ids)].reset_index(drop=True)


# ─── MODEL DEFINITIONS ─────────────────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc_out = nn.Linear(hidden_size, action_dim)
        # Use Tanh activation to bound output between -1 and 1; you can later scale if needed.
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        action = self.tanh(self.fc_out(x))
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE):
        super(Critic, self).__init__()
        # The input is the concatenation of state and action.
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
    
    def forward(self, state, action):
        # Concatenate state and action along the feature dimension
        x = torch.cat([state, action], dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        q_value = self.fc_out(x)
        return q_value

  
# ─── SOFT UPDATE ───────────────────────────────────────────────────────────────
def soft_update(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


# ─── DATA HELPERS ──────────────────────────────────────────────────────────────
def load_and_split_data():
    df = pd.read_csv('../sepsis_final_RAW_continuous_13.csv')
    train_idx = np.load('../train_indices.npy')
    val_idx   = np.load('../val_indices.npy')
    test_idx  = np.load('../test_indices.npy')

    train_df = copy.deepcopy(df.loc[train_idx])
    val_df   = copy.deepcopy(df.loc[val_idx].copy())
    test_df  = copy.deepcopy(df.loc[test_idx].copy())
    return df, train_df, val_df, test_df

def normalize_splits(train_df, val_df, test_df, feature_cols):
    mins   = train_df[feature_cols].min()
    ranges = (train_df[feature_cols].max() - mins).replace(0, 1)

    for split in (train_df, val_df, test_df):
        split[feature_cols] = 2 * (split[feature_cols] - mins) / ranges - 1

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def process_train_batch(train_df, state_cols, action_cols):
    batch = train_df.sample(n=BATCH_SIZE)
    S, A, R, NS, D = [], [], [], [], []

    for i in batch.index:
        s = train_df.loc[i, state_cols].values.astype(np.float32)
        a = train_df.loc[i, action_cols].values.astype(np.float32)
        r = train_df.loc[i, 'r:SOFA']
        if CLIP_REWARD:
            r = np.clip(r, -12.5, -2.5)

        if i != train_df.index[-1] and train_df.loc[i, 'traj'] == train_df.loc[i + 1, 'traj']:
            ns = train_df.loc[i + 1, state_cols].values.astype(np.float32)
            d = 0.0
        else:
            ns = np.zeros_like(s, dtype=np.float32)
            d = 1.0

        S.append(s)
        A.append(a)
        R.append(r)
        NS.append(ns)
        D.append(d)

    return (
        torch.tensor(S, dtype=torch.float32, device=DEVICE),
        torch.tensor(A, dtype=torch.float32, device=DEVICE),
        torch.tensor(R, dtype=torch.float32, device=DEVICE).unsqueeze(1),
        torch.tensor(NS, dtype=torch.float32, device=DEVICE),
        torch.tensor(D, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    )

# def process_eval_batch(eval_df, state_cols, action_cols):
#     for i in range(0, len(eval_df), BATCH_SIZE):
#         batch = eval_df.iloc[i:i + BATCH_SIZE]
#         S, A, R, NS, D = [], [], [], [], []

#         for idx in batch.index:
#             s = eval_df.loc[idx, state_cols].values.astype(np.float32)
#             a = eval_df.loc[idx, action_cols].values.astype(np.float32)
#             r = eval_df.loc[idx, 'r:SOFA']
#             if CLIP_REWARD:
#                 r = np.clip(r, -12.5, -2.5)

#             if idx != eval_df.index[-1] and eval_df.loc[idx, 'traj'] == eval_df.loc[idx + 1, 'traj']:
#                 ns = eval_df.loc[idx + 1, state_cols].values.astype(np.float32)
#                 d = 0.0
#             else:
#                 ns = np.zeros_like(s, dtype=np.float32)
#                 d = 1.0

#             S.append(s)
#             A.append(a)
#             R.append(r)
#             NS.append(ns)
#             D.append(d)

#         yield (
#             torch.tensor(S, dtype=torch.float32, device=DEVICE),
#             torch.tensor(A, dtype=torch.float32, device=DEVICE),
#             torch.tensor(R, dtype=torch.float32, device=DEVICE).unsqueeze(1),
#             torch.tensor(NS, dtype=torch.float32, device=DEVICE),
#             torch.tensor(D, dtype=torch.float32, device=DEVICE).unsqueeze(1)
#         )

# def do_eval(eval_df, actor, critic, state_cols, action_cols):
#     actor.eval(); critic.eval()
#     total_q, total_err, total_n = 0, 0, 0
#     with torch.no_grad():
#         for S, A, R, NS, D in process_eval_batch(eval_df, state_cols, action_cols):
#             next_A = actor(NS)
#             target_Q = critic(NS, next_A)
#             y = R + GAMMA * target_Q * (1 - D)
#             Q = critic(S, A)
#             total_err += F.l1_loss(Q, y, reduction='sum').item()
#             total_q += Q.sum().item()
#             total_n += len(S)
#     actor.train(); critic.train()
#     return total_q / total_n, total_err / total_n

# ─── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train(seed):
    ckpt = os.path.join(CHECKPOINT_DIR, f"seed_{seed}"); os.makedirs(ckpt, exist_ok=True)
    lgdir = os.path.join(LOG_DIR, f"seed_{seed}"); os.makedirs(lgdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(lgdir, f"training_{ts}.log")
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger()
    print(f"Logging to {logfile.replace(os.sep, '/')}")

    df, train_df, val_df, test_df = load_and_split_data()
    state_cols = [c for c in df.columns if c.startswith('o:')]
    action_cols = ['a:max_dose_vaso', 'a:input_4hourly']
    train_df, val_df, test_df = normalize_splits(train_df, val_df, test_df, state_cols + action_cols)

    # Apply sampling
    train_df = sample_trajectory_level(train_df, traj_col='traj', action_cols=action_cols,
                                       n_bins=5, strategy='under_over', K=1.0)

    actor = Actor(len(state_cols), len(action_cols)).to(DEVICE)
    critic = Critic(len(state_cols), len(action_cols)).to(DEVICE)
    target_actor = Actor(len(state_cols), len(action_cols)).to(DEVICE)
    target_critic = Critic(len(state_cols), len(action_cols)).to(DEVICE)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    opt_actor = optim.Adam(actor.parameters(), lr=1e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-4)

    for epoch in tqdm(range(1, NUM_EPOCHS+1), desc="CQL_TS Epochs"):
        S, A, R, NS, D = process_train_batch(train_df, state_cols, action_cols)

        with torch.no_grad():
            target_Q = target_critic(NS, target_actor(NS))
            y = R + GAMMA * target_Q * (1 - D)

        Q = critic(S, A)
        loss_mse = F.mse_loss(Q, y)

        candidate_A = actor(S).unsqueeze(1) + torch.randn(BATCH_SIZE, NUM_CANDIDATES, len(action_cols), device=DEVICE) * NOISE_STD
        S_exp = S.unsqueeze(1).expand(-1, NUM_CANDIDATES, -1)
        flat_S = S_exp.reshape(-1, len(state_cols))
        flat_A = candidate_A.reshape(-1, len(action_cols))
        Q_cand = critic(flat_S, flat_A).view(BATCH_SIZE, NUM_CANDIDATES)
        logsum_q = torch.logsumexp(Q_cand, dim=1)
        cql_penalty = (logsum_q - Q.squeeze(1)).mean()

        loss_critic = loss_mse + CQL_ALPHA * cql_penalty
        opt_critic.zero_grad(); loss_critic.backward(); opt_critic.step()

        loss_actor = -critic(S, actor(S)).mean()
        opt_actor.zero_grad(); loss_actor.backward(); opt_actor.step()

        soft_update(actor, target_actor, TAU)
        soft_update(critic, target_critic, TAU)

        if epoch % SAVE_INTERVAL == 0:
            torch.save(actor.state_dict(), os.path.join(ckpt, f"actor_ep{epoch}.pt"))
            torch.save(critic.state_dict(), os.path.join(ckpt, f"critic_ep{epoch}.pt"))
            logger.info(f"Epoch {epoch}: CriticLoss={loss_critic.item():.4f}, ActorLoss={loss_actor.item():.4f}, CQLPenalty={cql_penalty.item():.4f}")

        # if epoch % EVAL_INTERVAL == 0:
        #     val_q, val_err = do_eval(val_df, actor, critic, state_cols, action_cols)
        #     logger.info(f"Validation - AvgQ={val_q:.4f}, MAE={val_err:.4f}")

    logger.info("Training complete.")
    
# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
def load_actor_model(weight_path, state_dim, action_dim):
    actor = Actor(state_dim, action_dim).to(DEVICE)
    state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor

def load_critic_model(weight_path, state_dim, action_dim):
    critic = Critic(state_dim, action_dim).to(DEVICE)
    state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    critic.load_state_dict(state_dict)
    critic.eval()
    return critic

# ─── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    seed = load_seed(args.seeds_file, args.seed_index)
    print(f"Using seed #{args.seed_index}: {seed}")
    set_global_seed(seed)
    train(seed=seed)
