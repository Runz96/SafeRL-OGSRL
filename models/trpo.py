#!/usr/bin/env python3
import os
import json
import argparse
import random
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from copy import deepcopy
from tqdm import tqdm

# ─── HYPERPARAMETERS ───────────────────────────────────────────────────────────
GAMMA           = 0.99
LAMBDA          = 0.95
CLIP_REWARD     = True
NUM_EPOCHS      = 20000
MAX_TRAJ_LEN    = 20
BATCH_SIZE      = 32
HIDDEN_SIZE     = 128
LR_VALUE        = 1e-3
MAX_KL          = 0.01
DAMPING         = 1.0
SAVE_INTERVAL   = 100
CHECKPOINT_DIR  = "../checkpoints/trpo"
LOG_DIR         = "../logs/trpo"

# ─── DEVICE SETUP ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─── SEED UTILITIES ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TRPO on sepsis data with reproducible seeding"
    )
    parser.add_argument(
        "--seed-index", "-i",
        type=int, default=0,
        help="Index of the seed in seeds.json"
    )
    parser.add_argument(
        "--seeds-file", "-s",
        type=str, default="../seeds.json",
        help="Path to the JSON file containing seeds"
    )
    return parser.parse_args()


def load_seed(seeds_file, idx):
    with open(seeds_file, "r") as f:
        data = json.load(f)
    seeds = data.get("seeds", [])
    if not seeds:
        raise ValueError(f"No 'seeds' found in {seeds_file}")
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

# ─── DATA LOADING & PREPROCESSING ─────────────────────────────────────────────
def load_and_split_data():
    df = pd.read_csv('../sepsis_final_RAW_continuous_13.csv')
    train_idx = np.load('../train_indices.npy')
    val_idx   = np.load('../val_indices.npy')
    test_idx  = np.load('../test_indices.npy')

    train_df = df.loc[train_idx].copy()
    val_df   = df.loc[val_idx].copy()
    test_df  = df.loc[test_idx].copy()
    return df, train_df, val_df, test_df

def normalize_splits(train_df, val_df, test_df, feature_cols):
    mins   = train_df[feature_cols].min()
    ranges = (train_df[feature_cols].max() - mins).replace(0, 1)

    for split in (train_df, val_df, test_df):
        split[feature_cols] = 2 * (split[feature_cols] - mins) / ranges - 1

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def get_initial_states(df, state_cols):
    return df[df['step'] == 0][state_cols].values

# ─── MODEL DEFINITIONS ─────────────────────────────────────────────────────────
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean(h)
        std = torch.exp(self.log_std)
        cov = torch.diag_embed(std ** 2)
        return MultivariateNormal(mean, cov)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ─── TRPO HELPERS ──────────────────────────────────────────────────────────────
def flat_grad(loss, model, retain_graph=False):
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=retain_graph)
    return torch.cat([g.view(-1) for g in grads])

def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params(model, vector):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vector[offset:offset+numel].view_as(p))
        offset += numel

def conjugate_gradients(Avp, b, nsteps=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        Avp_p = Avp(p)
        alpha = rdotr / torch.dot(p, Avp_p)
        x += alpha * p
        r -= alpha * Avp_p
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

def surrogate_loss(policy, old_policy, states, actions, advantages):
    dist     = policy(states)
    old_dist = old_policy(states)
    ratio    = torch.exp(dist.log_prob(actions) - old_dist.log_prob(actions))
    return -(ratio * advantages).mean()

def kl_divergence(policy, old_policy, states):
    dist     = policy(states)
    old_dist = old_policy(states)
    return torch.distributions.kl.kl_divergence(old_dist, dist).mean()

def trpo_step(policy, old_policy, states, actions, advantages):
    loss_val = surrogate_loss(policy, old_policy, states, actions, advantages)
    g = flat_grad(loss_val, policy)
    g = torch.clamp(g, -1e7, 1e7)
    def Fvp(v):
        kl = kl_divergence(policy, old_policy, states)
        grads_kl = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
        flat_kl = torch.cat([g.view(-1) for g in grads_kl])
        kl_v = (flat_kl * v).sum()
        grads2 = torch.autograd.grad(kl_v, policy.parameters())
        flat2 = torch.cat([g.contiguous().view(-1) for g in grads2])
        return flat2 + DAMPING * v
    step_dir = conjugate_gradients(Fvp, -g)
    shs      = 0.5 * torch.dot(step_dir, Fvp(step_dir))
    lm       = torch.sqrt(shs / MAX_KL)
    full_step= step_dir / lm
    old_params = flat_params(policy)
    for i in range(30):
        new_params = old_params + (0.8 ** i) * full_step
        set_params(policy, new_params)
        if surrogate_loss(policy, old_policy, states, actions, advantages) < loss_val and kl_divergence(policy, old_policy, states) < MAX_KL:
            return
    set_params(policy, old_params)

def compute_gae(rewards, values, next_values, dones):
    deltas = rewards + GAMMA * next_values * (1 - dones) - values
    advs, gae = [], 0.0
    delta_np = deltas.detach().cpu().numpy()
    done_np  = dones.detach().cpu().numpy()
    for delta, d in zip(reversed(delta_np), reversed(done_np)):
        gae = delta + GAMMA * LAMBDA * gae * (1 - d)
        advs.insert(0, gae)
    return torch.tensor(advs, dtype=torch.float32, device=values.device)

# ─── EPISODE SIMULATION ────────────────────────────────────────────────────────
def simulate_step(state, action, nn_model, transitions, df, step_count):
    query = np.concatenate([state, action]).reshape(1, -1)
    _, idx = nn_model.kneighbors(query)
    next_state = transitions[idx[0][0]]
    df_row = df.iloc[idx[0][0]]
    done = (step_count >= MAX_TRAJ_LEN) or (abs(df_row["r:reward"]) == 1)
    return next_state, idx[0][0], done

def simulate_episode(actor, init_state, nn_model, transitions, df):
    states, actions, rewards, dones, next_states = [], [], [], [], []
    state, step = init_state.copy(), 0
    while step < MAX_TRAJ_LEN:
        s_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            dist = actor(s_t)
            action = dist.sample().cpu().numpy()[0]
        nxt, idx, done = simulate_step(state, action, nn_model, transitions, df, step)
        reward = df.iloc[idx]["r:SOFA"]
        if CLIP_REWARD:
            reward = np.clip(reward, -12.5, -2.5)
        states.append(state); actions.append(action); rewards.append(reward)
        dones.append(done); next_states.append(nxt)
        state, step = nxt.copy(), step + 1
        if done:
            break
    return {"states": np.array(states), "actions": np.array(actions), "rewards": np.array(rewards), "dones": np.array(dones), "next_states": np.array(next_states)}

# ─── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train(seed):
    # create directories
    ckpt = os.path.join(CHECKPOINT_DIR, f"seed_{seed}"); os.makedirs(ckpt,exist_ok=True)
    lgdir= os.path.join(LOG_DIR, f"seed_{seed}"); os.makedirs(lgdir,exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(lgdir, f"training_{ts}.log")
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    logger=logging.getLogger()
    print(f"Logging to {logfile.replace(os.sep, '/')}")

    df, train_df, val_df, test_df = load_and_split_data()
    state_cols  = [c for c in df.columns if c.startswith("o:")]
    action_cols = ['a:max_dose_vaso', 'a:input_4hourly']
    all_feats   = state_cols + action_cols
    train_df, val_df, test_df = normalize_splits(train_df, val_df, test_df, all_feats)
    init_states = get_initial_states(train_df, state_cols)

    lookup  = joblib.load('../lookup_data.pkl')
    nn_model = lookup['nn_model']; transitions = lookup['transitions']

    actor = GaussianPolicy(len(state_cols), len(action_cols)).to(DEVICE)
    value_fn  = ValueNet(len(state_cols)).to(DEVICE)
    opt_value = optim.Adam(value_fn.parameters(), lr=LR_VALUE)

    for epoch in tqdm(range(1, NUM_EPOCHS+1), desc="TRPO Epochs"):
        idxs = np.random.choice(len(init_states), BATCH_SIZE, replace=False)
        batch = [simulate_episode(actor, init_states[i], nn_model, transitions, df) for i in idxs]
        S  = np.concatenate([b['states'] for b in batch])
        A  = np.concatenate([b['actions'] for b in batch])
        R  = np.concatenate([b['rewards'] for b in batch])
        D  = np.concatenate([b['dones']   for b in batch])
        S2 = np.concatenate([b['next_states'] for b in batch])

        S_t  = torch.tensor(S, dtype=torch.float32, device=DEVICE)
        A_t  = torch.tensor(A, dtype=torch.float32, device=DEVICE)
        R_t  = torch.tensor(R, dtype=torch.float32, device=DEVICE)
        D_t  = torch.tensor(D, dtype=torch.float32, device=DEVICE)
        S2_t = torch.tensor(S2, dtype=torch.float32, device=DEVICE)

        V     = value_fn(S_t)
        V2    = value_fn(S2_t)
        advs  = compute_gae(R_t, V, V2, D_t)
        advs  = (advs - advs.mean()) / (advs.std()+1e-9)
        returns = advs + V.detach()

        for _ in range(2):
            loss_v = (value_fn(S_t) - returns).pow(2).mean()
            opt_value.zero_grad(); loss_v.backward(); opt_value.step()

        old_actor = deepcopy(actor)
        trpo_step(actor, old_actor, S_t, A_t, advs)

        kl   = kl_divergence(actor, old_actor, S_t).item()
        avgr = R_t.mean().item(); vloss = loss_v.item()
        logger.info(f"Epoch {epoch}: AvgR={avgr:.4f}, KL={kl:.6f}, VLoss={vloss:.6f}")

        if epoch % SAVE_INTERVAL == 0:
            torch.save(actor.state_dict(),
                       os.path.join(ckpt, f"actor_ep{epoch}.pt"))
            torch.save(value_fn.state_dict(),
                       os.path.join(ckpt, f"value_ep{epoch}.pt"))

    logger.info("Training complete.")

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────    
def load_actor_model(weight_path, state_dim, action_dim):
    actor = GaussianPolicy(state_dim, action_dim).to(DEVICE)
    state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor

if __name__ == "__main__":
    args = parse_args()
    seed = load_seed(args.seeds_file, args.seed_index)
    print(f"Using seed #{args.seed_index}: {seed}")
    set_global_seed(seed)
    train(seed)