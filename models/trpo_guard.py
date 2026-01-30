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
CLIP_REWARD     = True      # Whether to clip rewards
KDE_PENALTY     = -25       # Penalty for low-density
NUM_EPOCHS      = 20000
MAX_TRAJ_LEN    = 20
BATCH_SIZE      = 32
HIDDEN_SIZE     = 128
LR_VALUE        = 1e-3
MAX_KL          = 0.01
DAMPING         = 1.0
SAVE_INTERVAL   = 100
CHECKPOINT_DIR  = "../checkpoints/trpo_guard"
LOG_DIR         = "../logs/trpo_guard"

# ─── DEVICE SETUP ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─── SEED UTILITIES ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TRPO with KDE‐based guard on sepsis data"
    )
    parser.add_argument(
        "--seed-index", "-i",
        type=int, default=0,
        help="Index into the seeds list in JSON file"
    )
    parser.add_argument(
        "--seeds-file", "-s",
        type=str, default="../seeds.json",
        help="Path to JSON file containing seeds"
    )
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

# ─── DATA LOADING & PREPROCESSING ─────────────────────────────────────────────
def load_and_split_data():
    df = pd.read_csv('../sepsis_final_RAW_continuous_13.csv')
    train_idx = np.load('../train_indices.npy')
    val_idx   = np.load('../val_indices.npy')
    test_idx  = np.load('../test_indices.npy')

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df   = df.loc[val_idx].reset_index(drop=True)
    test_df  = df.loc[test_idx].reset_index(drop=True)
    return df, train_df, val_df, test_df


def normalize_features(df, train_df, val_df, test_df, feature_cols):
    mins   = train_df[feature_cols].min()
    maxs   = train_df[feature_cols].max()
    ranges = (maxs - mins).replace(0, 1)
    for split in (train_df, val_df, test_df):
        split[feature_cols] = 2 * (split[feature_cols] - mins) / ranges - 1
    df.loc[train_df.index, feature_cols] = train_df[feature_cols]
    df.loc[val_df.index,   feature_cols] = val_df[feature_cols]
    df.loc[test_df.index,  feature_cols] = test_df[feature_cols]
    return df


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
    r = b.clone(); p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        Avp_p = Avp(p)
        alpha = rdotr / torch.dot(p, Avp_p)
        x += alpha * p; r -= alpha * Avp_p
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol: break
        beta = new_rdotr / rdotr; p = r + beta * p; rdotr = new_rdotr
    return x

def surrogate_loss(policy, old_policy, states, actions, advantages):
    dist    = policy(states)
    old_dist= old_policy(states)
    ratio   = torch.exp(dist.log_prob(actions) - old_dist.log_prob(actions))
    return -(ratio * advantages).mean()

def kl_divergence(policy, old_policy, states):
    return torch.distributions.kl.kl_divergence(old_policy(states), policy(states)).mean()

def trpo_step(policy, old_policy, states, actions, advantages):
    loss_val = surrogate_loss(policy, old_policy, states, actions, advantages)
    g = torch.clamp(flat_grad(loss_val, policy), -1e7, 1e7)
    def Fvp(v):
        kl = kl_divergence(policy, old_policy, states)
        flat_kl = torch.cat([g.view(-1) for g in torch.autograd.grad(kl, policy.parameters(), create_graph=True)])
        kl_v = (flat_kl * v).sum()
        flat2= torch.cat([g.contiguous().view(-1) for g in torch.autograd.grad(kl_v, policy.parameters())])
        return flat2 + DAMPING * v
    step_dir = conjugate_gradients(Fvp, -g)
    shs      = 0.5 * torch.dot(step_dir, Fvp(step_dir))
    full    = step_dir / torch.sqrt(shs / MAX_KL)
    old_params   = flat_params(policy)
    for i in range(30):
        new_p = old_params + (0.8**i) * full
        set_params(policy, new_p)
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

# ─── SIMULATION GUARD ─────────────────────────────────────────────────────────
def simulate_step(state, action, nn_model, transitions, df, step_count):
    query = np.concatenate([state, action]).reshape(1, -1)
    _, idx = nn_model.kneighbors(query)
    next_state = transitions[idx[0][0]]
    df_row = df.iloc[idx[0][0]]
    done = (step_count >= MAX_TRAJ_LEN) or (abs(df_row["r:reward"]) == 1)
    return next_state, idx[0][0], done
    
def simulate_episode_guard(actor, init_state, nn_model, transitions,
                           df, kd_params, out_sa, out_s, target_vals):
    sigma = kd_params['sigma']; thr_kd = kd_params['thr_kd']
    states, actions, rewards, dones, next_states = [], [], [], [], []
    state, step, update_flag = init_state.copy(), 0, True
    while step < MAX_TRAJ_LEN:
        if update_flag:
            s_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad(): dist = actor(s_t); action = dist.sample().cpu().numpy()[0]
            next_state, idx, done = simulate_step(state, action, nn_model, transitions, df, step)
            states.append(state); actions.append(action); next_states.append(next_state)
            sa = np.concatenate([state, action])
            diffs = target_vals - sa
            norm2 = np.sum(diffs ** 2, axis=1)
            kern = np.exp(-0.5 * norm2 / sigma**2) / (sigma * np.sqrt(2 * np.pi))
            density = kern.mean()
            if density < thr_kd:
                reward = KDE_PENALTY; update_flag = False
            else:
                if out_sa[idx]:
                    reward = KDE_PENALTY; update_flag = False
                else:
                    reward = df.iloc[idx]["r:SOFA"]
                    if CLIP_REWARD:
                        reward = np.clip(reward, -12.5, -2.5)
                    state = next_state.copy()
        else:
            reward = KDE_PENALTY; next_state = state.copy(); done = False
            states.append(state); actions.append(action); next_states.append(next_state)
        rewards.append(reward); dones.append(done)
        if done: break
        step += 1
    return {"states": np.array(states), "actions": np.array(actions), 
            "rewards": np.array(rewards), "dones": np.array(dones),
            "next_states": np.array(next_states)
    }

# ─── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train(seed):
    ckpt = os.path.join(CHECKPOINT_DIR, f"seed_{seed}"); os.makedirs(ckpt,exist_ok=True)
    lgdir= os.path.join(LOG_DIR, f"seed_{seed}"); os.makedirs(lgdir,exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(lgdir, f"training_{ts}.log")
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    logger=logging.getLogger()
    print(f"Logging to {logfile.replace(os.sep, '/')}")

    df, tr, va, te = load_and_split_data()
    state_cols=[c for c in df.columns if c.startswith('o:')]
    action_cols=['a:max_dose_vaso','a:input_4hourly']
    df=normalize_features(df, tr, va, te, state_cols+action_cols)
    init_states=get_initial_states(tr,state_cols)
    lookup=joblib.load('../lookup_data.pkl')
    nn_model, transitions = lookup['nn_model'], lookup['transitions']
    kd_params = np.load('../sepsis_final_RAW_continuous_13_kd_params_state_act.npy',allow_pickle=True).item()
    out_sa = np.load('../sepsis_final_RAW_continuous_13_all_outlier_labels_state_act.npy')
    out_s  = np.load('../sepsis_final_RAW_continuous_13_all_outlier_labels_state.npy')

    # precompute target values for KDE
    feature_vals = df.filter(regex='^o:').values
    action_vals  = df[['a:max_dose_vaso','a:input_4hourly']].values
    target_vals  = np.concatenate([feature_vals, action_vals], axis=1)

    actor = GaussianPolicy(len(state_cols), len(action_cols)).to(DEVICE)
    value_fn = ValueNet(len(state_cols)).to(DEVICE)
    opt_value = optim.Adam(value_fn.parameters(), lr=LR_VALUE)

    for epoch in tqdm(range(1, NUM_EPOCHS+1), desc="TRPO with Guaridan Epochs"):      
        idxs = np.random.choice(len(init_states), BATCH_SIZE, replace=False)
        batch = [simulate_episode_guard(actor, init_states[i], nn_model, 
                                        transitions, df, kd_params, out_sa, out_s, target_vals)
                 for i in idxs]
        S  = np.concatenate([b['states'] for b in batch])
        A  = np.concatenate([b['actions'] for b in batch])
        R  = np.concatenate([b['rewards'] for b in batch])
        D  = np.concatenate([b['dones'] for b in batch])
        S2 = np.concatenate([b['next_states'] for b in batch])

        S_t  = torch.tensor(S, dtype=torch.float32, device=DEVICE)
        A_t  = torch.tensor(A, dtype=torch.float32, device=DEVICE)
        R_t  = torch.tensor(R, dtype=torch.float32, device=DEVICE)
        D_t  = torch.tensor(D, dtype=torch.float32, device=DEVICE)
        S2_t = torch.tensor(S2, dtype=torch.float32, device=DEVICE)

        V   = value_fn(S_t); V2 = value_fn(S2_t)
        adv = compute_gae(R_t, V, V2, D_t)
        adv = (adv-adv.mean())/(adv.std()+1e-9)
        returns = adv + V.detach()

        for _ in range(2):
            loss_v = (value_fn(S_t) - returns).pow(2).mean()
            opt_value.zero_grad(); loss_v.backward(); opt_value.step()

        old_actor = deepcopy(actor)
        trpo_step(actor, old_actor, S_t, A_t, adv)

        kl   = kl_divergence(actor, old_actor, S_t).item()
        ar   = R_t.mean().item(); vl = loss_v.item()
        logger.info(f"Epoch {epoch}: AvgR={ar:.4f}, KL={kl:.6f}, VLoss={vl:.6f}")

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
