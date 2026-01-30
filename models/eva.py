import importlib
import argparse
import numpy as np
import pandas as pd
import joblib
import os
from eva_func import run_and_save_offline_rollouts
import copy

def parse_args():
    # ─── Parse Arguments ────────────────────────────────────────────────────────  
    parser = argparse.ArgumentParser(description="Run offline rollouts for different experiments")
    parser.add_argument("--exp", type=str, required=True, help="Name of the experiment (e.g., ddpg_cql, trpo_guard)")
    parser.add_argument("--seed", type=int, default=40, help="Seed used during training")
    parser.add_argument("--data_split", type=str, default="train", help="Data split to use (train, test)")
    return parser.parse_args()

def run(exp, max_steps, onpolicy, data_split, path_states, epoch, seed_value):

    # ─── Load model-specific modules dynamically ─────────────────────────────────
    try:
        module = importlib.import_module(exp)
        load_actor_model = getattr(module, "load_actor_model")
        DEVICE = getattr(module, "DEVICE")
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load model-specific code for {exp}: {e}")

    # ─── Load data ───────────────────────────────────────────────────────────────
    df = pd.read_csv('../sepsis_final_RAW_continuous_13.csv')
    state_features = [c for c in df.columns if c.startswith('o:')]
    action_features = ['a:max_dose_vaso', 'a:input_4hourly']
    target_cols = state_features + action_features

    state_dim = len(state_features)
    action_dim = len(action_features)

    train_idx = np.load(f'../{seed_value}/train_indices.npy')
    val_idx = np.load(f'../{seed_value}/val_indices.npy')
    test_idx = np.load('../{seed_value}/test_indices.npy')

    train_df = copy.deepcopy(df.loc[train_idx].reset_index(drop=True))
    val_df = copy.deepcopy(df.loc[val_idx].reset_index(drop=True))
    test_df = copy.deepcopy(df.loc[test_idx].reset_index(drop=True))

    mins = train_df[target_cols].min()
    maxs = train_df[target_cols].max()
    ranges = (maxs - mins).replace(0, 1)

    for split_df in [train_df, val_df, test_df]:
        split_df[target_cols] = 2 * (split_df[target_cols] - mins) / ranges - 1

    # Load initial states
    input_states_indices_ini = np.load(path_states)
    if data_split == 'train':
        input_states_ini = train_df.loc[input_states_indices_ini, :][state_features].values.astype(np.float32)
    elif data_split == 'test':
        input_states_ini = test_df.loc[input_states_indices_ini, :][state_features].values.astype(np.float32)
    else:
        raise ValueError("data_split must be either 'train' or 'test'")

    # outlier_labels = np.load('../sepsis_final_RAW_continuous_13_all_outlier_labels_state_act.npy')
    lookup = joblib.load(f'../{seed_value}/lookup_data.pkl')
    nn_model, transitions = lookup['nn_model'], lookup['transitions']

    # ─── Load actor model ────────────────────────────────────────────────────────
    path_checkpoints_base = "../checkpoints"
    path_rollouts_base = "../results/rollouts_data"
    seed_folder = f"seed_{seed_value}"

    path_checkpoint = os.path.join(path_checkpoints_base, exp, seed_folder, f"actor_ep{epoch}.pt")
    actor = load_actor_model(path_checkpoint, state_dim, action_dim)

    # ─── Run and save offline rollouts ───────────────────────────────────────────
    save_dir = os.path.join(path_rollouts_base, exp, seed_folder, f"epoch_{epoch}")
    suffix = path_states.split('_')[-1].split('.')[0]
    save_filename_prefix = f"{data_split}_offline_rollouts_{suffix}"
    train_data = train_df[target_cols].values.astype(np.float32)

    run_and_save_offline_rollouts(
        input_states_ini,
        actor,
        train_data,
        df,
        nn_model,
        transitions,
        onpolicy=onpolicy,
        max_steps=max_steps,
        device=DEVICE,
        save_dir=save_dir,
        save_filename_prefix=save_filename_prefix
    )

if __name__ == "__main__":
    args = parse_args()
    exp = args.exp
    seed_value = args.seed
    data_split = args.data_split
    if exp in ['ddpg_cql', 'ddpg_cql_ts', 'ddpg_cql_guard', 'ddpg_cql_guard_ts']:
        onpolicy = False
    elif exp in ['trpo', 'trpo_guard','cpo', 'cpo_guard']:
        onpolicy = True
    else:
        raise ValueError(f"Unknown experiment type: {exp}")
    path_states_ini = f'../sepsis_final_RAW_continuous_13_{data_split}_indices_ini.npy'
    path_states_plots = [
        f"../sepsis_final_RAW_continuous_13_{data_split}_indices_plot_{i}.npy" 
        for i in range(5)
    ]
    # ─── Run the script with different max_steps values ───────────────────────────
    for epoch in [20000, 2000, 10000, 5000, 500]:
        for path_states_plot in path_states_plots:
            run(exp=exp, max_steps=1, onpolicy=onpolicy, data_split=data_split, path_states=path_states_plot, epoch=epoch, seed_value=seed_value)
        run(exp=exp, max_steps=20, onpolicy=onpolicy, data_split=data_split, path_states=path_states_ini, epoch=epoch, seed_value=seed_value)