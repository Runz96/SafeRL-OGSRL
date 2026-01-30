import os
import numpy as np
import torch
from tqdm import tqdm

def simulate_step(state, action, df, nn_model, transitions):
    """Simulate one environment step based on nearest neighbor lookup."""
    query = np.concatenate([state, action]).reshape(1, -1)
    _, indices = nn_model.kneighbors(query)
    idx = indices[0][0]

    next_state = transitions[idx]
    reward = df.iloc[idx]["r:reward"]
    done = abs(reward) == 1  # Done if absolute reward is 1 (e.g., terminal state)

    return next_state, idx, done

def offline_rollout(actor, initial_state, train_data, kd_params, df, nn_model, transitions,
                    onpolicy, max_steps, traj_id, device="cpu"):
    """Perform a single offline rollout starting from a given initial state."""
    states, actions, rewards, ood_flags, idxs, dones, next_states, traj_ids, steps = [], [], [], [], [], [], [], [], []
    sigma = kd_params['sigma']; thr_kd = kd_params['thr_kd']
    state = initial_state.copy()

    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            if onpolicy:
                action_dist = actor(state_tensor)
                action = action_dist.sample().cpu().numpy().squeeze(0)
            else:
                action = actor(state_tensor).cpu().numpy().squeeze(0)

        next_state, idx, done = simulate_step(state, action, df, nn_model, transitions)

        kde_sa = np.concatenate([state, action])
        diff = train_data - kde_sa
        norm_sq = np.sum(diff**2, axis=1)
        kernel_vals = np.exp(-0.5 * norm_sq / sigma**2) / (sigma * np.sqrt(2 * np.pi))
        densities = np.mean(kernel_vals)
        ood_flag = int(densities < thr_kd)

        # Record transition
        states.append(state)
        actions.append(action)
        rewards.append(df.iloc[idx]["r:SOFA"])
        ood_flags.append(ood_flag)
        idxs.append(idx)
        dones.append(done)
        next_states.append(next_state)
        traj_ids.append(traj_id)
        steps.append(step)

        if done:
            break  # End rollout if terminal state is reached

        state = next_state.copy()

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "ood_status": np.array(ood_flags),
        "indexes": np.array(idxs),
        "dones": np.array(dones),
        "next_states": np.array(next_states),
        "traj_ids": np.array(traj_ids),
        "steps": np.array(steps)
    }

def run_and_save_offline_rollouts(train_states_ini, actor, train_data, df, nn_model,
                                  transitions, onpolicy=True, max_steps=20, device="cpu",
                                  save_dir="../results/rollout_data",
                                  save_filename_prefix="offline_rollouts"):
    """Run offline rollouts from a list of initial states and save the results."""
    os.makedirs(save_dir, exist_ok=True)
    
    kd_params = np.load('../sepsis_final_RAW_continuous_13_kd_params_state_act.npy', allow_pickle=True).item()

    rollout_data = {
        "states": [],
        "actions": [],
        "rewards": [],
        "ood_status": [],
        "indexes": [],
        "dones": [],
        "next_states": [],
        "traj_ids": [],
        "steps": []
    }

    # Run rollouts
    for idx, initial_state in enumerate(tqdm(train_states_ini, desc="Running offline rollouts")):
        rollout = offline_rollout(
            actor=actor,
            initial_state=initial_state,
            train_data=train_data,
            kd_params=kd_params,
            df=df,
            # outlier_label=outlier_label,
            nn_model=nn_model,
            transitions=transitions,
            onpolicy=onpolicy,
            max_steps=max_steps,
            traj_id=idx,
            device=device
        )

        for key in rollout_data:
            data = rollout[key]
            if data.ndim == 1:
                data = data[:, None]
            rollout_data[key].append(data)

    # Stack collected transitions
    for key in rollout_data:
        rollout_data[key] = np.vstack(rollout_data[key])

    # Save as compressed file
    save_path = os.path.join(save_dir, f"{save_filename_prefix}.npz")
    np.savez_compressed(save_path, **rollout_data)

    print(f"Saved {rollout_data['states'].shape[0]} transitions to {save_path}")

    return save_path