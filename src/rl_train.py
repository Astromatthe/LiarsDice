import random
from collections import deque
import torch
import numpy as np
from src.rl_env import LiarsDiceEnv, get_legal_action_indices
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import NUM_ACTIONS, MAX_STATE_DIM, MAX_PLAYERS
import matplotlib.pyplot as plt
import os
from typing import Dict, Any
import importlib
from src.dqn_model import DQN



def select_action(policy_net, state_vec, epsilon):
    """
    Epsilon-greedy action selection with illegal action masking.
    
    Arguments:
        policy_net  : DQN model mapping state → Q-values (one per action index)
        state       : encoded RL state (numpy array or list)
        epsilon     : float in [0,1]
        
    Returns:
        action_index (int) — a legal action index
    """

    legal_actions = get_legal_action_indices(state_vec)

    if len(legal_actions) == 0:
        return None

    # Exploration case
    if random.random() < epsilon:
        return int(random.choice(legal_actions))
    
    # Exploitation
    # Convert state to torch tensor [1, state_dim]
    state_t = torch.tensor(state_vec, dtype = torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(state_t).squeeze(0)

    # illegal actions initialized to -inf
    masked_q = torch.full_like(q_values, float("-inf"))

    legal_t = torch.tensor(legal_actions, dtype=torch.long)

    masked_q[legal_t] = q_values[legal_t]

    # Select highest scoring legal action
    best_action = int(torch.argmax(masked_q).item())

    return best_action


def optimize_model(memory, batch_size, policy_net, target_net, optimizer, gamma):
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    s_batch, a_batch, r_batch, sp_batch, done_batch = zip(*batch)

    state_batch = torch.tensor(np.array(s_batch), dtype=torch.float32)
    action_batch = torch.tensor(a_batch, dtype=torch.long).unsqueeze(1)
    reward_batch = torch.tensor(r_batch, dtype=torch.float32)
    next_state_batch = torch.tensor(np.array(sp_batch), dtype=torch.float32)
    done_batch = torch.tensor(done_batch, dtype=torch.float32)

    q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    with torch.no_grad():
        q_next_all = target_net(next_state_batch)

        max_next_q_list = []

        for i in range(batch_size):
            sp_vec = sp_batch[i]
            legal_next = get_legal_action_indices(sp_vec)

            if len(legal_next) == 0:
                max_next_q_list.append(0.0)
                continue

            q_next_i = q_next_all[i]
            masked_q_i = torch.full_like(q_next_i, float("-inf"))
            masked_q_i[legal_next] = q_next_i[legal_next]

            max_next_q = torch.max(masked_q_i).item()
            max_next_q_list.append(max_next_q)
        
        max_next_q_values = torch.tensor(max_next_q_list, dtype = torch.float32)

        target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)
        

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def deserialize_roster(roster_dict):
    """
    Accepts:
        {"RiskyBot": 1}
        {"DQNBot": {"count": 1, "model": "x.pt"}}
        {"RiskyBot": 1, "DQNBot": {"count": 1, "model": "x.pt"}}

    Returns canonical form:
        {BotClass: {"count": int, "model": Optional[str]}}
    """

    bots_mod = importlib.import_module("src.bots")
    out = {}

    for name, val in roster_dict.items():
        cls = getattr(bots_mod, name, None)
        if cls is None:
            raise ValueError(f"Unknown bot class: {name}")

        # Special-case: DQNBot uses {"count": n, "model": file}
        if name == "DQNBot":
            if not isinstance(val, dict):
                raise ValueError("DQNBot roster entry must be dict {count, model}")
            count = int(val["count"])
            model = val.get("model")
            out[cls] = {"count": count, "model": model}

        # All other bots use integer counts
        else:
            if not isinstance(val, int):
                raise ValueError(f"Bot '{name}' expects int count, got {val}")
            out[cls] = {"count": val, "model": None}

    return out


def train_dqn(
    episodes,
    batch_size,
    learning_rate,
    gamma,
    epsilon_start,
    epsilon_min,
    epsilon_decay,
    target_update_freq,
    memory_size,
    device="cpu",
    checkpoint_path: str = None,
    resume: bool = False,
    save_every: int = 100,
    roster = None
):
    env = LiarsDiceEnv(roster=roster)

    

    policy_net = DQN(MAX_STATE_DIM, NUM_ACTIONS).to(device)
    target_net = DQN(MAX_STATE_DIM, NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr = learning_rate)

    memory = deque(maxlen=memory_size)

    epsilon = epsilon_start

    step_count = 0

    wins = 0

    win_rate_history = []

    # If requested, try to load checkpoint
    start_episode = 0
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        try:
            policy_net.load_state_dict(ckpt["policy_state"])
            target_state = ckpt.get("target_state", None)
            if target_state is not None:
                target_net.load_state_dict(target_state)
            else:
                target_net.load_state_dict(ckpt["policy_state"])

            optimizer.load_state_dict(ckpt.get("optimizer_state", optimizer.state_dict()))

            raw_ep = ckpt.get("episode", 0)
            if isinstance(raw_ep, dict):
                print(f"[Warning] Checkpoint 'episode' key is a dict. Resetting to episode 0.")
                start_episode = 0
            else:
                start_episode = int(raw_ep) + 1


            epsilon = ckpt.get("epsilon", epsilon_start)
            step_count = ckpt.get("step_count", 0)
            wins = ckpt.get("wins", 0)
            win_rate_history = ckpt.get("win_rate_history", [])

            # If memory was saved, reload it (saved as list)
            saved_mem = ckpt.get("memory", None)
            if saved_mem is not None:
                memory = deque(saved_mem, maxlen=memory_size)

            ckpt_roster = ckpt.get("roster", None)
            if ckpt_roster is not None:

                # Convert checkpoint roster from string-keyed JSON to canonical
                ckpt_roster = deserialize_roster(ckpt_roster)

                if roster is None:
                    # User did not supply a roster → use checkpoint roster
                    roster = ckpt_roster
                else:
                    # If user supplied a JSON roster (string keys), convert it once
                    if len(roster) > 0:
                        first_key = next(iter(roster.keys()))
                        if isinstance(first_key, str):
                            roster = deserialize_roster(roster)

                
                def _count_val(v):
                    if isinstance(v, dict):
                        return int(v.get("count", 0))
                    return int(v)
                
                total_players = 1 + sum(_count_val(v) for v in roster.values())

                assert total_players <= MAX_PLAYERS, "Roster exceeds MAX_PLAYERS"

                env = LiarsDiceEnv(roster=roster)



            print(f"Resuming from episode {start_episode}; epsilon={epsilon:.4f}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}\nStarting from scratch.")

    ### Main Training Loop ###

    for episode in range(start_episode, episodes):
        print(f"\rCurrent episode: {episode+1}", end="", flush=True)
        state = env.reset()
        done = False

        while not done:
            # epsilon-greedy action selection
            action_index = select_action(policy_net, state, epsilon)

            if action_index is None:
                assert env.game.is_game_over(), "No legal actions but game is not over!"
                done = True
                break

            next_state, reward, done, _ = env.step(action_index)

            memory.append((state, action_index, reward, next_state, done))

            state = next_state
            step_count += 1

            optimize_model(                  
                memory=memory,
                batch_size=batch_size,
                policy_net=policy_net,
                target_net=target_net,
                optimizer=optimizer,
                gamma=gamma
            )

            if step_count % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        winner = env.game.get_winner()

        if not env.game.is_game_over():
            
            print(f"[Warning] Episode {episode+1} ended but game.is_game_over() is False")


        if winner is not None and winner == env.rl_id:
            wins += 1
        win_rate = wins / (episode + 1)
        win_rate_history.append(win_rate)

        # Periodically save checkpoint
        if checkpoint_path is not None and (episode + 1) % save_every == 0:
            roster_serial = {}
            if roster is not None:
                for k, v in roster.items():
                    name = k.__name__ if isinstance(k, type) else str(k)
                    if isinstance(v, dict):
                        roster_serial[name] = {"count": int(v.get("count", 0)), "model": v.get("model")}
                    else:
                        roster_serial[name] = int(v)

            ckpt = {
                "policy_state": policy_net.state_dict(),
                "target_state": target_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "episode": episode,
                "epsilon": epsilon,
                "step_count": step_count,
                "wins": wins,
                "win_rate_history": win_rate_history,
                "memory": list(memory),
                "roster": roster_serial,
            }
            try:
                torch.save(ckpt, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path} at episode {episode+1}")
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")


    # Save final checkpoint
    if checkpoint_path is not None:
        roster_serial = {}
        if roster is not None:
            for k, v in roster.items():
                name = k.__name__ if isinstance(k, type) else str(k)
                if isinstance(v, dict):
                    roster_serial[name] = {"count": int(v.get("count", 0)), "model": v.get("model")}
                else:
                    roster_serial[name] = int(v)

        ckpt = {
            "policy_state": policy_net.state_dict(),
            "target_state": target_net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "episode": episodes - 1,
            "epsilon": epsilon,
            "step_count": step_count,
            "wins": wins,
            "win_rate_history": win_rate_history,
            "memory": list(memory),
            "roster": roster_serial,
        }
        try:
            torch.save(ckpt, checkpoint_path)
            print(f"Saved final checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save final checkpoint: {e}")

    print()

    plt.plot(win_rate_history)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Win Rate")
    plt.title("DQN RL-Bot Win Rate vs Training Episodes")
    plt.savefig("win_rate.png")
    plt.close()

    return policy_net, target_net
