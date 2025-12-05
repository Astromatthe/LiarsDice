import argparse
import json
import torch
import os
from typing import Dict, Any, List

from src.rl_env import LiarsDiceEnv, get_legal_action_indices
from src.dqn_model import DQN, DRONMoE
from config import MAX_STATE_DIM, NUM_ACTIONS, NON_OPPONENT_FEATURE_DIM, OPPONENT_FEATURE_DIM, DRON_MOE_GATE_HIDDEN, DRON_MOE_HIDDEN, DRON_MOE_NUM_EXPERTS

def load_single_roster(roster_json: Dict[str, Any]):

    import src.bots as bots_mod
    roster = {}

    for name, entry, in roster_json.items():
        cls = getattr(bots_mod, name, None)
        if cls is None:
            raise ValueError(f"Unknown bot class '{name}' in roster.")
        
        if isinstance(entry, int):
            roster[cls] = {"count": entry, "model": None}

        elif isinstance(entry, dict):
            count = entry.get("count", 1)
            model_path = entry.get("model", None)
            roster[cls] = {"count": count, "model": model_path}

        else:
            raise ValueError(f"Invalid roster entry for '{name}': {entry}")
    
    return roster

def greedy_action(policy_net, state_vec):
    device = next(policy_net.parameters()).device

    
    legal_actions = get_legal_action_indices(state_vec)

    if len(legal_actions) == 0:
        return None

    
    state_t = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(state_t).squeeze(0)

    # Mask illegal actions
    masked_q = torch.full_like(q_values, float("-inf"))
    legal_t = torch.tensor(legal_actions, dtype=torch.long, device=device)
    masked_q[legal_t] = q_values[legal_t]

    # Pick best legal action
    best = int(torch.argmax(masked_q).item())
    return best

def evaluate_once(policy_net, roster, episodes: int, device="cpu"):
    env = LiarsDiceEnv(roster=roster)
    wins = 0

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = greedy_action(policy_net, state)

            try:
                state, _, done, _ = env.step(action)
            except AssertionError as e:
                print("\n[Evaluation ERROR] Illegal env.step() during evaluation")
                print("Error:", e)
                print("Action:", action)
                print("State vec:", state)
                return 0.0

        if env.game.get_winner() == env.rl_id:
            wins += 1

    return wins / episodes

def evaluate_schedule(agent_path: str, schedule_path: str, episodes: int, device="cpu"):

    # 1. Load agent checkpoint
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Agent file '{agent_path}' not found.")

    ckpt = torch.load(agent_path, map_location=device)
    policy_state = ckpt.get("policy_state", None)
    if policy_state is None:
        raise ValueError(f"{agent_path} does not contain 'policy_state'.")

    model_type = ckpt.get("model_type", "dqn")
    if model_type == "dron_moe":
        policy_net = DRONMoE(state_dim=MAX_STATE_DIM, action_dim=NUM_ACTIONS, non_opp_dim=NON_OPPONENT_FEATURE_DIM, opp_dim=OPPONENT_FEATURE_DIM, num_experts=DRON_MOE_NUM_EXPERTS, hidden_dim=DRON_MOE_HIDDEN).to(device)
    else:
        policy_net = DQN(MAX_STATE_DIM, NUM_ACTIONS).to(device)
    
    policy_net.load_state_dict(policy_state)
    policy_net.eval()

    # 2. Load schedule file
    if not os.path.exists(schedule_path):
        raise FileNotFoundError(f"Schedule file '{schedule_path}' not found.")

    with open(schedule_path, "r") as f:
        schedule_json = json.load(f)

    if not isinstance(schedule_json, list):
        raise ValueError("Evaluation schedule must be a LIST of rosters.")

    results = []

    # 3. Evaluate each roster in schedule
    for idx, roster_entry in enumerate(schedule_json):
        print(f"\n=== Evaluating Roster {idx+1}/{len(schedule_json)} ===")

        roster = load_single_roster(roster_entry)

        winrate = evaluate_once(
            policy_net=policy_net,
            roster=roster,
            episodes=episodes,
            device=device
        )

        results.append((roster_entry, winrate))
        print(f"Win Rate = {winrate:.4f}")

    return results

def write_csv(results, csv_path="eval_log.csv", episodes=1000, agent=None):
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a") as f:
        if write_header:
            f.write("agent,episodes,roster,win_rate\n")

        for roster_entry, winrate in results:
            roster_str = json.dumps(roster_entry).replace('"', '""')
            f.write(f'"{agent}",{episodes},"{roster_str}",{winrate}\n')

    print(f"\nSaved evaluation results → {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent against multiple opponent rosters.")
    parser.add_argument("--agent", type=str, required=True, help="Path to trained agent .pt file")
    parser.add_argument("--schedule", type=str, required=True, help="Path to schedule JSON (list of rosters)")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per roster")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    results = evaluate_schedule(
        agent_path=args.agent,
        schedule_path=args.schedule,
        episodes=args.episodes,
        device=args.device
    )

    if args.csv:
        write_csv(results, csv_path=args.csv, episodes=args.episodes, agent=args.agent)


if __name__ == "__main__":
    main()       
