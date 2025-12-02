import argparse
import json
import os
import subprocess
import math
import shutil
import torch
from typing import List, Dict, Any


def load_schedule(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Schedule file '{path}' not found.")
    
    with open(path, "r") as f:
        schedule = json.load(f)

    if not isinstance(schedule, list):
        raise ValueError("Unsupported schedule format.")
    
    return schedule

def read_checkpoint_episode(path="checkpoint.pt"):
    if not os.path.exists(path):
        return 0
    try:
        ckpt = torch.load(path, map_location="cpu")
        return int(ckpt.get("episode", 0)) + 1
    except Exception:
        return 0


def compute_epsilon_decay(eps_start: float, eps_min: float, eps_min_at: float, episodes: int):
    """
    Solve:
        eps_min = eps_start * decay^(k)
    where k = eps_min_at * episodes

    Returns decay in (0,1)
    """

    k = eps_min_at * episodes
    if k <= 0:
        return 1.0
    
    if eps_start <= eps_min:
        return 1.0
    
    decay = (eps_min/eps_start)**(1.0/k)
    return decay

def write_roster_json(roster: Dict[str, Any]):
    with open("roster.json", "w") as f:
        json.dump(roster, f, indent=4)

def run_stage(stage_index: int, stage: Dict[str, Any]):
    stage_num = stage_index + 1
    print(f"\n===== Running Stage {stage_num} =====")

    raw_episodes = stage["episodes"]
    prev_ep = read_checkpoint_episode()
    episodes = prev_ep + raw_episodes

    lr = stage["lr"]
    roster = stage["roster"]

    eps_start = stage["eps_start"]
    eps_min = stage["eps_min"]
    eps_min_at = stage["eps_min_at"]

    model_type = stage.get("model_type", "dqn")

    # compute epsilon_decay to reach eps_min at eps_min_at%
    eps_decay = compute_epsilon_decay(
        eps_start=eps_start,
        eps_min=eps_min,
        eps_min_at=eps_min_at,
        episodes=raw_episodes
    )

    update_freq = stage["update"]

    # 1. Write roster.json for this stage
    write_roster_json(roster)
    print(f"[Stage {stage_num}] roster.json updated.")

    # 2. Build command for training
    cmd = [
        "python", "main.py",
        "--train",
        "--episodes", str(episodes),
        "--lr", str(lr),
        "--eps_start", str(eps_start),
        "--eps_end", str(eps_min),
        "--eps_decay", str(eps_decay),
        "--update", str(update_freq),
        "--checkpoint", "checkpoint.pt",
        "--resume",
        "--save_every", "200",
        "--model_type", model_type,
    ]

    print(f"[Stage {stage_num}] Running command:")
    print(" ".join(cmd))

    # 3. Execute training
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[Stage {stage_num}] ERROR: Training process failed!")
        exit(1)

    # 4. Save network snapshot
    os.makedirs("agents", exist_ok=True)
    snapshot_path = f"agents/agent_{stage_num}.pt"

   
    shutil.copyfile("checkpoint.pt", snapshot_path)
    print(f"[Stage {stage_num}] Saved agent snapshot → {snapshot_path}")



# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=str, required=True)
    parser.add_argument("--start_stage", type=int, default=1)
    

    args = parser.parse_args()

    training_stages = load_schedule(args.schedule)
    start_stage = max(1, args.start_stage)

    if start_stage > len(training_stages):
        print(f"Invalid start_stage {start_stage}. Only {len(training_stages)} stages exist.")
        return

    print(f"Scheduler starting at stage {start_stage} of {len(training_stages)}")

    for idx in range(start_stage - 1, len(training_stages)):
        run_stage(idx, training_stages[idx])

    print("\n===== ALL TRAINING STAGES COMPLETE =====")


if __name__ == "__main__":
    main()