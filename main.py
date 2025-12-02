from src.game import LiarsDiceGame
from src.bots import RandomBot, RiskyBot, RiskAverseBot, ConservativeBot, AggressiveBot
from src.gui import LiarsDiceGUI
import tkinter as tk
import random
import json
import os
import importlib
import torch

## TEST
def main():
    # start GUI (GUI only updates / shows results; main drives bot turns)
    root = tk.Tk()
    root.title("Liar's Dice")
    N_PLAYERS = 4

    selected = {"mode": None}

    start_frame = tk.Frame(root, padx = 20, pady = 20)
    start_frame.pack()

    tk.Label(start_frame, text="Select Game Mode:").pack(pady=(0,10))

    def select_mode(mode):
        selected["mode"] = mode
        start_frame.destroy()
        start_game(root,mode)

    tk.Button(start_frame, text = "All Random Bots",
              width = 30, command = lambda: select_mode("all_random")).pack(pady=3)
    tk.Button(start_frame, text = "All Risky Bots",
              width = 30, command = lambda: select_mode("all_risky")).pack(pady=3)
    tk.Button(start_frame, text = "All Risk-Averse Bots",
             width = 30, command = lambda: select_mode("all_risk_averse")). pack(pady=3)
    tk.Button(start_frame, text = "Mixed Bots",
             width = 30, command = lambda: select_mode("mixed")).pack(pady=3)
    tk.Button(start_frame, text = "All Wildcard Conservative Bots",
             width = 30, command = lambda: select_mode("all_cons")).pack(pady=3)
    tk.Button(start_frame, text = "All Wildcard Aggressive Bots",
             width = 30, command = lambda: select_mode("all_agg")).pack(pady=3)
    
    def start_game(root: tk.Tk, mode: str): 
        # create players: human (pid 0) + bots depending on mode
        players = [None] * N_PLAYERS
        players[0] = None # human player in GUI
        if mode == "all_random":
            for i in range(1, N_PLAYERS):
                players[i] = RandomBot(i)
        elif mode == "all_risky":
            for i in range(1, N_PLAYERS):
                players[i] = RiskyBot(i)
        elif mode == "all_risk_averse":
            for i in range(1, N_PLAYERS):
                players[i] = RiskAverseBot(i)
        elif mode == "mixed":
            for i in range(1, N_PLAYERS):
                rand = random.random()
                if rand < 0.33:
                    players[i] = RandomBot(i)
                elif rand < 0.66:
                    players[i] = RiskyBot(i)
                else:
                    players[i] = RiskAverseBot(i)
        elif mode == "all_cons":
            for i in range(1, N_PLAYERS):
                players[i] = ConservativeBot(i)
        elif mode == "all_agg":
            for i in range(1, N_PLAYERS):
                players[i] = AggressiveBot(i)
        else:
            # fallback to all random
            for i in range(1, N_PLAYERS):
                players[i] = RandomBot(i)

        game = LiarsDiceGame(players)
        game.deal(starting_player=0)

        app = LiarsDiceGUI(root, game, players)
        app.update_ui()

        def tick():
            # periodic game tick to handle turns and refresh GUI
            if game.is_game_over():
                winner = game.get_winner()
                app.show_game_over(winner)
                return
            # if it's a bot's turn, process exactly one bot action per tick
            if game.current_player is not None and game.current_player != 0:
                res = app.process_bot_action()
                if isinstance(res, dict) and not res.get("error"):
                    # show round result if a call resolved the round
                    if "actual" in res:
                        app.show_round_result(res)
            # schedule next tick
            root.after(300, tick) # 300 ms between ticks

        root.after(300, tick)

    root.mainloop()

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="Run the DQN training loop instead of launching GUI")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--eps_start", type=float,default=1.0)
    parser.add_argument("--eps_end", type=float, default= 0.01)
    parser.add_argument("--eps_decay", type=float, default = 0.995)
    parser.add_argument("--update",type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--model_type", type=str, default="dqn", choices=["dqn", "dron_moe"], help="Which Q-network architecture to train - DQN or DRON-MoE")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Use CPU or GPU for network")

    args = parser.parse_args()

    if args.train:
        from src.rl_train import train_dqn
        # Load roster from file if present (no CLI flags required)
        roster = None
        roster_file = os.environ.get("ROSTER_FILE", "roster.json")
        if args.checkpoint is None:
            # if no checkpoint CLI flag provided, still attempt default roster file
            pass
        if os.path.exists(roster_file):
            try:
                with open(roster_file, "r") as f:
                    raw = json.load(f)
                # raw expected format: { "RandomBot": 1, "RiskyBot": 1 }
                bots_mod = importlib.import_module("src.bots")
                roster = {}
                for name, cnt in raw.items():
                    cls = getattr(bots_mod, name, None)
                    if cls is None:
                        print(f"Warning: unknown bot name '{name}' in {roster_file}; skipping")
                        continue
                    # allow either int (count) or dict {"count": n, "model": "path.pt"}
                    if isinstance(cnt, dict):
                        c = int(cnt.get("count", 0))
                        model = cnt.get("model", None)
                        roster[cls] = {"count": c, "model": model}
                    else:
                        roster[cls] = int(cnt)
                print(f"Loaded roster from {roster_file}: {raw}")
            except Exception as e:
                print(f"Failed to load roster file {roster_file}: {e}")
        start_time = time.perf_counter()
        if torch.cuda.is_available() and args.device == "cuda":
            print("CUDA available, using GPU acceleration")
            device = torch.device("cuda")
        else:
            print("CUDA not used, using CPU.")
            device = torch.device("cpu")
        policy, target = train_dqn(
            episodes=args.episodes,
            batch_size=64,
            learning_rate=args.lr,
            gamma=0.99,
            epsilon_start=args.eps_start,
            epsilon_min=args.eps_end,
            epsilon_decay=args.eps_decay,
            target_update_freq=args.update,
            memory_size=10000,
            device=device,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
            save_every=args.save_every,
            roster=roster,
            model_type=args.model_type,
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Training time: {elapsed_time:.2f} s")
    else:
        main()   # launches GUI mode
