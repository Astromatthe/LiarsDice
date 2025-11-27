from src.game import LiarsDiceGame
from src.bots import RandomBot, RiskyBot, RiskAverseBot, ConservativeBot, AggressiveBot
from src.gui import LiarsDiceGUI
import tkinter as tk
import random

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
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--buffer", type=int, default=10000)

    args = parser.parse_args()

    if args.train:
        from src.rl_train import train_dqn
        start_time = time.perf_counter()
        policy, target = train_dqn(
            episodes=args.episodes,
            batch_size=args.batch,
            learning_rate=args.lr,
            gamma=args.gamma,
            epsilon_start=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            target_update_freq=1000,
            memory_size=args.buffer
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Training time: {elapsed_time:.2f} s")
    else:
        main()   # launches GUI mode
