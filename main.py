from src.game import LiarsDiceGame
from src.bots import RandomBot, RiskyBot, RiskAverseBot
from src.gui import LiarsDiceGUI
from src.state import *
import tkinter as tk
import random

## TEST
def main():
    # create players: human + 3 random bots
    players = [None] * 4
    players[0] = None # human player in GUI
    for i in range(1, 4):
        players[i] = RandomBot(i)
    game = LiarsDiceGame(players)
    game.deal(starting_player=0)

    # start GUI (GUI only updates / shows results; main drives bot turns)
    root = tk.Tk()
    root.title("Liar's Dice")

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
    
    def start_game(root: tk.Tk, mode: str): 
        # create players: human (pid 0) + bots depending on mode
        players = [None] * 4
        players[0] = None # human player in GUI
        if mode == "all_random":
            for i in range(1, 4):
                players[i] = RandomBot(i)
        elif mode == "all_risky":
            for i in range(1, 4):
                players[i] = RiskyBot(i)
        elif mode == "all_risk_averse":
            for i in range(1, 4):
                players[i] = RiskAverseBot(i)
        elif mode == "mixed":
            for i in range(1, 4):
                rand = random.random()
                if rand < 0.33:
                    players[i] = RandomBot(i)
                elif rand < 0.66:
                    players[i] = RiskyBot(i)
                else:
                    players[i] = RiskAverseBot(i)
        else:
            # fallback to all random
            for i in range(1, 4):
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
    main()