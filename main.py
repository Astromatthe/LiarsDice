from src.game import LiarsDiceGame
from src.bots import RandomBot
from src.gui import LiarsDiceGUI
from src.state import *
import tkinter as tk

## TEST
def main():
    # create players: human + 3 random bots
    players = [None] * 4
    players[0] = None # human player in GUI
    for i in range(1, 4):
        players[i] = RandomBot(i)
    game = LiarsDiceGame(players)
    game.deal(starting_player=0)
    print("Dealt dice:", game.dice)

    # start GUI (GUI only updates / shows results; main drives bot turns)
    root = tk.Tk()
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