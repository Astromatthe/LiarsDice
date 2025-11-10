import tkinter as tk
from tkinter import messagebox
from src.game import LiarsDiceGame
from src.bots import RandomBot

class LiarsDiceGUI:
    def __init__(self, root: tk.Tk, game: LiarsDiceGame, players: list):
        self.root = root
        self.players = players
        self.game = game

        root.title("Liar's Dice")

        # UI elements
        self.info_label = tk.Label(root, text="")
        self.info_label.pack()

        self.dice_frame = tk.Frame(root)
        self.dice_frame.pack()
        self.dice_labels = []
        for i in range(len(self.players)):
            lbl = tk.Label(self.dice_frame, text=f"P{i}: {self.game.dice[i]}")
            lbl.grid(row=0, column=i, padx=5)
            self.dice_labels.append(lbl)

        self.bid_frame = tk.Frame(root)
        self.bid_frame.pack(pady=10)
        tk.Label(self.bid_frame, text="Quantity:").grid(row=0, column=0)
        self.qty_entry = tk.Entry(self.bid_frame, width=5)
        self.qty_entry.grid(row=0, column=1)
        tk.Label(self.bid_frame, text="Face:").grid(row=0, column=2)
        self.face_entry = tk.Entry(self.bid_frame, width=5)
        self.face_entry.grid(row=0, column=3)
        self.bid_button = tk.Button(self.bid_frame, text="Bid", command=self.human_bid)
        self.bid_button.grid(row=0, column=4, padx=5)
        self.call_button = tk.Button(self.bid_frame, text="Call", command=self.human_call)
        self.call_button.grid(row=0, column=5, padx=5)

    def update_ui(self):
        """ Update the UI elements based on the current game state. """
        self.info_label.config(text=f"Current Player: P{self.game.current_player}, Current Bid: {self.game.current_bid}")
        for i, lbl in enumerate(self.dice_labels):
            lbl.config(text=f"P{i}: {self.game.dice[i]}")
        # enable/disable human controls depending on turn
        if self.game.current_player == 0:
            self.bid_button.config(state = "normal")
            self.call_button.config(state = "normal")
        else:
            self.bid_button.config(state = "disabled")
            self.call_button.config(state = "disabled")

    def human_bid(self):
        """ Handle human bid action. """
        try:
            q = int(self.qty_entry.get())
            f = int(self.face_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Quantity and face must be integers.")
            return
        res = self.game.step(0, ("bid", [q, f]))
        if isinstance(res, dict) and res.get("error"):
            messagebox.showerror("Invalid Bid", f"Error: {res['error']}")
            return
        self.update_ui()

    def human_call(self):
        """ Handle human call action. """
        res = self.game.step(0, ("call", None))
        if isinstance(res, dict):
            if res.get("error"):
                messagebox.showerror("Invalid Call", f"Error: {res['error']}")
                return
            if "actual" in res:
                self.show_round_result(res)
        self.update_ui()
    
    def process_bot_action(self):
        """Process one bot action (called from main loop). Returns the result of step()."""
        if self.game.current_player is None or self.game.current_player == 0:
            return None  # not bot's turn
        bot = self.players[self.game.current_player]
        action = bot.act(self.game)
        res = self.game.step(self.game.current_player, action)
        self.update_ui()
        return res
    
    def show_round_result(self, res: dict):
        """Display the result of a round after a call."""
        msg = f"Bid {res['bid']} actual: {res.get('actual')}\nWinner: P{res.get('winner')}\nLoser: P{res.get('loser')}"
        if res.get("eliminated"):
            msg += f"\nEliminated: {res['eliminated']}"
        messagebox.showinfo("Round Result", msg)
        self.update_ui()

    def show_game_over(self, winner: int):
        """Display game over message."""
        messagebox.showinfo("Game Over", f"Player P{winner} wins the game!")
        self.root.quit()