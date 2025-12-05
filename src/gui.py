import tkinter as tk
from tkinter import messagebox
from src.game import LiarsDiceGame

class LiarsDiceGUI:
    def __init__(self, root: tk.Tk, game: LiarsDiceGame, players: list):
        self.root = root
        self.players = players
        self.game = game

        self.player_types = []
        for i, p in enumerate(self.players):
            if i == 0 or p is None:
                self.player_types.append("Human")
            else:
                self.player_types.append(type(p).__name__)

        root.title("Liar's Dice")

        # UI elements
        self.info_label = tk.Label(root, text="")
        self.info_label.pack()

        # show other players dice toggle
        self.controls_frame = tk.Frame(root)
        self.controls_frame.pack(pady=4)
        self.show_others_var = tk.BooleanVar(value=False)
        self.show_others_cb = tk.Checkbutton(
            self.controls_frame,
            text="Show other players' dice",
            variable=self.show_others_var,
            command=self.update_ui
        )
        self.show_others_cb.pack(side="left")

        self.dice_frame = tk.Frame(root)
        self.dice_frame.pack()
        self.dice_labels = []
        for i in range(len(self.players)):
            prefix = f"P{i} ({self.player_types[i]}): "
            lbl = tk.Label(self.dice_frame, text= prefix + str(self.game.dice[i]))
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

        self.message_label= tk.Label(root, text="", justify="left", wraplength=400, fg = "blue")
        self.message_label.pack(pady=6)

        self.update_ui()

    def update_ui(self):
        """ Update the UI elements based on the current game state. """
        cur = self.game.current_player
        bid = self.game.current_bid
        self.info_label.config(text=f"Current Player: P{cur if cur is not None else '-'}, Current Bid: {bid}")
        show_others = self.show_others_var.get()
        for i, lbl in enumerate(self.dice_labels):
            prefix = f"P{i} ({self.player_types[i]}): "
            if i == 0 or show_others:
                lbl.config(text= prefix + str(self.game.dice[i]))
            else:
                # show '?' for each hidden die; if a die has been removed/marked (0 or None) show 'X'
                hidden_str = ''.join('X' if d == 0 or d is None else '?' for d in self.game.dice[i])
                lbl.config(text= prefix + hidden_str)
        # enable/disable human controls depending on turn
        if self.game.current_player == 0:
            self.bid_button.config(state = "normal")
            self.call_button.config(state = "normal")
        else:
            self.bid_button.config(state = "disabled")
            self.call_button.config(state = "disabled")
        if self.game.current_player is None:
            # game over
            self.bid_button.config(state = "disabled")
            self.call_button.config(state = "disabled")

    def human_bid(self):
        # TODO: limit input to legal bids
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
        # if round ended, show result in GUI
        if isinstance(res, dict) and "actual" in res:
            self.show_round_result(res)
        # if game ended, show game over
        if isinstance(res, dict) and res.get("winner") is not None and self.game.current_player is None:
            # some game implementations may set current_player to None on game over
            self.show_game_over(res.get("winner"))
        return res
    
    def show_round_result(self, res: dict):
        """Display the result of a round after a call."""
        msg = f"Bid {res['bid']} actual: {res.get('actual')}\nWinner: P{res.get('winner')}\nLoser: P{res.get('loser')}"
        if res.get("eliminated"):
            msg += f"\nEliminated: P{res['eliminated']}"
        # show in GUI instead of messagebox
        self.message_label.config(text=msg, fg="blue")
        self.update_ui()

    def show_game_over(self, winner: int):
        """Display game over message in the GUI and disable controls."""
        self.message_label.config(text=f"Game Over — Player P{winner} wins the game!", fg="red")
        # disable the controls so user cannot continue interacting
        try:
            self.bid_button.config(state="disabled")
            self.call_button.config(state="disabled")
            self.show_others_cb.config(state="disabled")
        except Exception:
            pass
        # leave window open so user can see final state
        self.update_ui()