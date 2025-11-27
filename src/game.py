from config import *
from src.rules import *
from src.bots import *
import numpy as np
import random
from typing import List, Tuple
import os
import json
import datetime
import tkinter as tk
import uuid

def _create_players_from_types(types: List[str]) -> List[object]:
        """
        types: list/tuple of strings length N_PLAYERS.
        Supported tokens:
        - "human", "h" -> human player (None in players list; GUI will handle)
        - "rand", "random" -> RandomBot
        - "risky", "risk" -> RiskyBot
        - "con", "conservative", "risk_averse" -> RiskAverseBot
        - "wildcard_conservative", "wildcard conservative" -> ConservativeBot
        - "aggressive", "wildcard_risky", "wildcard risky" -> AggressiveBot
        Returns list of player objects / None for human.
        """
        if len(types) <= 1:
            raise ValueError("At least two player types must be specified")
        
        players = [None] * len(types)
        for i, t in enumerate(types):
            tt = (t or "").strip().lower()
            if tt in ("human", "h"):
                players[i] = None  # human player
            elif tt in ("rand", "random"):
                players[i] = RandomBot(i)
            elif tt in ("risky", "risk"):
                players[i] = RiskyBot(i)
            elif tt in ("con", "conservative", "risk_averse", "risk-averse"):
                players[i] = RiskAverseBot(i)
            elif tt in ("wildcard conservative", "wildcard_conservative", "wildcard-conservative"):
                players[i] = ConservativeBot(i)
            elif tt in ("aggressive","wildcard risky", "wildcard_risky", "wildcard-risky"):
                players[i] = AggressiveBot(i)
            else:
                raise ValueError(f"Unknown player type: {t} at position {i}")
        return players

class LiarsDiceGame:

    def __init__(self, 
             players, 
             player_types: List[str] | None = None, 
             save_history: bool = True):
        self.players = players
        self.player_types = player_types or self.infer_player_types()
        self.n_players = len(players)
        # maintain per-player dice list 
        self.total_dice = DICE_PER_PLAYER * self.n_players
        self.dice = [[0] * DICE_PER_PLAYER for _ in range(self.n_players)]
        self.dice_counts = [DICE_PER_PLAYER for _ in range(self.n_players)]
        self.current_bid = [0, 0]  # quantity, face
        self.full_history = []  # to store history of bids and actions
        self._current_round = None
        self.last_bidder = None
        self.current_player = 0  # index of current player
        self.round_active = False
        self.save_history = save_history
    
    def active_players(self) -> List[int]:
        """Returns a list of active player indices."""
        return [pid for pid, c in enumerate(self.dice_counts) if c > 0]

    def deal(self, starting_player: int = 0):
        """Deal dice to active players and start a new round.
        starting_player: pid who acts first this round.
        Dice lists are always length DICE_PER_PLAYER; lost dice are 0.
        """
        for pid in range(self.n_players):
            if self.dice_counts[pid] > 0:
                active_vals = list(np.random.randint(1, FACE_COUNT + 1, self.dice_counts[pid])) # active dice values
                padding = [0] * (DICE_PER_PLAYER - len(active_vals)) # lost dice are 0
                self.dice[pid] = active_vals + padding # lost dice are 0
            else:
                # eliminated player all zeros
                self.dice[pid] = [0] * DICE_PER_PLAYER
        self.current_bid = [0, 0]

        self._current_round = {
            "dice": [list(row) for row in self.dice],
            "bids": [],
            "resolution": None
        }
        self.last_bidder = None
        self.current_player = starting_player
        self.round_active = True
    
    def step(self, actor_id: int, action: Tuple[str, any]):
        """Perform an action by actor_id. Returns:
           - None for a completed bid
           - dict with resolution info when a call is resolved
        """
        if self.is_game_over():
            return {"error": "not_your_turn", "expected": self.current_player}

        # ensure actor is the expected current player
        if actor_id != self.current_player:
            return {"error": "not_your_turn", "expected": self.current_player}

        if action[0] == "bid":
            q, f = action[1]
            if not is_bid_higher(self.current_bid, [q, f]):
                return {"error": "invalid_bid", "current_bid": self.current_bid}
            self.current_bid = [q, f]
            self._current_round["bids"].append((actor_id, (q, f)))
            self.last_bidder = actor_id
            # advance to next active player
            self.current_player = self.next_active_player(actor_id)
            return None
        
        elif action[0] == "call":
            # cannot call if no bid has been made
            if self.current_bid[0] == 0:
                return {"error": "no_bid_to_call"}
            # resolve call
            q, f = self.current_bid
            actual = count_face_total(self.dice, f)
            caller = actor_id
            last_bidder = self.last_bidder
            result = {
                "bid": (q, f),
                "actual": actual,
                "winner": None,
                "loser": None,
                "eliminated": None,
            }
            if actual >= q:
                # last bidder was correct -> caller loses one die
                result["winner"] = last_bidder
                result["loser"] = caller
            else:
                # last bidder was lying -> last bidder loses one die
                result["winner"] = caller
                result["loser"] = last_bidder
            # apply penalty
            loser = result["loser"]
            if loser is not None:
                self._remove_one_die(loser)
                if self.dice_counts[loser] == 0:
                    result["eliminated"] = loser
            # record history
            self._current_round["bids"].append((actor_id, "call")) # record the call action
            self._current_round["resolution"] = result # resolution info
            self.full_history.append(self._current_round) # append completed round to history
            self._current_round = None # reset current round

            # If the game just ended (only one or zero active players), don't compute a starter
            # (calling next_active_player when no other active player exists would loop indefinitely).
            if self.is_game_over():
                self.current_player = None
                self.round_active = False
                return result

            # End round: determine who starts next round
            # Common rule: loser starts next round if still active, else next active player after loser
            if loser is not None and self.dice_counts[loser] > 0:
                starter = loser
            else:
                starter = None
                if loser is None:
                    # fallback to next after current caller
                    starter = self.next_active_player(caller)
                else:
                    for i in range(1, self.n_players + 1):
                        candidate = (loser + i) % self.n_players
                        if self.dice_counts[candidate] > 0:
                            starter = candidate
                            break
            # reset for next round if game is not over
            self.round_active = False
            if not self.is_game_over():
                self.deal(starting_player=starter)
            else:
                # game over
                winner = self.get_winner()
                self.full_history.append({"winner": winner})

                if self.save_history:
                    save_path = self.save_history_json()
                    print(f"Game over! Winner: P{winner}. Game history saved to {save_path}")
                self.current_player = None

            return result
        else:
            return {"error": "invalid_action"}
    
    @classmethod
    def game(cls,*player_types, save_json: bool = True, dir: str = "data") -> int | None:
        """
        Start and run a full game given player types.
        Example: game("human", "con", "rand", "con")
        Returns winner pid (int) or None if no winner.
        Note: bot objects are created here; bots must implement act(game) -> action tuple.
        """
        types = list(player_types)
        players = _create_players_from_types(types)
        g = cls(players, player_types=types, save_history=save_json)
        g.deal(starting_player=0)

        # If any human present -> use GUI (blocking until GUI exit), else run headless bots-only loop
        if any(p is None for p in players):
            from src.gui import LiarsDiceGUI
            root = tk.Tk()
            root.title("Liar's Dice")
            app = LiarsDiceGUI(root, g, players)
            app.update_ui()

            def tick():
                if g.is_game_over():
                    winner = g.get_winner()
                    app.show_game_over(winner)
                    return
                if g.current_player is not None and g.current_player !=0:
                    res = app.process_bot_action()
                    if isinstance(res, dict) and not res.get("error") and "actual" in res:
                        app.show_round_result(res)
                root.after(1000, tick)  # check again after 100 ms
            root.after(1000, tick)
            try: 
                root.mainloop()
            finally:
                try:
                    root.destroy()
                except Exception:
                    pass
            
        else:
            # Headless bots-only loop
            while not g.is_game_over():
                cp = g.current_player
                if cp is None:
                    break
                bot = players[cp]
                if bot is None:
                    raise RuntimeError("Human turn encountered in headless mode")
                action = bot.act(g)
                res = g.step(cp, action)

        # game over
        winner = g.get_winner()
        g.full_history.append({"winner": winner})
        if save_json:
            save_path = g.save_history_json(dir=dir)
            print(f"Game over! Winner: P{winner}. Game history saved to {save_path}")
        return g.get_winner()

    def next_active_player(self, pid: int) -> int:
        """Return next pid with at least one die (wrap-around)."""
        n = self.n_players
        next_pid = (pid + 1) % n
        while self.dice_counts[next_pid] == 0:
            next_pid = (next_pid + 1) % n
        return next_pid
    
    def get_legal_bids(self):
        """Return list of legal bids [q,f] given current bid and current total dice."""
        legal = []
        for q in range(1, self.total_dice + 1):
            for f in range(1, FACE_COUNT + 1):
                if is_bid_higher(self.current_bid, [q, f]):
                    legal.append([q, f])
        return legal
    
    def is_game_over(self) -> bool:
        return len(self.active_players()) <= 1

    def get_winner(self) -> int | None:
        active = self.active_players()
        if len(active) == 1:
            return active[0]
        return None  # no winner yet
    
    def _remove_one_die(self, pid: int):
        """Remove one die from player pid by setting one non-zero die to 0 and update counts."""
        if self.dice_counts[pid] <= 0:
            return  # no dice to remove
        # find indices of non-zero dice
        nonzero_indices = [i for i, face in enumerate(self.dice[pid]) if face != 0]
        if not nonzero_indices:
            # nothing to remove (shouldn't happen if dice_counts > 0), but keep counts consistent
            self.dice_counts[pid] = 0
            return
        # choose one die to lose and set to 0
        idx = random.choice(nonzero_indices)
        self.dice[pid][idx] = 0
        self.dice_counts[pid] -= 1
    
    def infer_player_types(self) -> List[str]:
        """
        Infer player types from self.players list.
        Returns list of strings with player types.
        """
        types = []
        for p in self.players:
            if p is None:
                types.append("human")
            elif isinstance(p, RandomBot):
                types.append("rand")
            elif isinstance(p, RiskyBot):
                types.append("risky")
            elif isinstance(p, RiskAverseBot):
                types.append("conservative")
            elif isinstance(p, ConservativeBot):
                types.append("wildcard_conservative")
            elif isinstance(p, AggressiveBot):
                types.append("wildcard_risky")
            else:
                types.append("unknown")
        return types
    
    def save_history_json(self, dir: str = "data") -> str:
        """
        Save the complete game history (self.full_history) into a JSON file.
        Returns the path to the created JSON file.
        """
        os.makedirs(dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        gid = f"{ts}_{uuid.uuid4().hex[:6]}"
        filename = f"game_{gid}.json"
        path = os.path.join(dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "game_id": gid,
                "player_types": self.player_types,
                "full_history": self.full_history
            }, f, ensure_ascii=False, cls=NumpyEncoder, indent=2)
        return path
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)