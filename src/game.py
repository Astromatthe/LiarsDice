from config import *
from src.rules import *
import numpy as np
import random
from typing import List, Tuple, Dict

class LiarsDiceGame:

    def __init__(self, players):
        self.players = players
        self.n_players = N_PLAYERS
        # maintain per-player dice list 
        self.total_dice = TOTAL_DICE
        self.dice = [[0] * DICE_PER_PLAYER for _ in range(self.n_players)]
        print(self.dice)
        self.dice_counts = [DICE_PER_PLAYER for _ in range(self.n_players)]
        self.current_bid = [0, 0]  # quantity, face
        self.history = []  # to store history of bids and actions
        self.last_bidder = None
        self.current_player = 0  # index of current player
        self.round_active = False
    
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
        self.history = []
        self.last_bidder = None
        self.current_player = starting_player
        self.round_active = True
        # full deal: each player gets DICE_PER_PLAYER dice
        # self.dice = [list(np.random.randint(1, FACE_COUNT + 1, DICE_PER_PLAYER)) for _ in range(N_PLAYERS)]

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
            self.history.append((actor_id, ("bid", (q, f))))
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
                "eliminated": []
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
                    result["eliminated"].append(loser)
            # record history
            self.history.append((actor_id, ("call", None)))

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
                self.current_player = None

            return result
        else:
            return {"error": "invalid_action"}