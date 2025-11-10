import random
from src.players import Player
from src.rules import is_bid_higher
from typing import List, Tuple

class RandomBot(Player):
    def act(self, game):
        """Makes a random valid action."""
        # game exposes current bid (q,face) and legal next bids
        prev_bid, prev_face = game.current_bid

        # choose either to call or to make a legal higher bid
        choices = ["call"]
        legal_bids = [] 
        for q in range(1, game.total_dice + 1):
            for f in range(1, 7):
                if is_bid_higher(game.current_bid, [q, f]):
                    legal_bids.append([q, f])   # all legal higher bids
        if legal_bids:
            choices.append("bid") # if there are legal bids, can choose to bid 
        action = random.choice(choices) # randomly choose action

        if action == "call":
            return ("call", None)
        else:
            bid = random.choice(legal_bids)
            return ("bid", bid)