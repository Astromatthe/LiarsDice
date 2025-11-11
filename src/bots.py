import random
import math
from src.players import Player
from src.rules import is_bid_higher
from typing import List, Tuple

class RandomBot(Player):
    def act(self, game):
        """Makes a random valid action."""
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
        
class _StatBot(Player):
    """Base bot that reasons about probability a bid is true given its own dice and game history"""

    def _own_info(self, game):
        # number of active dice and own non-zero dice and own faces
        active_total = sum(game.dice_counts)
        own_count = game.dice_counts[self.pid]
        own_faces = [face for face in game.dice[self.pid] if face != 0]
        return active_total, own_count, own_faces
    
    def _prob_bid_true(self,game, bid: List[int]) -> float:
        """Estimate P(actual count of face f>= q) given own dice and only uniform prior for others."""
        q, f = bid
        active_total, own_count, own_faces = self._own_info(game)
        own_known = sum(1 for d in own_faces if d == f) # of f's in own dice
        unknown = active_total - own_count # number of dice we don't know about
        required = q - own_known # number of f's needed from unknown dice

        if required <= 0:
            return 1.0  # already have enough f's in own dice
        if required > unknown:
            return 0.0  # impossible to meet the bid
        # binomial tail: P(X >= required) where X ~ Binomial(unknown, 1/6)
        p = 1.0 / game.FACE_COUNT if hasattr(game, 'FACE_COUNT') else 1.0 / 6.0
        # use math.comb for binomial coefficients
        prob = 0.0
        for k in range(required, unknown + 1):
            prob += math.comb(unknown, k) * (p ** k) * ((1 - p) ** (unknown - k)) # binomial probability
        return prob
    
    def _legal_bids(self, game):
        return game.get_legal_bids()
    
    def _best_bid_by_score(self, game, candidate_bids: List[List[int]], risk_factor: float = 0.0):
        """Score candidate bids by estimated probability minus a penalty for large quantity.
           risk_factor in [0,1] shifts preference to larger bids (higher => more aggressive).
        """
        best = None
        best_score = -1.0

        for bid in candidate_bids:
            prob = self._prob_bid_true(game, bid)
            # penalty proportional to q/active_total; risk_factor reduces penalty for risky bots
            active_total = sum(game.dice_counts)
            penalty = (bid[0] / max(1, active_total)) * (1.0 - risk_factor)
            score = prob - penalty
            if score > best_score:
                best_score = score
                best = bid
        return best, best_score
    
class RiskyBot(_StatBot):
    """Bot that makes bids favoring higher quantities (more aggressive). Lower threshhold to call."""

    def act(self, game):
        # if no legal bids exist, must call
        legal = self._legal_bids(game)
        if not legal:
            return ("call", None)
        
        # evaluate current bid probability
        bid = game.current_bid
        q,f = bid
        # if no bid yet, prefer making aggressive opening bid
        if q == 0:
            # pick face from own most common face or random
            _, _, own_faces = self._own_info(game)
            preferred_face = None
            if own_faces:
                counts = {}
                for d in own_faces:
                    counts[d] = counts.get(d, 0) + 1
                preferred_face = max(counts, key=counts.get)
            if preferred_face is None:
                preferred_face = random.randint(1,6)
            # aggressiveq: expected + 1
            active_total = sum(game.dice_counts)
            expected = sum(1 for d in own_faces if d == preferred_face) + (active_total - len(own_faces)) / 6.0
            q_choice = min(active_total, max(1, int(math.ceil(expected)) + 1))

            # ensure legal
            if is_bid_higher(bid, [q_choice, preferred_face]):
                return ("bid", [q_choice, preferred_face])
            else:
                # fallback to random legal bid
                bid_choice = random.choice(legal)
                return ("bid", bid_choice)
            
        prob = self._prob_bid_true(game, bid)
        # risky threshhold: call if very unlikely 
        if prob < 0.35:
            return ("call", None)
        
        # Otherwise, pick aggressive bid maximizing score with risk_factor high
        bid, score = self._best_bid_by_score(game, legal, risk_factor=0.8)
        if bid is None:
            return ("call", None) # no good bid found
        return ("bid", bid)
    
class RiskAverseBot(_StatBot):
    """Bot that makes conservative bids (lower quantities). Higher threshhold to call."""

    def act(self, game):
        legal = self._legal_bids(game)
        if not legal:
            return ("call", None)
        
        bid = game.current_bid
        q,f = bid
        if q == 0:
            # conservative opening: bid expected rounded down
            _, _, own_faces = self._own_info(game)
            # choose face with own count if any, else random
            preferred_face = None
            if own_faces:
                counts = {}
                for d in own_faces:
                    counts[d] = counts.get(d, 0) + 1
                preferred_face = max(counts, key=counts.get)
            if preferred_face is None:
                preferred_face = random.randint(1,6)
            active_total = sum(game.dice_counts)
            expected = sum(1 for d in own_faces if d == preferred_face) + (active_total - len(own_faces)) / 6.0
            q_choice = max(1, int(math.floor(expected)))

            if q_choice < 1:
                q_choice = 1 # ensure at least 1
            if is_bid_higher(bid, [q_choice, preferred_face]):
                return ("bid", [q_choice, preferred_face]) # conservative opening bid
            else:
                bid_choice = random.choice(legal)
                return ("bid", bid_choice)
            
        prob = self._prob_bid_true(game, bid)
        # risk-averse threshhold: call if less likely
        if prob < 0.60:
            return ("call", None)
        
        # otherwise, make conservative minimal raise: pick smallest legal bid by (q, face) order
        legal_sorted = sorted(legal, key=lambda bf: (bf[0], bf[1]))
        # prefer bids with reasonable probability
        for candidate in legal_sorted:
            cand_prob = self._prob_bid_true(game, candidate)
            if cand_prob >= 0.5:
                return ("bid", candidate)
        # if none reasonable, pick the minimal increase
        return ("bid", legal_sorted[0])