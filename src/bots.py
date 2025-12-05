import random
import math
from src.players import Player
from src.rules import is_bid_higher
from typing import List
import numpy as np
import torch
from src.encode import decode_rl_action, encode_rl_state, encode_rl_action
from src.beliefs import OpponentBelief
from config import FACE_COUNT, MAX_STATE_DIM



class RandomBot(Player):
    def act(self, game):
        """Makes a random valid action."""
        # choose either to call or to make a legal higher bid
        choices = ["call"]
        legal_bids = [] 
        current_total = sum(game.dice_counts)
        for q in range(1, current_total + 1):
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

    def __init__(self, pid: int, call_threshold: float = 0.35, risk_factor: float = 0.8):
        super().__init__(pid)
        self.call_threshold = call_threshold
        self.risk_factor = risk_factor

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
        if prob < self.call_threshold:
            return ("call", None)
        
        # Otherwise, pick aggressive bid maximizing score with risk_factor high
        bid, score = self._best_bid_by_score(game, legal, risk_factor=self.risk_factor)
        if bid is None:
            return ("call", None) # no good bid found
        return ("bid", bid)
    
class RiskAverseBot(_StatBot):
    """Bot that makes conservative bids (lower quantities). Higher threshhold to call."""

    def __init__(self, pid: int, call_threshold: float = 0.05, safe_prob: float = 0.9):
        super().__init__(pid)
        self.call_threshold = call_threshold
        self.safe_prob = safe_prob

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
        if prob < self.call_threshold:
            return ("call", None)
        
        # otherwise, make conservative minimal raise: pick smallest legal bid by (q, face) order
        legal_sorted = sorted(legal, key=lambda bf: (bf[0], bf[1]))
        # prefer bids with reasonable probability
        for candidate in legal_sorted:
            cand_prob = self._prob_bid_true(game, candidate)
            if cand_prob >= self.safe_prob:
                return ("bid", candidate)
        # if none reasonable, pick the minimal increase
        return ("bid", legal_sorted[0])
    
class MixedBot(_StatBot):
    """Mixed behavior bot that acts aggressive with high dice but conservative with low dice."""

    def __init__(self, pid: int, risky_call_threshold: float = 0.3, risky_risk_factor: float = 0.5, ra_call_threshold: float = 0.1, ra_safe_prob: float = 0.8):
        super().__init__(pid)
        self.risky_bot = RiskyBot(pid, call_threshold=risky_call_threshold, risk_factor=risky_risk_factor)
        self.risk_averse_bot = RiskAverseBot(pid, call_threshold=ra_call_threshold, safe_prob=ra_safe_prob)

    def act(self, game):
        _, own_count, _ = self._own_info(game)

        if own_count < 4:
            return self.risk_averse_bot.act(game)
        else:
            return self.risky_bot.act(game)
    
class ConservativeBot(_StatBot):
    """Conservative Bot from WiLDCARD"""

    def act(self, game):
        legal = self._legal_bids(game)
        if not legal:
            return ("call", None)
        
        bid = game.current_bid
        q,f = bid
        _, _, own_faces = self._own_info(game)
        counts = {}
        if own_faces:
            for d in own_faces:
                counts[d] = counts.get(d,0) + 1
            preferred_face = max(counts, key=counts.get) if counts else None

        if q==0: #opening bid
            if preferred_face is None:
                preferred_face = random.randint(1,6)
            q_choice = 1

            if is_bid_higher(bid, [q_choice, preferred_face]):
                return ("bid", [q_choice, preferred_face])
            else:
                bid_choice = random.choice(legal)
                return ("bid", bid_choice)

        #conservative raise
        safe_raises = [(n,f) for (n,f) in legal if n <= counts.get(f,0) and is_bid_higher(bid, (n,f))]
        if safe_raises:
            safe_raises.sort(key=lambda bf: (bf[0], bf[1]))
            #print(f"[DEBUG] Bot bidding {safe_raises[0]} with {own_faces}.")
            return ("bid", safe_raises[0])
        else:
            #print(f"[DEBUG] No safe raises of {own_faces} vs {bid}, calling.")
            return ("call", None) #call if no safe raises exist
        
class AggressiveBot(_StatBot):
    """Aggressive bot from WiLDCARD"""

    def act(self, game):
        legal = self._legal_bids(game)
        if not legal:
            return ("call", None)
        
        bid = game.current_bid
        q, f = bid
        _, _, own_faces = self._own_info(game)

        rand = np.random.randint(0, 100)
        if rand < 50 or q == 0:
            legal_sorted = sorted(legal, key=lambda bf: (bf[0], bf[1]))
            # print(f"[DEBUG] Making bid {legal_sorted[0]} with {own_faces}")
            return ("bid", legal_sorted[0])
        else:
            # print(f"[DEBUG] Making call of {own_faces} vs {bid}")
            return("call", None)
        



class DQNBot:
    """
    A self-play bot whose behavior is EXACTLY consistent with:
      - rl_train.select_action()
      - rl_env.get_legal_action_indices()
      - encode_rl_state() / decode_rl_action()

    It uses the same action space, same legal-action logic,
    same masking logic, and same state encoding as the RL agent.

    This ensures training-time and inference-time policies match
    1:1 and prevents illegal actions or corrupted game states.
    """

    

    def __init__(self, pid, policy_net, total_players, device="cpu", epsilon=0.0):
        self.pid = pid
        self.policy_net = policy_net.to(device)
        self.policy_net.eval()

        self.device = device
        self.epsilon = epsilon 
        self.total_players = total_players

        
        self.opponent_beliefs = {
            p: OpponentBelief()
            for p in range(total_players)
            if p != self.pid
        }

        assert self.pid not in self.opponent_beliefs

   
    # Build RL State 
    def _build_state(self, game):
        total_dice = sum(game.dice_counts)

        my_faces = game.dice[self.pid]
        agent_dice_vec = [my_faces.count(f) for f in range(1, FACE_COUNT + 1)]
        agent_dice_count = game.dice_counts[self.pid]

        current_bid = game.current_bid  # list of [q,f]

        # opponent beliefs (in consistent order)
        opponent_belief_vecs = []
        for p in range(self.total_players):
            if p == self.pid:
                continue
            belief = self.opponent_beliefs[p]
            opponent_belief_vecs.append(belief.sample_belief().tolist())

        terminal_flag = int(game.is_game_over())

        state_vec = encode_rl_state(
            total_dice=total_dice,
            agent_dice_count=agent_dice_count,
            agent_dice_vector=agent_dice_vec,
            current_bid=current_bid,
            opponent_beliefs=opponent_belief_vecs,
            terminal_flag=terminal_flag
        )

        # ensure matches MAX_STATE_DIM
        assert len(state_vec) == MAX_STATE_DIM, \
            f"DQNBot: incorrect state dim {len(state_vec)} != {MAX_STATE_DIM}"

        return state_vec

    
    # Compute action like rl_train.select_action
    def act(self, game):
        from src.rl_env import get_legal_action_indices
        # Build RL-format state vec
        state_vec = self._build_state(game)

        # Compute legal action indices 
        legal_indices = get_legal_action_indices(state_vec)  

        if len(legal_indices) == 0:
            # If terminal or weird edge case, default CALL
            assert game.is_game_over(), "DQNBot: no legal actions but game is not over."
            return ("call", None)

        # eps-greedy (rare, usually epsilon=0.0 for evaluation)
        if random.random() < self.epsilon:
            idx = int(random.choice(legal_indices))
            decoded = decode_rl_action(idx)
            return ("call", None) if decoded == "call" else ("bid", decoded)

        # Exploitation — identical to training loop
        state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_vals = self.policy_net(state_t).squeeze(0)

        # initialize masked Q to -inf
        masked_q = torch.full_like(q_vals, float("-inf"))

        legal_t = torch.tensor(legal_indices, dtype=torch.long, device=self.device)
        masked_q[legal_t] = q_vals[legal_t]

        best_idx = int(torch.argmax(masked_q).item())
        decoded = decode_rl_action(best_idx)

        # decode into game action
        if decoded == "call":
            return ("call", None)
        else:
            return ("bid", decoded)

    
    def update_belief_from_bid(self, bidder_pid, bid):
        """
        If external code calls this, ensure that self-play bot
        updates its internal beliefs like RL env does.
        """
        if bidder_pid in self.opponent_beliefs:
            self.opponent_beliefs[bidder_pid].update_from_bid(bid)

        
class DQNBotA(DQNBot):
    """Tiny subclass to enable multiple DQN bots during scheduled training."""
    pass

class DQNBotB(DQNBot):
    pass