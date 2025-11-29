from typing import List, Tuple
from config import FACE_COUNT, MAX_STATE_DIM, MAX_PLAYERS, NUM_ACTIONS
from src.game import LiarsDiceGame
from src.bots import RandomBot, RiskAverseBot, RiskyBot
from src.beliefs import OpponentBelief
from src.encode import encode_rl_state, decode_rl_action, encode_rl_action
from src.rules import is_bid_higher
import random
import numpy as np
import torch
from src.dqn_model import DQN
import os



class LiarsDiceEnv:
    def __init__(self, rl_id: int = 0, roster=None):
        """
        roster: dict mapping BotClass → count
            Example: {RandomBot: 1, RiskyBot: 1}
            total_players = 1 (RL agent) + sum(counts)
        """

        if roster is None:
            roster = {RandomBot: 1}

        self.roster = roster
        self.rl_id = rl_id

        # roster values may be ints or dicts {"count": n, "model": ...}
        def _count_val(v):
            return int(v) if not isinstance(v, dict) else int(v.get("count", 0))

        self.total_players = 1 + sum(_count_val(v) for v in roster.values())

        assert self.total_players <= MAX_PLAYERS, \
            f"Roster creates {self.total_players} players, but MAX_PLAYERS={MAX_PLAYERS}"
        
        self.players = [None]

        for BotClass, raw_count in roster.items():
            # raw_count may be an int or a dict like {"count": n, "model": "file.pt"}
            if isinstance(raw_count, dict):
                count = int(raw_count.get("count", 0))
                model_path = raw_count.get("model", None)
            else:
                count = int(raw_count)
                model_path = None

            for _ in range(count):
                pid = len(self.players)
                # If this is a DQNBot-like class and a model path is provided, load model
                try:
                    name = BotClass.__name__
                except Exception:
                    name = str(BotClass)

                if name == "DQNBot" and model_path is not None and os.path.exists(model_path):
                    # build model with correct dimensions and load weights
                    model = DQN(MAX_STATE_DIM, NUM_ACTIONS)
                    try:
                        ckpt = torch.load(model_path, map_location="cpu")
                        # If checkpoint contains policy_state dict, use it
                        if isinstance(ckpt, dict) and "policy_state" in ckpt:
                            model.load_state_dict(ckpt["policy_state"])
                        else:
                            # assume file is raw state_dict
                            model.load_state_dict(ckpt)
                        print(f"Loaded DQN model for opponent from {model_path}")
                    except Exception as e:
                        print(f"Failed to load DQN model from {model_path}: {e}\nUsing uninitialized model.")
                    # instantiate bot with model and n_players placeholder; DQNBot expects (pid, model, n_players)
                    try:
                        from src.bots import DQNBot
                        # bots.DQNBot signature: (pid, policy_net, total_players, device='cpu', epsilon=0.0)
                        self.players.append(DQNBot(pid, policy_net=model, total_players=self.total_players, device='cpu', epsilon=0.0))
                        continue
                    except Exception:
                        # fallback to plain BotClass(pid)
                        pass

                # default instantiation
                self.players.append(BotClass(pid))

        self.game = LiarsDiceGame(self.players, save_history=False)

        self.opponent_beliefs = {
            pid: OpponentBelief() 
            for pid in range(self.total_players) 
            if pid != self.rl_id
        }

    def reset(self, starting_player: int = None):
        # restart game without having to rebuild player list

        self.game = LiarsDiceGame(self.players, save_history = False)

        if starting_player is None:
            starting_player = random.randrange(self.total_players)

        self.game.deal(starting_player=starting_player)

        for belief in self.opponent_beliefs.values():
            # reset dice roll belief to uniform prior
            belief.alpha[:] = 1.0
        
        _, done = self._advance_to_rl_turn()

        if done: 
            return self._observe()
        
        state = self._observe()

        assert len(state) == MAX_STATE_DIM, (
            f"State dimension mismatch during reset: got {len(state)} expected {MAX_STATE_DIM}"
        )

        return state

    
    def _observe(self):
        total_dice = sum(self.game.dice_counts)
        faces = self.game.dice[self.rl_id]
        agent_dice_vec = [faces.count(f) for f in range(1, FACE_COUNT+1)]
        agent_dice_count = self.game.dice_counts[self.rl_id]
        current_bid = self.game.current_bid


        opponent_beliefs_vecs = []
        for pid in range(self.total_players):
            if pid == self.rl_id:
                continue
            belief = self.opponent_beliefs[pid]
            opponent_beliefs_vecs.append(belief.sample_belief().tolist())

        terminal_flag = int(self.game.is_game_over()) # TODO

        state =  encode_rl_state(
            total_dice=total_dice,
            agent_dice_count=agent_dice_count,
            agent_dice_vector=agent_dice_vec,
            current_bid=current_bid,
            opponent_beliefs=opponent_beliefs_vecs,
            terminal_flag=terminal_flag
        )

        assert len(state) == MAX_STATE_DIM, (
            f"State dimension mismatch: got {len(state)}, expected {MAX_STATE_DIM}. "
            f"Check padding, encoding, or MAX_PLAYERS settings."
        )


        return state
    
    def step(self, action_index: int):
        """
        One RL step:
        - RL-bot acts (using decoded action)
        - environment simulates exactly one following opponent action
        - Dirichlet beliefs updated on opponent bids
        - reward computed from this 2-move transition
        """
        
        assert self.game.current_player == self.rl_id, \
            f"Env.step called when it's not RL's turn (current_player={self.game.current_player})"
        

        action = decode_rl_action(action_index)

        reward = 0.0
        done = False
        info = {}

        # RL action
        rl_res = self.game.step(self.rl_id, ("call", None) if action == "call" else ("bid", action))

        if action != 'call':
            self._update_all_beliefs_from_bid(self.rl_id, action)

        # === CASE 1: RL called bluff ===
        if action == "call":
            if isinstance(rl_res, dict) and not rl_res.get("error"):
                reward += self._reward_from_resolution(
                        rl_res,
                        rl_action=action,
                        rl_was_caller=True
                    )
                if self.game.is_game_over():
                    reward += self._terminal_reward()
                    done = True
                    return self._observe(), reward, done, info
                
                r2, d2 = self._advance_to_rl_turn()
                return self._observe(), reward + r2, d2, info
            
            else:
                raise RuntimeError("RL attempted CALL but resolution dict not returned.")
            

        # === CASE 2: RL PLACED A BID ===

        # If game ended from RL action (shouldn't happen)
        if self.game.is_game_over():
            reward += self._terminal_reward()
            done = True
            return self._observe(), reward, done, info
        
        opp_id = self.game.current_player

        if opp_id == self.rl_id:
            return self._observe(), reward, done, info

        opp_bot = self.players[opp_id]
        opp_act = opp_bot.act(self.game)

        if opp_act[0] == "bid":
            #self.opponent_beliefs[opp_id].update_from_bid(opp_act[1])
            self._update_all_beliefs_from_bid(opp_id, opp_act[1])

        opp_res = self.game.step(opp_id, opp_act)

        resolution = opp_res if (isinstance(opp_res, dict) and not opp_res.get("error")) else None
                
        reward += self._reward_from_transition(
                    rl_action=action,
                    opp_id=opp_id,
                    opp_action=opp_act,
                    resolution=resolution
                )

        # If the opponent called:
        if resolution is not None:

            if self.game.is_game_over():
                reward += self._terminal_reward()
                return self._observe(), reward, True, info

            # Round ended; simulate until RL turn
            r2, d2 = self._advance_to_rl_turn()
            return self._observe(), reward + r2, d2, info

        # Otherwise: opponent bid → RL got +100 → now simulate full game until RL turn
        r2, d2 = self._advance_to_rl_turn()
        return self._observe(), reward + r2, d2, info
    
    def _update_all_beliefs_from_bid(self, bidder_pid: int, bid):
        """
        Update opponent-belief objects for:
          - the RL agent (self.opponent_beliefs)
          - any bots that expose .opponent_beliefs (e.g., DQNBot)
        """

        # Update RL agent's belief
        if bidder_pid in self.opponent_beliefs:
            self.opponent_beliefs[bidder_pid].update_from_bid(bid)

        # Update any DQNBots that track opponent beliefs
        for bot in self.players:
            if bot is None:
                continue
            if hasattr(bot, "opponent_beliefs") and bidder_pid in bot.opponent_beliefs:
                bot.opponent_beliefs[bidder_pid].update_from_bid(bid)

        

        
    def _terminal_reward(self) -> float:
        winner = self.game.get_winner()
        if winner == self.rl_id:
            return 5000.00
        if self.game.dice_counts[self.rl_id] == 0:
            return -5000.00
        return 0.0
    
    def _reward_from_resolution(self, res: dict, rl_action, rl_was_caller: bool) -> float:

        """
        res:
            {
              "bid": (q, f),
              "actual": actual,
              "winner": pid,
              "loser": pid,
              ...
            }
        """

        r = 0.0

        if rl_was_caller:
            if res["winner"] == self.rl_id:
                r += 500.0
            elif res["loser"] == self.rl_id:
                r -= 1000.0
            
        else:
            pass # handled by _reward_from_transition

        return r
    
    def _reward_from_transition(self, rl_action, opp_id, opp_action, resolution):
        """
        rl_action: [q,f] (assume no call)
        opp_action: ("call", None) or ("bid",[q,f])
        resolution: None if no call; dict if call resolved round
        """

        r = 0.0

        # If no call, RL made bid that is not called
        if opp_action[0] == "bid" and resolution is None:
            r += 100.0
            return r
        
        if isinstance(resolution, dict) and not resolution.get("error"):
            if resolution["loser"] == self.rl_id:
                # bid was called and lost
                r -= 500.0

        return r
    
    def _advance_to_rl_turn(self):
        """
            Simulate opponents until either:
            - it becomes RL's turn, or
            - the game ends.
            No RL-specific reward is assigned inside this loop.

            Returns (reward, done)
        """

        reward = 0.0
        done = False

        while True:

            if self.game.is_game_over():
                reward += self._terminal_reward()
                done = True
                break

            pid = self.game.current_player

            if pid == self.rl_id:
                break
            
            # Skiping eliminated players
            if self.game.dice_counts[pid] == 0:
                self.game.current_player = (pid + 1) % self.total_players
                continue
            
            bot = self.players[pid]
            act = bot.act(self.game)

            if act[0] == "bid":
                #assert pid in self.opponent_beliefs, f"Missing belief for pid={pid}"
                #self.opponent_beliefs[pid].update_from_bid(act[1])
                self._update_all_beliefs_from_bid(pid, act[1])


            self.game.step(pid, act)

        return reward, done
    

def get_legal_action_indices(state_vec):
        """
        Given an encoded RL state vector, return all legal action indices
        (same format as env.get_legal_action_indices(), but independent of env).
        """
        

        total_dice = state_vec[0]
        current_bid_q = state_vec[8]
        current_bid_f = state_vec[9]
        terminal_flag = state_vec[-1]

        current_bid = [current_bid_q, current_bid_f]
        
        legal_actions = []

        # no legal actions from terminal state
        if terminal_flag == 1 or total_dice <= 1:
            return np.array([], dtype = np.int64)
        
        # call is legal if someone has bid
        if (current_bid_q > 0) and (current_bid_f > 0):
                call_idx = encode_rl_action("call")
                legal_actions.append(call_idx)

        for q in range(1, total_dice + 1):
            for f in range(1, FACE_COUNT + 1):
                if is_bid_higher(current_bid, [q, f]):
                    idx = encode_rl_action([q, f])
                    legal_actions.append(idx)
        
        return np.array(legal_actions, dtype=np.int64)
