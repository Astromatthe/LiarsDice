from typing import List, Tuple
from config import N_PLAYERS, DICE_PER_PLAYER, TOTAL_DICE
from src.game import LiarsDiceGame
from src.bots import RandomBot, RiskAverseBot, RiskyBot
from src.beliefs import OpponentBelief
from src.encode import encode_rl_state, encode_rl_action, decode_rl_action
import random
import numpy as np


class LiarsDiceEnv:
    def __init__(self, rl_id:int = 0):
        self.rl_id = rl_id
        self.players = [None]*N_PLAYERS

        # Right now all other players are random bots. 
        # Change later to allow different learning modes.
        for pid in range(N_PLAYERS):
            if pid==rl_id:
                self.players[pid] = None
            else:
                self.players[pid] = RandomBot(pid)
        
        self.game = LiarsDiceGame(self.players)

        # Each non-rlbot player gets uniform dirichlet prior
        self.opponent_beliefs = {
            pid: OpponentBelief()
            for pid in range(N_PLAYERS) if pid != self.rl_id
        }

    def reset(self, starting_player: int = None):
        # restart game without having to rebuild player list

        self.game = LiarsDiceGame(self.players)

        if starting_player is None:
            starting_player = random.randrange(N_PLAYERS)

        self.game.deal(starting_player=starting_player)

        for belief in self.opponent_beliefs.values():
            # reset dice roll belief to uniform prior
            belief.alpha[:] = 1.0
        
        _, done = self._advance_to_rl_turn()

        if done: 
            return self._observe()

        return self._observe()

    
    def _observe(self):
        total_dice = sum(self.game.dice_counts)
        agent_dice_vec = self.game.dice[self.rl_id]
        agent_dice_count = self.game.dice_counts[self.rl_id]
        current_bid = self.game.current_bid

        opponent_beliefs_vecs = []
        for pid in range(N_PLAYERS):
            if pid == self.rl_id:
                continue
            belief = self.opponent_beliefs[pid]
            opponent_beliefs_vecs.append(belief.sample_belief().tolist())

        terminal_flag = int(self.game.is_game_over()) # TODO

        return encode_rl_state(
            total_dice=total_dice,
            agent_dice_count=agent_dice_count,
            agent_dice_vector=agent_dice_vec,
            current_bid=current_bid,
            opponent_beliefs=opponent_beliefs_vecs,
            terminal_flag=terminal_flag
        )
    
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
        opp_bot = self.players[opp_id]
        opp_act = opp_bot.act(self.game)

        if opp_act[0] == "bid":
            self.opponent_beliefs[opp_id].update_from_bid(opp_act[1])

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

            if self.game.current_player == self.rl_id:
                break

            pid = self.game.current_player
            bot = self.players[pid]
            act = bot.act(self.game)

            if act[0] == "bid":
                self.opponent_beliefs[pid].update_from_bid(act[1])

            res = self.game.step(pid, act)

            continue

        return reward, done
    
    def get_legal_action_indices(self):
        """
        Return a numpy array of integer action indices that are legal from the current
        game state for the RL agent.
        """

        legal_indices = []

        game = self.game
        rl_id = self.rl_id

        if game.current_bid not in (None, [0, 0], (0, 0)):
            call_idx = encode_rl_action(("call", None))
            legal_indices.append(call_idx)

        legal_bids = game.get_legal_bids()

        for bid in legal_bids:
            idx = encode_rl_action(("bid",bid))
            legal_indices.append(idx)

        return np.array(legal_indices, dtype = np.int64)


