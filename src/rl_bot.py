from src.players import Player
from src.encode import encode_rl_state, encode_rl_action
from src.game import LiarsDiceGame
from src.beliefs import OpponentBelief


class DQNBot(Player):
    def __init__(self, name, model, player_id, n_players):
        super().__init__(name)
        self.model = model
        self.id = player_id
        self.n_players = n_players

        # one belief object per opponent, keyed by player id for simplicity
        self.opponent_beliefs = {
            pid: OpponentBelief()
            for pid in range(n_players)
            if pid != self.id
        }
    
    def get_legal_actions(self, game):
        legal_bids = game.get_legal_bids()

        legal_actions = ["call"] + legal_bids

        legal_action_indices = [encode_rl_action(a) for a in legal_actions]

        return legal_action_indices

    def get_observation(self, game):
        
        total_dice = sum(game.dice_counts)

        
        agent_dice_list = game.dice[self.id]
        agent_dice_count = game.dice_counts[self.id]
        agent_dice_vector = dice_to_vector(agent_dice_list)

        # current bid [q, f]
        current_bid = game.current_bid

        # sample beliefs for each opponent in a consistent player-id order
        opponent_belief_vectors = []
        for pid in range(self.n_players):
            if pid == self.id:
                continue
            belief_vec = self.opponent_beliefs[pid].sample_belief()
            opponent_belief_vectors.append(belief_vec)

        # terminal flag: 1 = win, -1 = loss, 0 = ongoing
        if game.is_game_over():
            winner = game.get_winner()
            if winner == self.id:
                terminal_flag = 1
            else:
                terminal_flag = -1
        else:
            terminal_flag = 0

        return {
            "total_dice": total_dice,
            "agent_dice_count": agent_dice_count,
            "agent_dice_vector": agent_dice_vector,
            "current_bid": current_bid,
            "opponent_beliefs": opponent_belief_vectors,
            "terminal_flag": terminal_flag,
        }

    def get_state(self, game):
        obs = self.get_observation(game)
        return encode_rl_state(
            obs["total_dice"],
            obs["agent_dice_count"],
            obs["agent_dice_vector"],
            obs["current_bid"],
            obs["opponent_beliefs"],
            obs["terminal_flag"],
        )
    
    def update_beliefs_from_bid(self, bidder_id, bid):
        """
        Update the Dirichlet alpha parameters for the opponent belief model
        based on the bid made by another player.
        
        Each bid [q, f] increments α[f] by +1 for that specific opponent.
        """
        # No update when self
        if bidder_id == self.id:
            return
        
        # update opponents corresponding belief model
        if bidder_id in self.opponent_beliefs:
            self.opponent_beliefs[bidder_id].update_from_bid(bid)

   
def dice_to_vector(dice_list):
    """Convert [1,5,2,0,0] → [1,1,1,0,0,0] (counts of faces 1..6)."""
    vec = [0] * 6
    for d in dice_list:
        if d != 0:
            vec[d - 1] += 1
    return vec







    
    



        






    




