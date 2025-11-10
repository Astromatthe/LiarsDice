from config import N_PLAYERS, DICE_PER_PLAYER
from typing import List, Tuple

def encode_game_state(
        dice_belief: List[List[int]], 
        current_bid: List[int],
        dice_actual: List[List[int]],
        current_player: int
    ) -> List[int]:
    """Encodes the game state into a flat list of integers."""
    code = []
    # code is filled from left to right
    # beliefs
    for player in dice_belief:
        for die in player:
            code.append(die)    # 0 is no die (no zero-indexing)

    # actual dice
    for player in dice_actual:
        for die in player:
            code.append(die)    # 0 is no die (no zero-indexing)
    
    # current player
    code.append(current_player)
        
    # current bid
    code.extend(current_bid)    # 2 elements

    # add more if needed

    return code

def decode_game_state(code: List[int]) -> Tuple[
    List[List[int]], 
    List[int], 
    List[List[int]], 
    int]:
    """Decodes the flat list of integers back into the game state."""
    index = 0
    dice_belief = []
    for _ in range(N_PLAYERS):
        player_dice = []
        for _ in range(DICE_PER_PLAYER):
            player_dice.append(code[index])
            index += 1
        dice_belief.append(player_dice)

    dice_actual = []
    for _ in range(N_PLAYERS):
        player_dice = []
        for _ in range(DICE_PER_PLAYER):
            player_dice.append(code[index])
            index += 1
        dice_actual.append(player_dice)

    current_player = code[index]
    index += 1

    current_bid = [code[index], code[index + 1]]
    index += 2

    return dice_belief, dice_actual, current_player, current_bid

def get_reduced_state(
        dice_belief: List[List[int]], 
        current_bid: List[int],
        current_player: int
    ) -> List[int]:
    """Generates a reduced state representation for the current player."""
    code = []
    # beliefs
    for player in dice_belief:
        for die in player:
            code.append(die)    # 0 is no die (no zero-indexing)
    
    # current player
    code.append(current_player)
        
    # current bid
    code.extend(current_bid)    # 2 elements

    return code