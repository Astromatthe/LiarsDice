from config import FACE_COUNT
from typing import List


def encode_rl_state(
        total_dice: int, 
        agent_dice_count: int,
        agent_dice_vector: List[int],
        current_bid: List[int],
        opponent_beliefs: List[List[int]],
        terminal_flag: int

) -> List[float]:
    """Producuced DQN input vector"""

    state = []
    

    state.append(total_dice)
    state.append(agent_dice_count)
    state.extend(agent_dice_vector)
    state.extend(current_bid)
    
    for belief in opponent_beliefs:
        state.extend(belief)
    
    state.append(terminal_flag)


    return state

def decode_rl_state(
        flat: List[float],
        n_opponents: int
):
    """
    Reconstructs the RL state for debugging or rollout inspection.
    """

    idx = 0

    total_dice = flat[idx]; idx += 1
    agent_dice_count = flat[idx]; idx += 1

    agent_dice_vector = flat[idx:idx+6]; idx += 6

    current_bid = flat[idx:idx+2]; idx += 2

    opponent_beliefs = []
    for _ in range(n_opponents):
        opponent_beliefs.append(flat[idx:idx+6])
        idx += 6

    terminal_flag = flat[idx]

    return(total_dice, agent_dice_count, agent_dice_vector, current_bid, opponent_beliefs, terminal_flag)


def encode_rl_action(action):

    """
    Convert an action into a fixed integer index.
    
    Accepts:
        action = "call"
        OR
        action = [q, f] bid
    
    Returns:
        integer index (0 to ACTION_DIM-1)
    """

    if action == "call":
        return 0
    
    q,f = action

    bid = 1 + (q-1) * FACE_COUNT + (f-1)

    return bid

def decode_rl_action(index):

    """
    Convert an integer index back to an action tuple
    used by the game environment.
    
    Returns:
        "call"
        OR
        [q, f]
    """

    if index == 0:
        return 'call'
    
    shifted = index - 1

    q = shifted // FACE_COUNT + 1
    f = shifted % FACE_COUNT + 1

    return [q,f]
