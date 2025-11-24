import random
import torch
import numpy as np
from rl_env import LiarsDiceEnv


def select_action(env, policy_net, state, epsilon):
    """
    Epsilon-greedy action selection with illegal action masking.
    
    Arguments:
        env         : LiarsDiceEnv instance (must have get_legal_action_indices())
        policy_net  : DQN model mapping state → Q-values (one per action index)
        state       : encoded RL state (numpy array or list)
        epsilon     : float in [0,1]
        
    Returns:
        action_index (int) — a legal action index
    """

    legal_actions = env.get_legal_action_indices()

    # Exploration case
    if random.random() < epsilon:
        return int(random.choice(legal_actions))
    
    # Exploitation
    # Convert state to torch tensor [1, state_dim]
    state_t = torch.tensor(state, dtype = torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(state_t).squeeze(0)

    # illegal actions initialized to -inf
    masked_q = torch.full_like(q_values, float("-inf"))


    legal_t = torch.tensor(legal_actions, dtype=torch.long)

    masked_q[legal_t] = q_values[legal_t]

    # Select highest scoring legal action
    best_action = int(torch.argmax(masked_q).item())

    return best_action

    



    
