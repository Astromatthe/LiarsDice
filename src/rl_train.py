import random
from collections import deque
import torch
import numpy as np
from rl_env import LiarsDiceEnv, get_legal_action_indices
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import NUM_ACTIONS

class DQN(nn.Module):
    """
    Simple feedforward network for approximating Q(s,a).
    
    Inputs:
        state_dim  : length of encoded state vector
        action_dim : number of discrete actions (NUM_ACTIONS)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()

        # First fully-connected layer: from state_dim -> hidden_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # Second fully-connected layer: hidden_dim -> hidden_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer: hidden_dim -> action_dim (one Q-value per action)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        x: Tensor of shape [batch_size, state_dim]
           (or [state_dim], which we’ll handle)
        Returns:
           Tensor of shape [batch_size, action_dim]
           containing Q-values for each action.
        """

        # If a single state comes in as [state_dim], add batch dim -> [1, state_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Layer 1 + ReLU
        x = F.relu(self.fc1(x))

        # Layer 2 + ReLU
        x = F.relu(self.fc2(x))

        # Output layer (no activation: raw Q-values)
        q_values = self.out(x)

        return q_values



def select_action(policy_net, state_vec, epsilon):
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

    legal_actions = get_legal_action_indices(state_vec)

    if len(legal_actions) == 0:
        return None

    # Exploration case
    if random.random() < epsilon:
        return int(random.choice(legal_actions))
    
    # Exploitation
    # Convert state to torch tensor [1, state_dim]
    state_t = torch.tensor(state_vec, dtype = torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(state_t).squeeze(0)

    # illegal actions initialized to -inf
    masked_q = torch.full_like(q_values, float("-inf"))

    legal_t = torch.tensor(legal_actions, dtype=torch.long)

    masked_q[legal_t] = q_values[legal_t]

    # Select highest scoring legal action
    best_action = int(torch.argmax(masked_q).item())

    return best_action


def optimize_model(memory, batch_size, policy_net, target_net, optimizer, gamma):
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    s_batch, a_batch, r_batch, sp_batch, done_batch = zip(*batch)

    state_batch = torch.tensor(np.array(s_batch), dtype=torch.float32)
    action_batch = torch.tensor(a_batch, dtype=torch.long).unsqueeze(1)
    reward_batch = torch.tensor(r_batch, dtype=torch.float32)
    next_state_batch = torch.tensor(np.array(sp_batch), dtype=torch.float32)
    done_batch = torch.tensor(done_batch, dtype=torch.float32)

    q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    with torch.no_grad():
        q_next_all = target_net(next_state_batch)

        max_next_q_list = []

        for i in range(batch_size):
            sp_vec = sp_batch[i]
            legal_next = get_legal_action_indices(sp_vec)

            if len(legal_next) == 0:
                max_next_q_list.append(0.0)
                continue

            q_next_i = q_next_all[i]
            masked_q_i = torch.full_like(q_next_i, float("-inf"))
            masked_q_i[legal_next] = q_next_i[legal_next]

            max_next_q = torch.max(masked_q_i).item()
            max_next_q_list.append(max_next_q)
        
        max_next_q_values = torch.tensor(max_next_q_list, dtype = torch.float32)

        target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)
        

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_dqn(
    episodes,
    batch_size,
    learning_rate,
    gamma,
    epsilon_start,
    epsilon_min,
    epsilon_decay,
    target_update_freq,
    memory_size,
    device="cpu",
):
    env = LiarsDiceEnv()

    init_state = env.reset()
    state_dim = len(init_state)

    policy_net = DQN(state_dim, NUM_ACTIONS).to(device)
    target_net = DQN(state_dim, NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr = learning_rate)

    memory = deque(maxlen=memory_size)

    epsilon = epsilon_start

    step_count = 0

    ### Main Training Loop ###

    for episode in range(episodes):

        state = env.reset()
        done = False

        while not done:
            # epsilon-greedy action selection
            action_index = select_action(policy_net, state, epsilon)

            if action_index is None:
                break

            next_state, reward, done, _ = env.step(action_index)

            memory.append((state, action_index, reward, next_state, done))

            state = next_state
            step_count += 1

            optimize_model(                  
                memory=memory,
                batch_size=batch_size,
                policy_net=policy_net,
                target_net=target_net,
                optimizer=optimizer,
                gamma=gamma
            )

            if step_count % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return policy_net, target_net


        
        

        

    
