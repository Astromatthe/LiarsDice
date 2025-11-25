import numpy as np

class OpponentBelief:
    def __init__(self):
        self.alpha = np.ones(6)

    def update_from_bid(self, bid):
        
        _, face = bid
        self.alpha[face - 1] += 1
    
    def sample_belief(self):
        
        return np.random.dirichlet(self.alpha)
    
    
    
    ###### TODO #######

    """
    When opponents turn

    opponent_beliefs = [
    opponent_belief[i].sample_belief()
    for each opponent i
]
    """
