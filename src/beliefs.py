import numpy as np

class OpponentBelief:
    def __init__(self):
        self.alpha = np.ones(6)

    def update_from_bid(self, bid):
        
        _, face = bid
        self.alpha[face - 1] += 1
    
    def sample_belief(self):
        alpha = np.array(self.alpha)
        return alpha / np.sum(alpha)