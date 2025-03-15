import numpy as np

class Agent: 
    def __init__(self, init_loc: np.array = [0, 0]):
        self.location = np.array(init_loc)
        self.complete = False # True if the search policy has taken effects
    
    def set_loc(self, loc: np.array):
        self.location = np.array(loc)

    def update_loc(self, action: np.array):
        self.location = self.location + np.array(action)
    
    def update_state(self, action, rssi):
        """Override this method to update the agent's state"""
        raise NotImplementedError;

    def action(self, rssi):
        """Override this method to return the agent's action"""
        raise NotImplementedError;

    def reset(self, loc = [0, 0]):
        self.location = np.array(loc)
        self.complete = False