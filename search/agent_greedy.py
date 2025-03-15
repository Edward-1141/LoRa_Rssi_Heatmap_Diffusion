import random

import numpy as np

from .agent import Agent
from .utils import rand_action

# TODO: 
# 1. What if action leads to out of bound? (Possible solution: ignore the action and reset the agent?)
# 2. It direct update state after getting the action, this should be done after the action is taken

class AgentGreedy(Agent):
    def __init__(self, explore=0.2, decay=0.2):
        self.explore = explore  # probability for exploration
        self.decay = decay       # decay for momentum 
        
        self.mot = np.array([0, 0]) 
        self._last_rssi = None
        self._last_action = None
        super().__init__()

    def update_state(self, action, rssi): 
        self._last_action = action
        self._last_rssi = rssi

    def action(self, rssi, **kwargs):
        # init 
        if self._last_rssi == None:
            action = rand_action()
            self.update_state(action, rssi)
            return action
    
        self.mot = self.mot * self.decay + (rssi-self._last_rssi) * self._last_action

        if np.linalg.norm(self.mot) == 0:
            action = rand_action()
        else:
            # mot to action
            self.complete = True
            action = np.array([0, 0])
            max_idx = int(self.mot[1]>self.mot[0])
            dim = 1-max_idx if random.random()<self.explore else max_idx
            if self.mot[dim]>0:
                action[dim]=1
            elif self.mot[dim]<0:
                action[dim]=-1
            else:
                action[dim] = random.choice([-1, 1])
        self.update_state(action, rssi)
        
        return action
    
    def reset(self, loc=[0, 0]):
        self.mot = np.array([0, 0])
        self._last_rssi = None
        self._last_action = None
        self.complete = False

        super().reset(loc)
