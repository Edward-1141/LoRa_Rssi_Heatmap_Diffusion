import random
import numpy as np

from .agent import Agent
from .utils import rand_action

# TODO: It direct update state after getting the action, this should be done after the action is taken

class AgentHeatmapGreedy(Agent):
    def __init__(self, explore=0.2, visited_penalty=0.5, temperature=1.0, max_history=100):
        """
        Initialize the heatmap-based greedy agent.
        
        Args:
            explore (float): Probability of taking a random action
            visited_penalty (float): Penalty factor for visited locations (0-1)
            temperature (float): Controls randomness in hotspot selection (higher = more random)
            max_history (int): Maximum number of past locations to remember
        """
        self.explore = explore
        self.visited_penalty = visited_penalty
        self.temperature = temperature
        self.max_history = max_history
        
        self.visited = set()  # Set of visited locations (row, col)
        self.visit_count = {}  # Count of visits per location
        self._last_rssi = None
        self._last_action = None
        self._last_heatmap = None
        super().__init__()
    
    def update_state(self, action, rssi, **kwargs): 
        self._last_action = action
        self._last_rssi = rssi
        
        # Record current location as visited
        loc_tuple = tuple(self.location)
        self.visited.add(loc_tuple)
        
        # Update visit count for current location
        self.visit_count[loc_tuple] = self.visit_count.get(loc_tuple, 0) + 1
        
        # Limit history size by removing oldest entries if needed
        if len(self.visited) > self.max_history:
            oldest = next(iter(self.visited))
            self.visited.remove(oldest)
            if oldest in self.visit_count:
                del self.visit_count[oldest]
        
        # Store heatmap if provided
        if 'heatmap' in kwargs:
            self._last_heatmap = kwargs['heatmap']

    def action(self, rssi, **kwargs):
        """
        Choose the next action based on rssi and predicted heatmap.
        
        Args:
            rssi (float): Current RSSI value
            **kwargs: Additional parameters, including 'heatmap' as a numpy array
            
        Returns:
            np.array: The selected action as [dx, dy]
        """
        # Get heatmap if provided
        heatmap = kwargs.get('heatmap', self._last_heatmap)
        offset_row = heatmap.shape[0] // 2
        offset_col = heatmap.shape[1] // 2
        
        # If no heatmap is available or with probability explore, take random action
        if heatmap is None or random.random() < self.explore:
            action = rand_action()
            self.update_state(action, rssi, **kwargs)
            return action
        
        # Current position in the grid (from -offset to offset (-1))
        curr_row, curr_col = self.location
        
        # Create a modified heatmap with penalties for visited locations
        modified_heatmap = heatmap.copy()
        
        # Apply penalty to visited locations
        for (r, c) in self.visited:
            # Translate visited location to heatmap indices
            heatmap_r = r + offset_col
            heatmap_c = c + offset_row
            # Check if location is within heatmap bounds
            if 0 <= heatmap_r < heatmap.shape[0] and 0 <= heatmap_c < heatmap.shape[1]:
                visits = self.visit_count.get((r, c), 1)
                modified_heatmap[heatmap_r, heatmap_c] *= (1 - self.visited_penalty) ** visits
        
        # Find the hottest spot in the modified heatmap
        # Using softmax-weighted sampling to allow for some randomness in selection
        flattened = modified_heatmap.flatten()
        softmax_values = np.exp(flattened / self.temperature)
        softmax_values = softmax_values / np.sum(softmax_values)
        
        # Sample from the softmax distribution
        choice_idx = np.random.choice(len(flattened), p=softmax_values)
        hotspot_row, hotspot_col = np.unravel_index(choice_idx, modified_heatmap.shape)
        
        # Translate hotspot indices back to agent coordinates
        hotspot_row -= offset_col
        hotspot_col -= offset_row
        
        # Determine the action to move towards the hotspot
        dr = 0
        dc = 0
        
        # Determine vertical movement (prioritize the dimension with larger difference)
        row_diff = hotspot_row - curr_row
        col_diff = hotspot_col - curr_col
        
        if abs(row_diff) > abs(col_diff):
            dr = 1 if row_diff > 0 else -1 if row_diff < 0 else 0
        else:
            dc = 1 if col_diff > 0 else -1 if col_diff < 0 else 0
            
        # If we're already at the hotspot or both dimensions have equal differences,
        # choose one randomly
        if dr == 0 and dc == 0:
            if row_diff != 0:
                dr = 1 if row_diff > 0 else -1
            elif col_diff != 0:
                dc = 1 if col_diff > 0 else -1
            else:
                # We're exactly at the hotspot, take a random action
                action = rand_action()
                self.update_state(action, rssi, **kwargs)
                return action
        
        action = np.array([dr, dc])
        self.update_state(action, rssi, **kwargs)
        return action
    
    def reset(self, loc=[0, 0]):
        """Reset the agent state."""
        self.visited = set()
        self.visit_count = {}
        self._last_rssi = None
        self._last_action = None
        self._last_heatmap = None
        self.complete = False
        
        super().reset(loc)