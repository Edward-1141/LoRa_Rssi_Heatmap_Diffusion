import random
import math

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
            visited_penalty (float): Penalty factor for visited locations which should be positive (Not used yet)
            temperature (float): Controls randomness in hotspot selection (higher = more random)
            max_history (int): Maximum number of past locations to remember
        """
        # TODO: Add check for visited_penalty we use it in the future
        # if visited_penalty < 0:
        #     raise ValueError("visited_penalty should be positive")
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
        
        # Find the hottest spot
        flattened = heatmap.flatten()
        softmax_values = np.exp(flattened / self.temperature)
        softmax_values = softmax_values / np.sum(softmax_values)
        
        # Sample from the softmax distribution
        choice_idx = np.random.choice(len(flattened), p=softmax_values)
        hotspot_row, hotspot_col = np.unravel_index(choice_idx, heatmap.shape)
        
        # Translate hotspot indices back to agent coordinates
        hotspot_row -= offset_col
        hotspot_col -= offset_row
        
        # Determine possible actions and their scores
        possible_actions = []
        action_scores = []
        
        # Check all four possible actions
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # right, down, left, up
            next_row = curr_row + dr
            next_col = curr_col + dc
            next_loc = (next_row, next_col)
            
            # Calculate score based on direction towards hotspot
            row_diff = hotspot_row - curr_row
            col_diff = hotspot_col - curr_col
            
            # Calculate directional score
            def get_directional_score(move_dir, target_diff):
                """Calculate score for a direction based on whether it moves towards or away from target."""
                def sigmoid(x):
                    return 1 / (1 + math.exp(-x))

                if move_dir == 0:
                    return 0
                # Higher score for moving in correct direction with larger magnitude
                x = abs(target_diff) if (move_dir * target_diff) > 0 else -abs(target_diff)
                x = x / (1 + abs(x)) # prevent sigmoid saturation
                return sigmoid(x)
            
            # Combine scores from both dimensions
            score = get_directional_score(dr, row_diff) + get_directional_score(dc, col_diff)

            # Apply penalty if next location is visited
            if next_loc in self.visited:
                # Here the penalty is very strong which I want to prioritize exploration
                score -= 2 # since 2*sigmoid(x) is in (0, 2)
                
                # A more tunable version
                # visits = self.visit_count.get(next_loc, 0)
                # score *= math.exp(-self.visited_penalty * visits)

            possible_actions.append((dr, dc))
            action_scores.append(score)
        
        # Choose action with highest score
        best_action_idx = np.argmax(action_scores)
        action = np.array(possible_actions[best_action_idx])
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