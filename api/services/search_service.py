import io
import importlib

import numpy as np
import matplotlib.pyplot as plt

from search.agent import Agent
from search.utils import get_lon_lat_limits, lon_lat_to_xy, xy_to_lon_lat, rssi_normalize
from api.services.model_service import ModelService
from api.services.config import SEARCH_CONFIG, HEATMAP_MODEL_CONFIG

class SearchService:
    def __init__(self, model_service: ModelService, agent='heatmap_greedy'):
        self.model_service = model_service
        self.agent = None
        self.agent_type = None
        self.heatmap_model_needed = False
        self.last_heat_map = None
        self.ready = False
        self.rssi_history = []
        self.grid_size = None
        self.num_canvas = None
        self.radius = None
        self.origin_loc = None
        self.set_agent(agent)

        # Demo experiment purpose
        self.demo_data = {}
    
    def init_params(self, origin_loc, grid_size, num_canvas, start_loc, **kwargs):
        """
        Set the parameters for the agent to start a new search.
        
        Args:
            origin_loc (np.array): Origin location of the search (lat, lon)
            start_loc (np.array): Starting location of the agent (lat, lon)
            grid_size (int): Size of the grid in meters
            num_canvas (int): Number of grids in the canvas, needed to match with the heatmap model output if needed
        """
        self.grid_size = grid_size
        self.num_canvas = num_canvas
        self.radius = grid_size * num_canvas / 2
        self.origin_loc = origin_loc
        self.rssi_history = []

        # Convert the start location to x, y coordinates (in meters)
        x, y = lon_lat_to_xy(
            lat=start_loc[0],
            lon=start_loc[1],
            origin_lat=self.origin_loc[0],
            origin_lon=self.origin_loc[1],
        )
        # convert the x, y coordinates to the grid index
        self.start_row = int(x / self.grid_size) # -self.num_canvas // 2 - self.grid_size // 2
        self.start_col = int(y / self.grid_size) # -self.num_canvas // 2 - self.grid_size // 2
        self.agent.set_loc([self.start_row, self.start_col])
        self.demo_data["current_loc_idx"] = [self.start_row, self.start_col]

        search_radius = num_canvas // 2 * grid_size
        self.lat_lon_limits = get_lon_lat_limits(
            origin_lat=self.origin_loc[0],
            origin_lon=self.origin_loc[1],
            radius=search_radius
        )
        self.x_min, self.y_min = lon_lat_to_xy(
            lat=self.lat_lon_limits['min_lat'],
            lon=self.lat_lon_limits['min_lon'],
            origin_lat=self.origin_loc[0],
            origin_lon=self.origin_loc[1]
        )

        self.x_max, self.y_max = lon_lat_to_xy(
            lat=self.lat_lon_limits['max_lat'],
            lon=self.lat_lon_limits['max_lon'],
            origin_lat=self.origin_loc[0],
            origin_lon=self.origin_loc[1]
        )
        
        self.ready = True

    def set_agent(self, agent_type, force_reload=False):
        """
        Set the agent type dynamically.
        
        Args:
            agent_type (str): The type of agent to use.
        
        Require the agent to be re-initialized after calling this method to start a new search.
        """
        if agent_type not in SEARCH_CONFIG['agent_types']:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        
        if not force_reload and self.agent_type == agent_type:
            return
        
        agent_config = SEARCH_CONFIG['agent_types'][agent_type]
        agent_class = agent_config['class']
        module_name = agent_config['module_name']

        # Import the agent class
        module = importlib.import_module(f"search.{module_name}")
        agent_class = getattr(module, agent_class)
        self.agent = agent_class(**agent_config['params'])
        self.agent_type = agent_type
        self.heatmap_model_needed = agent_config['heatmap-model-needed']
        self.reset()

    def reset(self):
        """
        Reset the agent's state.

        Rquires the agent to be initialized after calling this method to start a new search.
        """
        self.ready = False
        self.rssi_history = []
        self.agent.reset([0, 0])
        
    def get_last_heatmap_image(self):
        """
        Generate the last heatmap as an image in memory.

        Returns:
            BytesIO: The image in memory.
        """
        if self.last_heat_map is None:
            raise ValueError("No heatmap available to generate.")

        # Generate the heatmap image without the color bar
        buf = io.BytesIO()
        plt.imshow(self.last_heat_map, cmap='viridis', interpolation='nearest', origin='lower', extent=[self.x_min, self.x_max, self.y_min, self.y_max])
        plt.colorbar(label='RSSI')
        plt.title('Predicted RSSI distribution')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.scatter(0, 0, color='red', marker='x', s=100) # scatter the origin location for reference
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return buf
        
 

    def get_next_target(self, current_loc, rssi, **kwargs):
        """
        Get the next action for the agent based on the current RSSI value and location.
        
        Args:
            current_loc (np.array): Current location of the agent (lat, lon)
            rssi (float): Current RSSI value
            "model_version" (str): Model version to use for the heatmap (optional)
            "guide_weight" (float): Guide weight for the heatmap (optional)
        
        Returns:
            np.array: The new location of the agent (lon, lat)
        """
        if not self.ready:
            raise ValueError("Agent is not ready to start a new search, please set the parameters first")
        
        action_kwargs = {}

        # Convert the current location to x, y coordinates (in meters)
        x, y = lon_lat_to_xy(
            lat=current_loc[0],
            lon=current_loc[1],
            origin_lat=self.origin_loc[0],
            origin_lon=self.origin_loc[1]
        )

        # Get the current row and column in the grid based on the current location
        current_row = int(x / self.grid_size)
        current_col = int(y / self.grid_size)

        self.demo_data["current_loc_idx"] = [current_row, current_col]
        
        # Raise an error if the agent is outside the search area
        if abs(x) > self.radius or abs(y) > self.radius:
            raise ValueError("Agent is outside the search area")
        
        if self.heatmap_model_needed:
            # Check the grid size and the number of canvas to match with the heatmap model output
            heatmap_model_version = kwargs.get('model_version', 'v1') # TODO: Default based on config instead of hardcoding
            if heatmap_model_version not in HEATMAP_MODEL_CONFIG['model_types']:
                raise ValueError(f"Unsupported model version: \"{heatmap_model_version}\"")
            if HEATMAP_MODEL_CONFIG['model_types'][heatmap_model_version]['output_dim'] != [self.num_canvas, self.num_canvas]:
                raise ValueError("The heatmap model output dimension doesn't match with the search area")

            offset_row = self.num_canvas // 2
            offset_col = self.num_canvas // 2

            self.rssi_history.append((
                current_row + offset_row,
                current_col + offset_col,
                rssi_normalize(rssi)
            ))
            self.last_heat_map = self.model_service.generate_heatmap(
                rssi_list=self.rssi_history,
                model_version=heatmap_model_version,
                guide_weight=kwargs.get('guide_weight', 2.0) # TODO: Default based on config instead of hardcoding
            )
            action_kwargs['heatmap'] = self.last_heat_map

                  
        action = self.agent.action(rssi, **action_kwargs)
        self.agent.update_loc(action)

        self.demo_data["action"] = action.tolist()
        self.demo_data["next_loc_idx"] = self.agent.location.tolist()

        # Convert the new location to lon, lat
        new_x = self.agent.location[0] * self.grid_size
        new_y = self.agent.location[1] * self.grid_size

        return xy_to_lon_lat(
            x=new_x,
            y=new_y,
            origin_lat=self.origin_loc[0],
            origin_lon=self.origin_loc[1]
        )

    def get_last_heatmap(self):
        """
        Get the last heatmap used by the agent.

        Returns:
            np.array: The last heatmap used by the agent or None if not available
        """
        return self.last_heat_map
        

    def get_available_agents(self):
        """
        Get the list of available agent types."

        Returns:
            list: List of available agent types
        """
        return list(SEARCH_CONFIG['agent_types'].keys())
        