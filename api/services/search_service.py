import io
import importlib

import numpy as np
import matplotlib.pyplot as plt

from search.agent import Agent
from search.utils import lon_lat_to_xy, xy_to_lon_lat, rssi_normalize
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
    
    def init_params(self, current_loc, grid_size, num_canvas, **kwargs):
        """
        Set the parameters for the agent to start a new search.
        
        Args:
            current_loc (np.array): Current location of the agent (lon, lat)
            grid_size (int): Size of the grid in meters
            num_canvas (int): Number of grids in the canvas, needed to match with the heatmap model output if needed
        """
        self.grid_size = grid_size
        self.num_canvas = num_canvas
        self.radius = grid_size * num_canvas / 2
        self.origin_loc = current_loc
        self.agent.set_loc([0, 0])
        self.rssi_history = []

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
        self.ready = False  # Reset readiness when changing agent
        self.rssi_history = []  # Clear RSSI history when changing agent

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
        plt.imshow(self.last_heat_map, cmap='viridis')
        plt.axis('off')  # Turn off the axis
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return buf
        
 

    def get_next_target(self, current_loc, rssi, **kwargs):
        """
        Get the next action for the agent based on the current RSSI value and location.
        
        Args:
            current_loc (np.array): Current location of the agent (lon, lat)
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
            lon=current_loc[0],
            lat=current_loc[1],
            origin_lon=self.origin_loc[0],
            origin_lat=self.origin_loc[1]
        )

        # Get the current row and column in the grid based on the current location
        current_row = int(x / self.grid_size)
        current_col = int(y / self.grid_size)
        
        # Raise an error if the agent is outside the search area
        if abs(x) > self.radius or abs(y) > self.radius:
            raise ValueError("Agent is outside the search area")
        
        if self.heatmap_model_needed:
            # Check the grid size and the number of canvas to match with the heatmap model output
            heatmap_model_version = kwargs.get('model_version', 'v1') # TODO: Default based on config instead of hardcoding
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

        # Convert the new location to lon, lat
        new_x = self.agent.location[0] * self.grid_size
        new_y = self.agent.location[1] * self.grid_size

        return xy_to_lon_lat(
            x=new_x,
            y=new_y,
            origin_lon=self.origin_loc[0],
            origin_lat=self.origin_loc[1]
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
        

if __name__ == "__main__":
    model_service = ModelService()
    search_service = SearchService(
        model_service,
        agent='heatmap_greedy'
    )

    prev_loc = origin_loc = [22.084, 37.422]

    search_service.init_params(
        current_loc=origin_loc,
        grid_size=250,
        num_canvas=56
    )


    additional_params = {
        'model_version': 'v1',
        'guide_weight': 1.2345,
    }

    for rssi in [-119, -120, -110]:
        new_loc = search_service.get_next_target(
            current_loc=prev_loc,
            rssi=rssi,
            **additional_params
        )
        print(new_loc)
        prev_loc = new_loc
    
    print(search_service.agent.location)
    print(type(search_service.get_last_heatmap()))
    # Save the last heatmap image
    # buf = search_service.get_last_heatmap_image()
    # with open('heatmap.png', 'wb') as f:
    #     f.write(buf.read())
    # # Example of switching agents
    # search_service.set_agent('greedy')
    # search_service.init_params(
    #     current_loc=origin_loc,
    #     grid_size=250,
    #     num_canvas=56
    # )
    # for rssi in [-120, -115, -110, -105, -100, -95, -90]:
    #     new_loc = search_service.get_next_target(
    #         current_loc=prev_loc,
    #         rssi=rssi
    #     )
    #     print(new_loc)
    #     prev_loc = new_loc