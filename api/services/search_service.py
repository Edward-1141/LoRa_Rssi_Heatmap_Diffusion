import importlib

from search.agent import Agent
from search.utils import lon_lat_to_xy, xy_to_lon_lat, get_lon_lax_limits, rssi_normalize
from api.services.model_service import ModelService
from api.services.config import SEARCH_CONFIG, HEATMAP_MODEL_CONFIG

class SearchService:
    def __init__(self, model_service: ModelService, agent='greedy'):
        self.model_service = model_service
        
        if agent not in SEARCH_CONFIG['agent_types']:
            raise ValueError(f"Unsupported agent type: {agent}")
        
        agent_config = SEARCH_CONFIG['agent_types'][agent]
        agent_class = agent_config['class']
        module_name = agent_config['module_name']

        # Import the agent class
        module = importlib.import_module(f"search.{module_name}")
        agent_class = getattr(module, agent_class)
        self.agent:Agent = agent_class(**agent_config['params'])
        self.heatmap_model_needed = agent_config['heatmap-model-needed']

        self.ready = False # Whether the agent is ready to start a new search
        self.rssi_history = [] # History of RSSI values (row, col, rssi)
        # Required to set the search parameters before starting a new search by calling init_params()
    
    def init_params(self, current_loc, grid_size, num_canvas):
        """
        Set the parameters for the agent to start a new search
        This set the search area and the grid for the agent to move in
        
        Args:
            current_loc (np.array): Current location of the agent (lon, lat)
            grid_size (int): Size of the grid in meters
            num_canvas (int): Number of grids in the canvas, needed to match with the heatmap model output if needed
        
        TODO: Maybe not a must to start in the center of the search area (0, 0)
        """
        # Set the initial location of the agent
        self.grid_size = grid_size
        self.num_canvas = num_canvas
        self.radius = grid_size * num_canvas / 2
        self.origin_loc = current_loc
        self.agent.set_loc([0, 0])

        # These are needed in the future for error checking
        # self.current_loc_xy = [0, 0]
        # self.current_loc = current_loc
        
        self.ready = True
        self.rssi_history = []
       
    def reset(self, current_loc):
        """
        Reset the agent's state
        """
        self.ready = False
        self.agent.reset([0, 0])
        
        if self.grid_size and self.num_canvas:
            self.init_params(current_loc, self.grid_size, self.num_canvas)
    
    def get_next_target(self, current_loc, rssi, **kwargs):
        """
        Get the next action for the agent based on the current RSSI value and location

        Args:
            current_loc (np.array): Current location of the agent (lon, lat)
            rssi (float): Current RSSI value
            "model_version" (str): Model version to use for the heatmap (optional)
            "guide_weight" (float): Guide weight for the heatmap (optional)
        
        Assumption:
        - The agent will always successfully reach the target location, otherwise, it will be reset.
          TODO: Can be improved by adding a response mechanism before updating the agent's location based on the action.
        - The agent is always within the search area.
        """
        print(self.agent.location)

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
        
        # Raise an error if the agent is outside the search area (TODO: Or reset the agent)
        if abs(x) > self.radius or abs(y) > self.radius:
            raise ValueError("Agent is outside the search area")
        # TODO: Raise an error if the agent is not at the correct location,
        #  i.e. 1. it doesn't match with history
        #       2. it doesn't take the correct action to reached the target grid (this actually can ignore if there is a response mechanism and handle it bettter in the agent's update_loc/update_state methods)
        # elif # some condition:
        #     raise ValueError("Agent is not at the correct location")
        
        if self.heatmap_model_needed:
            # Check the grid size and the number of canvas to match with the heatmap model output
            heatmap_model_version = kwargs.get('model_version', 'v1')
            if HEATMAP_MODEL_CONFIG['model_types'][heatmap_model_version]['output_dim'] != [self.num_canvas, self.num_canvas]:
                raise ValueError("The heatmap model output dimension doesn't match with the search area")

            self.rssi_history.append((current_row, current_col, rssi_normalize(rssi)))
            action_kwargs['heatmap'] = self.model_service.generate_heatmap(
                rssi_list=self.rssi_history,
                model_version=heatmap_model_version,
                guide_weight=kwargs.get('guide_weight', 2.0)
            )
                  
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

    def get_available_agents(self):
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

    for rssi in [-120, -115, -110, -105, -100, -95, -90]:
        new_loc = search_service.get_next_target(
            current_loc=prev_loc,
            rssi=rssi
        )
        print(new_loc)
        prev_loc = new_loc
    
    print(search_service.agent.location)
    print(search_service.get_available_agents())

    