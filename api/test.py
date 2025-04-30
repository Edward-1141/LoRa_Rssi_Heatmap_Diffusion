import math
import os
import sys
import dotenv
import json
dotenv.load_dotenv()
sys.path.append(os.getenv('DRONE_SEARCH_ROOT_DIR', '.'))

import pytest
from flask import Flask
from tqdm import tqdm

from api.routes.websocket_routes import init_socketio
from api.routes.routes import search_blueprint
from api.services.config import HEATMAP_MODEL_CONFIG, SEARCH_CONFIG

def test_and_record_search_results_with_testing_env(test_client, 
                                                    save_path='./output/test_results.json', 
                                                    env_path='./data/gw_1.json', 
                                                    search_steps=2):
    def env_key(row, col):
        return f'{row}_{col}'

    # Load the environment
    with open(env_path, 'r') as f:
        env = json.load(f)
    print(f'loaded env from {env_path}')
    
    record_data = {
        'data': [],
        'env': env,
        'distance_to_origin': []
    }
    
    # connect to the test client
    test_client.emit('connect')
    test_client.get_received().pop() # ignore the connect event
    
    # Init the search with the testing environment
    init_object = env['init_object']
    test_client.emit('init_search', {
        'agent': init_object['agent'],
        'grid_size': init_object['grid_size'],
        'num_canvas': init_object['num_canvas'],
        'origin_loc': init_object['origin_loc'],
        'start_loc': init_object['start_loc']
    })
    event = test_client.get_received().pop()
    demo_data = event['args'][0]['demo_data']

    # Set the current location and RSSI for search
    cur_loc = init_object['start_loc']
    cur_loc_idx = demo_data['current_loc_idx']
    cur_rssi = env[env_key(cur_loc_idx[0], cur_loc_idx[1])]['rssi']

    # Start the search
    for _ in tqdm(range(search_steps)):
        test_client.emit('get_next_target', {
            'current_loc': cur_loc,
            'rssi': cur_rssi,
            'model_version': 'v2'
        })
        event = test_client.get_received().pop()
        response = event['args'][0]
        demo_data = response['demo_data']

        # Record the whole response object
        record_data['data'].append(response)

        # Update the current location and RSSI for next step
        cur_loc = response['next_target']
        cur_loc_idx = response['demo_data']['next_loc_idx']
        cur_rssi = env[env_key(cur_loc_idx[0], cur_loc_idx[1])]['rssi']

        # calculate the distance to the origin
        distance_to_origin = math.sqrt((cur_loc[0] - init_object['origin_loc'][0])**2 + (cur_loc[1] - init_object['origin_loc'][1])**2)
        record_data['distance_to_origin'].append(distance_to_origin)

    # Save the record data
    with open(save_path, 'w') as f:
        json.dump(record_data, f, indent=4)

def main():
    # Add the project root to sys.path TODO: remove by setting environment variable
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    
    def create_test_client():
        app = Flask(__name__)
        # Register REST blueprints if needed
        app.register_blueprint(search_blueprint)
        socketio = init_socketio(app)
        test_client = socketio.test_client(app)
        return test_client
    
    test_client = create_test_client()
    test_and_record_search_results_with_testing_env(test_client, 
                                                    save_path='./data/cached_gw_1.json', 
                                                    env_path='./data/gw_1.json', 
                                                    search_steps=200)

if __name__ == '__main__':
    main()

