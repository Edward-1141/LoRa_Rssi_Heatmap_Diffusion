import base64
import json
import os
from unittest.mock import patch
import pytest
from flask_socketio import SocketIOTestClient

def assert_valid_init_search(event, agent, grid_size, num_canvas):
    assert event['name'] == 'search_initialized'
    assert event['args'][0]['agent'] == agent
    assert event['args'][0]['grid_size'] == grid_size
    assert event['args'][0]['num_canvas'] == num_canvas
    assert event['args'][0]['demo_data'] is not None
    assert isinstance(event['args'][0]['demo_data'], dict)
    assert 'current_loc_idx' in event['args'][0]['demo_data']


def assert_valid_next_target(event, current_loc, rssi):
    assert event['name'] == 'next_target'
    assert event['args'][0]['current_loc'] == current_loc
    assert event['args'][0]['rssi'] == rssi
    assert event['args'][0]['next_target'] is not None
    assert len(event['args'][0]['next_target']) == 2
    assert isinstance(event['args'][0]['next_target'][0], float)
    assert isinstance(event['args'][0]['next_target'][1], float)

    assert event['args'][0]['demo_data'] is not None
    assert isinstance(event['args'][0]['demo_data'], dict)
    assert 'current_loc_idx' in event['args'][0]['demo_data']
    assert 'next_loc_idx' in event['args'][0]['demo_data']
    assert 'action' in event['args'][0]['demo_data']

def assert_valid_heatmap(event, agent, model_version, num_canvas, output_dir):
    assert event['args'][0]['heatmap'] is not None
    assert isinstance(event['args'][0]['heatmap'], list)
    assert len(event['args'][0]['heatmap']) == len(event['args'][0]['heatmap'][0]) == num_canvas

    # Test heatmap image
    heatmap_image = event['args'][0]['heatmap_image']
    assert heatmap_image is not None
    assert isinstance(heatmap_image, str)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f'test-heatmap-{agent}-{model_version}.png'), 'wb') as f:
        f.write(base64.b64decode(heatmap_image))
    

@pytest.mark.asyncio
async def test_socketio_list_models(socketio_client: SocketIOTestClient, model_config: dict):
    """Test client list models"""
    socketio_client.emit('list_models')
    list_models_event_found = False
    for event in socketio_client.get_received():
        if event['name'] == 'available_models':
            list_models_event_found = True
            assert event['args'][0]['models'] == list(model_config["model_types"].keys())
    assert list_models_event_found

@pytest.mark.asyncio
async def test_socketio_list_models_error(socketio_client: SocketIOTestClient):
    """Test client list models error"""
    with patch('api.services.model_service.ModelService.get_available_models', side_effect=Exception("Test error")):
        socketio_client.emit('list_models')
        event = socketio_client.get_received().pop()
        assert event['name'] == 'error'
        assert event['args'][0]['error'] == 'Test error'

@pytest.mark.asyncio
async def test_socketio_list_search_methods_error(socketio_client: SocketIOTestClient):
    """Test client list search methods error"""
    with patch('api.services.search_service.SearchService.get_available_agents', side_effect=Exception("Test error")):
        socketio_client.emit('list_search_methods')
        event = socketio_client.get_received().pop()
        assert event['name'] == 'error'
        assert event['args'][0]['error'] == 'Test error'

@pytest.mark.asyncio
async def test_socketio_list_search_methods(socketio_client: SocketIOTestClient, search_config: dict):
    """Test client list search methods"""
    socketio_client.emit('list_search_methods')
    list_search_methods_event_found = False
    for event in socketio_client.get_received():
        if event['name'] == 'available_methods':
            list_search_methods_event_found = True
            assert event['args'][0]['methods'] == list(search_config['agent_types'].keys())
    assert list_search_methods_event_found

@pytest.mark.asyncio
async def test_websocket_init_search_missing_params(socketio_client: SocketIOTestClient):
    """Test websocket init search with missing parameters"""
    agent = 'heatmap_greedy'
    grid_size = 250
    num_canvas = 56
    origin_loc = [22.084, 37.422]
    start_loc = [22.084, 37.422]

    # Define test cases with missing parameters
    test_cases = [
        {
            'params': {
                'agent': agent,
                'grid_size': grid_size,
                'num_canvas': num_canvas,
                'start_loc': start_loc
            },
            'expected_error': 'Origin location is required'
        },
        {
            'params': {
                'agent': agent,
                'grid_size': grid_size,
                'num_canvas': num_canvas,
                'origin_loc': origin_loc
            },
            'expected_error': 'Start location is required'
        },
        {
            'params': {
                'agent': agent,
                'grid_size': grid_size,
                'origin_loc': origin_loc,
                'start_loc': start_loc
            },
            'expected_error': 'Number of canvas is required'
        },
        {
            'params': {
                'agent': agent,
                'num_canvas': num_canvas,
                'origin_loc': origin_loc,
                'start_loc': start_loc
            },
            'expected_error': 'Grid size is required'
        }
    ]

    # Run each test case
    for test_case in test_cases:
        socketio_client.emit('init_search', test_case['params'])
        event = socketio_client.get_received().pop()
        assert event['name'] == 'error'
        assert event['args'][0]['error'] == test_case['expected_error']

@pytest.mark.asyncio
async def test_search(socketio_client: SocketIOTestClient, output_dir: str):
    """Test client search"""
    grid_size = 150
    num_canvas = 56
    origin_loc = [22.541, 114.058]
    start_loc = [22.521, 114.04]
    rssi = -110.0
    guide_weight = 2.0
    steps = 2

    configs = [
        {
            'agent': 'heatmap_greedy',
            'model_version': 'v1',
            'steps': 2
        },
        {
            'agent': 'heatmap_greedy',
            'model_version': 'v2',
            'steps': 2
        },
        {
            'agent': 'greedy',
            'steps': 10
        },
        {
            'agent': 'greedy',
            'steps': 10
        }
    ]

    for config in configs:
        agent = config['agent']
        model_version = config['model_version'] if 'model_version' in config else 'v2'
        steps = config['steps']

        # Initialize the search
        socketio_client.emit('init_search', {
            'agent': agent,
            'grid_size': grid_size,
            'num_canvas': num_canvas,
            'origin_loc': origin_loc,
            'start_loc': start_loc
        })
        event = socketio_client.get_received().pop()
        assert_valid_init_search(event, agent, grid_size, num_canvas)

        # Getting next target for multiple steps
        for step in range(steps):
            current_loc = event['args'][0]['next_target'] if step > 0 else start_loc

            rssi_target = rssi + step if step > steps // 2 else rssi # roughly simulate the RSSI change

            socketio_client.emit('get_next_target', {
                'current_loc': current_loc,
                'rssi': rssi_target,
                'model_version': model_version,
                'guide_weight': guide_weight
            })

            event = socketio_client.get_received().pop()
            assert_valid_next_target(event, current_loc, rssi_target)
            if agent == 'heatmap_greedy':
                assert_valid_heatmap(event, agent, model_version, num_canvas, output_dir)
        
        # re-initializing the search to reset the search state
        socketio_client.emit('init_search', {
            'agent': agent,
            'grid_size': grid_size,
            'num_canvas': num_canvas,
            'origin_loc': origin_loc,
            'start_loc': start_loc
        })
        event = socketio_client.get_received().pop()
        assert_valid_init_search(event, agent, grid_size, num_canvas)

        # Search for one extra step
        socketio_client.emit('get_next_target', {
            'current_loc': start_loc,
            'rssi': rssi,
            'model_version': model_version,
            'guide_weight': guide_weight
        })
        event = socketio_client.get_received().pop()
        assert_valid_next_target(event, start_loc, rssi)
            
@pytest.mark.asyncio
async def test_search_without_init(socketio_client: SocketIOTestClient):
    """Test client search without init"""
    current_loc = [22.541, 114.058]
    rssi = -110.0
    model_version = 'v2'
    guide_weight = 2.0

    socketio_client.emit('get_next_target', {
        'current_loc': current_loc,
        'rssi': rssi,
        'model_version': model_version,
        'guide_weight': guide_weight
    })

    event = socketio_client.get_received().pop()
    assert event['name'] == 'error'
    assert event['args'][0]['error'] == 'Agent is not ready to start a new search, please set the parameters first'

@pytest.mark.asyncio
async def test_search_with_invalid_params(socketio_client: SocketIOTestClient):
    """Test client search with invalid parameters"""
    socketio_client.emit('init_search', {
        'agent': 'heatmap_greedy',
        'grid_size': 150,
        'num_canvas': 56,
        'origin_loc': [22.541, 114.058],
        'start_loc': [22.521, 114.04]
    })
    socketio_client.get_received().pop() # ignore the init search event

    socketio_client.emit('get_next_target', {
        'current_loc': [22.541, 114.058],
    })
    event = socketio_client.get_received().pop()
    assert event['name'] == 'error'
    assert event['args'][0]['error'] == 'RSSI is required'

    socketio_client.emit('get_next_target', {
        'rssi': -110.0,
    })
    event = socketio_client.get_received().pop()
    assert event['name'] == 'error'
    assert event['args'][0]['error'] == 'Current location is required'

    socketio_client.emit('get_next_target', {
        'current_loc': [22.541, 114.058],
        'rssi': -110.0,
        'model_version': 'wrong-model-version',
    })
    event = socketio_client.get_received().pop()
    assert event['name'] == 'error'
    assert event['args'][0]['error'] == 'Unsupported model version: "wrong-model-version"'

