import base64
import json
import os
import pytest
from flask_socketio import SocketIOTestClient

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
async def test_search(socketio_client: SocketIOTestClient, output_dir: str):
    """Test client search"""
    grid_size = 150
    num_canvas = 56
    agent = 'heatmap_greedy'
    origin_loc = [22.541, 114.058]
    start_loc = [22.521, 114.04]
    rssi = -110.0
    model_version = 'v2'
    guide_weight = 2.0

    socketio_client.emit('init_search', {
        'agent': agent,
        'grid_size': grid_size,
        'num_canvas': num_canvas,
        'origin_loc': origin_loc,
        'start_loc': start_loc
    })

    event = socketio_client.get_received().pop()
    assert event['name'] == 'search_initialized'
    assert event['args'][0]['agent'] == agent
    assert event['args'][0]['grid_size'] == grid_size
    assert event['args'][0]['num_canvas'] == num_canvas

    socketio_client.emit('get_next_target', {
        'current_loc': start_loc,
        'rssi': rssi,
        'model_version': model_version,
        'guide_weight': guide_weight
    })

    event = socketio_client.get_received().pop()
    assert event['name'] == 'next_target'
    assert event['args'][0]['current_loc'] == start_loc
    assert event['args'][0]['rssi'] == rssi
    assert event['args'][0]['next_target'] is not None
    assert len(event['args'][0]['next_target']) == 2
    assert isinstance(event['args'][0]['next_target'][0], float)
    assert isinstance(event['args'][0]['next_target'][1], float)

    # Test heatmap
    assert event['args'][0]['heatmap'] is not None
    assert isinstance(event['args'][0]['heatmap'], list)
    assert len(event['args'][0]['heatmap']) == len(event['args'][0]['heatmap'][0]) == num_canvas

    # Test heatmap image
    heatmap_image = event['args'][0]['heatmap_image']
    assert heatmap_image is not None
    assert isinstance(heatmap_image, str)
    with open(os.path.join(output_dir, 'test-heatmap.png'), 'wb') as f:
        f.write(base64.b64decode(heatmap_image))
    
    # Test demo data
    assert event['args'][0]['demo_data'] is not None
    assert isinstance(event['args'][0]['demo_data'], dict)
    assert 'current_loc_idx' in event['args'][0]['demo_data']
    assert 'next_loc_idx' in event['args'][0]['demo_data']
    assert 'action' in event['args'][0]['demo_data']

    # Test whole response
    with open(os.path.join(output_dir, 'test-response.json'), 'w') as f:
        json.dump(event['args'][0], f)

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

 
