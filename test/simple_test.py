import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import aiohttp
import socketio

from api.services.config import HEATMAP_MODEL_CONFIG, SEARCH_CONFIG

"""
Simple manual test for the search API and the websocket connection
"""

BASE_URL = 'http://localhost:5000/'
API_URL = BASE_URL + 'api'

@pytest.fixture(scope="function")
async def client_provider():
    """Async fixture to create and connect a Socket.IO client"""
    sio = socketio.AsyncSimpleClient()
    
    # Connect to the server
    try:
        await sio.connect(BASE_URL)
        connection_event = await sio.receive(timeout=1)
        assert connection_event[0] == 'connection_established'
    except Exception as e:
        pytest.fail(f"Connection failed during setup: {str(e)}")
    
    yield sio  # Provide client to test
    
    # Cleanup: disconnect after test completes
    await sio.disconnect()

@pytest.mark.asyncio
async def test_list_models():
    """Test REST API endpoint for listing models"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{API_URL}/models') as response:
            assert response.status == 200
            data = await response.json()
            assert 'models' in data
            assert isinstance(data['models'], list)
            assert data['models'] == list(HEATMAP_MODEL_CONFIG["model_types"].keys())

@pytest.mark.asyncio
async def test_list_search_methods():
    """Test REST API endpoint for listing search methods"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{API_URL}/search-methods') as response:
            assert response.status == 200
            data = await response.json()
            assert 'methods' in data
            assert isinstance(data['methods'], list)
            assert data['methods'] == list(SEARCH_CONFIG['agent_types'].keys())


@pytest.mark.asyncio
async def test_websocket_connection(client_provider):
    """Test basic connection using fixture"""
    async for client in client_provider:
        assert client.sid is not None
        assert client.connected

        try:
            # List models
            await client.emit('list_models')
            event = await client.receive(timeout=1)

            assert event[0] == 'available_models'
            assert event[1]['models'] == list(HEATMAP_MODEL_CONFIG["model_types"].keys())

            # List search methods
            await client.emit('list_search_methods')
            event = await client.receive(timeout=1)

            assert event[0] == 'available_methods'
            assert event[1]['methods'] == list(SEARCH_CONFIG['agent_types'].keys())
            
        except socketio.exceptions.TimeoutError:
            pytest.fail("List models or search methods timed out")
        except Exception as e:
            pytest.fail(f"Error received: {e}")
        

@pytest.mark.asyncio
async def test_search(client_provider):
    """Test simple search"""
    grid_size = 250
    num_canvas = 56
    agent = 'heatmap_greedy'
    current_loc = [22.541, 114.058]
    rssi = -110.0

    async for client in client_provider:
        try:
            await client.emit('init_search', {
                'agent': agent,
                'grid_size': grid_size,
                'num_canvas': num_canvas,
                'current_loc': current_loc
            })

            event = await client.receive(timeout=1)
            
            assert event[0] == 'search_initialized' # Name of the event
            assert event[1]['message'] == f'Search initialized with agent: {agent}'
            assert event[1]['agent'] == agent
            assert event[1]['grid_size'] == grid_size
            assert event[1]['num_canvas'] == num_canvas

        except socketio.exceptions.TimeoutError:
            pytest.fail("Search initialization timed out")
        except Exception as e:
            pytest.fail(f"Error received: {e}")
        
        try:
            await client.emit('get_next_target', {
                'current_loc': current_loc,
                'rssi': rssi,
                'model_version': 'v1',
                'guide_weight': 2.0
            })

            event = await client.receive(timeout=10)
            assert event[0] == 'next_target'
            assert event[1]['current_loc'] == current_loc
            assert event[1]['rssi'] == rssi
            assert event[1]['next_target'] is not None
            assert len(event[1]['next_target']) == 2
            assert isinstance(event[1]['next_target'][0], float)
            assert isinstance(event[1]['next_target'][1], float)

            # Test heatmap 
            assert event[1]['heatmap'] is not None
            assert isinstance(event[1]['heatmap'], list)
            assert len(event[1]['heatmap']) == len(event[1]['heatmap'][0]) == num_canvas

        except socketio.exceptions.TimeoutError:
            pytest.fail("Next target timed out")
        except Exception as e:
            pytest.fail(f"Error received: {e}")

        
@pytest.mark.asyncio
async def test_search_without_init(client_provider):
    """Test simple search"""
    current_loc = [22.541, 114.058]
    rssi = -110.0

    async for client in client_provider:
        try:
            await client.emit('get_next_target', {
                'current_loc': current_loc,
                'rssi': rssi,
                'model_version': 'v1',
                'guide_weight': 2.0
            })

            event = await client.receive(timeout=10)
            assert event[0] == 'error'

        except socketio.exceptions.TimeoutError:
            pytest.fail("Next target timed out")
        except Exception as e:
            pytest.fail(f"Error received: {e}")
        