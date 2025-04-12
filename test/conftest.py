import os
import sys
import dotenv

dotenv.load_dotenv()
sys.path.append(os.getenv('DRONE_SEARCH_ROOT_DIR'))

import pytest
from flask import Flask

from api.routes.websocket_routes import init_socketio
from api.routes.routes import search_blueprint
from api.services.config import HEATMAP_MODEL_CONFIG, SEARCH_CONFIG

@pytest.fixture(scope="module")
def app():
    """Create a Flask app for testing"""
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(search_blueprint)
    return app

@pytest.fixture(scope="module")
def socketio(app):
    """Create a SocketIO instance for testing"""
    return init_socketio(app)

@pytest.fixture(scope="module")
def app_client(app):
    """Create a test client for the app"""
    return app.test_client()

@pytest.fixture(scope="function")
def socketio_client(app, socketio):
    """Create a Socket.IO client for testing"""
    test_client = socketio.test_client(app)
    test_client.emit('connect')
    connection_event_found = False
    for event in test_client.get_received():
        if event['name'] == 'connection_established':
            connection_event_found = True
            assert event['args'][0]['session_id'] is not None
            break
    assert connection_event_found
    yield test_client
    test_client.disconnect()

@pytest.fixture(scope="session")
def model_config():
    return HEATMAP_MODEL_CONFIG

@pytest.fixture(scope="session")
def search_config():
    return SEARCH_CONFIG

@pytest.fixture(scope="session")
def output_dir():
    return os.path.join(os.getenv('DRONE_SEARCH_ROOT_DIR'), 'output')

