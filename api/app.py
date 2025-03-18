import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import eventlet
eventlet.monkey_patch()

from flask import Flask

from api.routes.websocket_routes import init_socketio
from api.routes.routes import search_blueprint  

# Add the project root to sys.path

def create_app():
    app = Flask(__name__)
    # Register REST blueprints if needed
    app.register_blueprint(search_blueprint)
    
    # Initialize and get the socketio instance
    socketio = init_socketio(app)
    
    return app, socketio

if __name__ == '__main__':
    app, socketio = create_app()
    # Use socketio.run instead of app.run
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 