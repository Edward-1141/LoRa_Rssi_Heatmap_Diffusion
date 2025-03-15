from flask import Flask
from flask_cors import CORS
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.routes.routes import search_blueprint

def create_app():
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(search_blueprint)
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)