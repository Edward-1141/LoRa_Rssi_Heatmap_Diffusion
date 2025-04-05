import os
import sys
import argparse
import socket
import re
from typing import Union

def validate_port(port: Union[str, int]) -> int:
    """Validate port number is within valid range."""
    port_str = str(port)
    
    if not re.match(r'^\d+$', port_str):
        raise argparse.ArgumentTypeError(f"Port must be a valid integer, got {port}")
    
    port_num = int(port_str)
    if port_num < 1 or port_num > 65535:
        raise argparse.ArgumentTypeError(f"Port must be between 1 and 65535, got {port}")
    
    return port_num

def validate_host(host: str) -> str:
    """Validate host is a valid IP address (IPv4 or IPv6)."""
    try:
        # Try to parse as IPv4 first
        socket.inet_aton(host)
        return host
    except socket.error:
        # If IPv4 fails, try IPv6
        try:
            socket.inet_pton(socket.AF_INET6, host)
            return host
        except socket.error:
            raise argparse.ArgumentTypeError(f"Invalid IP address format: {host}")

def parse_args():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description='Run the Flask application with custom parameters')
    parser.add_argument('-p', '--port', 
                       type=validate_port,
                       default=5000, 
                       help='Port to run the server on (default: 5000, range: 1-65535)')
    parser.add_argument('-i', '--host', 
                       type=validate_host,
                       default='0.0.0.0', 
                       help='Host to run the server on (default: 0.0.0.0, must be valid IP address)')
    parser.add_argument('--debug', 
                       action='store_true', 
                       help='Enable debug mode')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Add the project root to sys.path TODO: remove by setting environment variable
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    
    import eventlet
    eventlet.monkey_patch()
    
    from flask import Flask
    from api.routes.websocket_routes import init_socketio
    from api.routes.routes import search_blueprint
    
    def create_app():
        app = Flask(__name__)
        # Register REST blueprints if needed
        app.register_blueprint(search_blueprint)
        socketio = init_socketio(app)
        
        return app, socketio
    
    app, socketio = create_app()
    socketio.run(app, debug=args.debug, host=args.host, port=args.port)

if __name__ == '__main__':
    main() 