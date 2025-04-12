from flask import current_app, request
from flask_socketio import SocketIO, emit
import base64

from api.services.websocket_manager import WebSocketManager

socketio = SocketIO()
ws_manager = None

@socketio.on('connect')
def handle_connect():
    """Handle new client connection"""
    sid = request.sid
    current_app.logger.info(f"Client connected: {sid}")
    ws_manager.create_session(sid)
    emit('connection_established', {'session_id': sid})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnect"""
    sid = request.sid
    current_app.logger.info(f"Client disconnected: {sid}")
    ws_manager.remove_session(sid)

@socketio.on('init_search')
def handle_init_search(data, **kwargs):
    """Initialize search with parameters"""
    sid = request.sid
    search_service = ws_manager.get_session(sid)
    current_app.logger.info(f"Initializing search with data: {data}")
    
    try:
        agent = data.get('agent', 'heatmap_greedy')
        grid_size = data.get('grid_size')
        num_canvas = data.get('num_canvas')
        origin_loc = data.get('origin_loc')
        start_loc = data.get('start_loc')
        
        if origin_loc is None:
            current_app.logger.error("Origin location is required")
            raise ValueError("Origin location is required")
        
        if start_loc is None:
            current_app.logger.error("Start location is required")
            raise ValueError("Start location is required")
        
        if grid_size is None:
            current_app.logger.error("Grid size is required")
            raise ValueError("Grid size is required")
        grid_size = int(grid_size)
        
        if num_canvas is None:
            current_app.logger.error("Number of canvas is required")
            raise ValueError("Number of canvas is required")
        num_canvas = int(num_canvas)

        search_service.set_agent(agent)
        search_service.init_params(
            **data
        )
        
        emit('search_initialized', {
            'message': f'Search initialized with agent: {agent}',
            'agent': agent,
            'grid_size': grid_size,
            'num_canvas': num_canvas,
            'demo_data': search_service.demo_data
        })
    except Exception as e:
        current_app.logger.error(f"Error initializing search: {e}")
        emit('error', {'error': str(e)})

@socketio.on('get_next_target')
def handle_next_target(data, **kwargs):
    """Get next target location"""
    sid = request.sid
    search_service = ws_manager.get_session(sid)
    current_app.logger.info(f"Getting next target with data: {data}")
    
    try:
        current_loc = data.get('current_loc')
        rssi = data.get('rssi')
        if current_loc is None:
            raise ValueError("Current location is required")
        if rssi is None:
            raise ValueError("RSSI is required")
        
        
        next_target = search_service.get_next_target(
            **data
        )
        
        response = {
            'current_loc': current_loc,
            'rssi': rssi,
            'next_target': next_target,
            'demo_data': search_service.demo_data
        }
        
        # Handle heatmap data
        heatmap = search_service.get_last_heatmap()
        if heatmap is not None:
            response['heatmap'] = heatmap.tolist()
            
            # Optionally generate and send heatmap image
            try:
                img_buf = search_service.get_last_heatmap_image()
                img_buf.seek(0)
                img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')
                response['heatmap_image'] = img_b64
            except Exception as e:
                current_app.logger.error(f"Error generating heatmap image: {e}")
        
        emit('next_target', response)
    except Exception as e:
        current_app.logger.error(f"Error getting next target: {e}")
        emit('error', {'error': str(e)})

@socketio.on('list_models')
def handle_list_models(*args, **kwargs):
    """List available models"""
    try:
        models = ws_manager.model_service.get_available_models()
        emit('available_models', {'models': models})
    except Exception as e:
        current_app.logger.error(f"Error listing models: {e}")  
        emit('error', {'error': str(e)})

@socketio.on('list_search_methods')
def handle_list_search_methods(*args, **kwargs):
    """List available search methods"""
    sid = request.sid
    search_service = ws_manager.get_session(sid)
    try:
        methods = search_service.get_available_agents()
        emit('available_methods', {'methods': methods})
    except Exception as e:
        current_app.logger.error(f"Error listing search methods: {e}")
        emit('error', {'error': str(e)})

def init_socketio(app):
    """Initialize SocketIO with the Flask app"""
    global ws_manager
    socketio.init_app(app, cors_allowed_origins="*")
    ws_manager = WebSocketManager(socketio)
    return socketio 