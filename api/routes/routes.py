from flask import Blueprint, request, jsonify, send_file, current_app
import os
import io
import base64

from api.services.model_service import ModelService
from api.services.search_service import SearchService

search_blueprint = Blueprint('search', __name__, url_prefix='/api')
model_service = ModelService()
search_service = SearchService(model_service)

@search_blueprint.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    try:
        models = model_service.get_available_models()
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@search_blueprint.route('/search-methods', methods=['GET'])
def list_search_methods():
    """List available search methods"""
    try:
        methods = search_service.get_available_agents()
        return jsonify({"methods": methods})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@search_blueprint.route('/init-searce-service', methods=['POST'])
def init_search_service():
    """Initialize the search service with the specified agent"""
    data = request.get_json()
    agent = data.get('agent', 'heatmap_greedy')
    current_loc = data.get('current_loc')
    grid_size = int(data.get('grid_size', 250))
    num_canvas = int(data.get('num_canvas', 56))

    if not current_loc:
        return {"error": "Missing required parameters"}, 400

    try:
        search_service.set_agent(agent)
        search_service.init_params(grid_size=grid_size, num_canvas=num_canvas, current_loc=current_loc)
        return jsonify({
            "message": f"Search service initialized with agent: {agent}",
            "agent": agent,
            "grid_size": grid_size,
            "num_canvas": num_canvas
        }) 
    except Exception as e:
        return {"error": str(e)}, 500

@search_blueprint.route('/next-target', methods=['POST'])
def get_next_target():
    """Get the next target location based on the current RSSI value"""
    data = request.get_json()
    current_loc = data.get('current_loc')
    rssi = data.get('rssi')
    if not current_loc or not rssi:
        return {"error": "Missing required parameters"}, 400
    try:
        response = {
            "current_loc": current_loc,
            "rssi": rssi,
            "next_target": search_service.get_next_target(current_loc=current_loc, rssi=rssi)
        }
        heatmap = search_service.get_last_heatmap()
        if heatmap is not None:
            heatmap = heatmap.tolist()
            response['heatmap'] = heatmap
        else:
            response['heatmap'] = None
        return jsonify(response)
    except Exception as e:
        return {"error": str(e)}, 500

@search_blueprint.route('/heatmap_image', methods=['GET'])
def get_heatmap_image():
    """Generate and return the last heatmap image."""
    try:
        img_buf = search_service.get_last_heatmap_image()
        return send_file(img_buf, mimetype='image/png', as_attachment=True, download_name='heatmap.png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500