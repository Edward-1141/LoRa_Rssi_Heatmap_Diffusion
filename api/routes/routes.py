from flask import Blueprint, request, jsonify, send_file
import os
import io
import base64

from api.services.model_service import ModelService
from api.services.search_service import SearchService

search_blueprint = Blueprint('search', __name__, url_prefix='/api')
model_service = ModelService()
search_service = SearchService(init_loc=[0, 0], model_service=model_service)

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