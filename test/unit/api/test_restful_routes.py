import pytest
from flask.testing import FlaskClient
from unittest.mock import patch

@pytest.mark.asyncio
async def test_rest_list_models(app_client: FlaskClient, model_config: dict):
    """Test REST list models"""
    response = app_client.get('/api/models')
    assert response.status_code == 200
    assert response.json['models'] == list(model_config["model_types"].keys())


@pytest.mark.asyncio
async def test_rest_list_search_methods(app_client: FlaskClient, search_config: dict):
    """Test REST list search methods"""
    response = app_client.get('/api/search-methods')
    assert response.status_code == 200
    assert response.json['methods'] == list(search_config['agent_types'].keys())


@pytest.mark.asyncio
async def test_rest_list_models_error(app_client: FlaskClient):
    """Test REST list models error handling"""
    with patch('api.services.model_service.ModelService.get_available_models', side_effect=Exception("Test error")):
        response = app_client.get('/api/models')
        assert response.status_code == 500
        assert response.json['error'] == "Test error"


@pytest.mark.asyncio
async def test_rest_list_search_methods_error(app_client: FlaskClient):
    """Test REST list search methods error handling"""
    with patch('api.services.search_service.SearchService.get_available_agents', side_effect=Exception("Test error")):
        response = app_client.get('/api/search-methods')
        assert response.status_code == 500
        assert response.json['error'] == "Test error"
