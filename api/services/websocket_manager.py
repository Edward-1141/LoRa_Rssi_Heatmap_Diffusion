from flask_socketio import SocketIO
from api.services.model_service import ModelService
from api.services.search_service import SearchService

class WebSocketManager:
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.model_service = ModelService()  # Shared model service
        self.active_sessions = {}  # Map of session_id -> SearchService instances
    
    def create_session(self, sid):
        """Create a new search session for a client connection"""
        search_service = SearchService(self.model_service)
        self.active_sessions[sid] = search_service
        return search_service
    
    def get_session(self, sid):
        """Get the search service for a specific session"""
        if sid not in self.active_sessions:
            return self.create_session(sid)
        return self.active_sessions[sid]
    
    def remove_session(self, sid):
        """Remove a session when client disconnects"""
        if sid in self.active_sessions:
            del self.active_sessions[sid] 