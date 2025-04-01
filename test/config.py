import os

# Default configuration
DEFAULT_CONFIG = {
    'BASE_URL': 'http://localhost:5000/',
    'API_URL': 'http://localhost:5000/api'
}

# Load from environment variable if available
BASE_URL = os.getenv('TEST_BASE_URL', DEFAULT_CONFIG['BASE_URL'])
API_URL = os.getenv('TEST_API_URL', DEFAULT_CONFIG['API_URL']) 