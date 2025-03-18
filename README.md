# LoRa Conditional Diffusion

## Installation

To set up the environment, use the enviroments.yml file provided in the repository. This file contains all the necessary dependencies.

```sh
conda env create -f enviroments.yml
conda activate diffusion
```

## Project Structure

### Directories

- **api**: Contains the Flask API code, including services and routes.
- **model**: Diffusion model code.
- **search**: Search agent code.
- **script**: Contains testing scripts and notebooks.

## Starting the Flask Server

To start the Flask server for development, run the following command:

```sh
python api/app.py
```

## API Documentation

The system provides two communication interfaces: REST API and WebSocket. WebSocket is recommended for stateful search operations, especially when multiple concurrent searches are needed.

### REST API

#### List Available Models

**Endpoint**: `/api/models`

**Method**: `GET`

**Description**: Lists all available models.

#### List Available Search Methods

**Endpoint**: `/api/search-methods`

**Method**: `GET`

**Description**: Lists all available search methods.


### WebSocket API

The WebSocket API provides real-time, bidirectional communication with built-in session management. Each client connection gets its own dedicated search service instance, allowing multiple concurrent searches without conflicts.

#### Connection

Connect to the server's WebSocket endpoint:
```javascript
// Using socket.io client
const socket = io('http://localhost:5000');

socket.on('connect', () => {
  console.log('Connected, session ID:', socket.id);
});

socket.on('connection_established', (data) => {
  console.log('Server acknowledged connection:', data.session_id);
});
```

#### Events

##### List Models

**Event**: `list_models`

**Description**: Lists all available models.

**Example**:
```javascript
socket.emit('list_models');

socket.on('available_models', (data) => {
  console.log('Available models:', data.models);
});
```

##### List Search Methods

**Event**: `list_search_methods`

**Description**: Lists all available search methods.

**Example**:
```javascript
socket.emit('list_search_methods');

socket.on('available_methods', (data) => {
  console.log('Available search methods:', data.methods);
});
```

##### Initialize Search

**Event**: `init_search`

**Description**: Initializes a new search with the specified parameters.

**Parameters**:
```javascript
{
  "agent": "heatmap_greedy",  // Search agent/method to use
  "current_loc": [22.084, 37.422],  // Starting location [longitude, latitude]
  "grid_size": 250,  // Size of each grid cell in meters
  "num_canvas": 56  // Number of grid cells in each direction
}
```

**Response Event**: `search_initialized`

**Example**:
```javascript
socket.emit('init_search', {
  agent: 'heatmap_greedy',
  current_loc: [22.084, 37.422],
  grid_size: 250,
  num_canvas: 56
});

socket.on('search_initialized', (data) => {
  console.log('Search initialized:', data);
  // data contains: message, agent, grid_size, num_canvas
});
```

##### Get Next Target

**Event**: `get_next_target`

**Description**: Gets the next target location based on the current RSSI value.

**Parameters**:
```javascript
{
  "current_loc": [22.084, 37.422],  // Current location [longitude, latitude]
  "rssi": -110,  // Current RSSI reading
  "model_version": "v1",  // Optional: Heatmap model version
  "guide_weight": 2.0  // Optional: Guide weight for heatmap generation
}
```

**Response Event**: `next_target`

**Example**:
```javascript
socket.emit('get_next_target', {
  current_loc: [22.084, 37.422],
  rssi: -110,
  model_version: 'v1',
  guide_weight: 2.0
});

socket.on('next_target', (data) => {
  console.log('Next target location:', data.next_target);
  // data also contains: current_loc, rssi, heatmap (optional), heatmap_image (optional)
  
  if (data.heatmap_image) {
    // Display base64-encoded heatmap image
    const img = document.createElement('img');
    img.src = 'data:image/png;base64,' + data.heatmap_image;
    document.getElementById('heatmap-container').appendChild(img);
  }
});
```

##### Error Handling

All events can return an error response:

```javascript
socket.on('error', (data) => {
  console.error('Error:', data.error);
});
```

## Testing

The project includes test files for both the REST API and WebSocket interfaces:

```sh
# Run tests
pytest test/simple_test.py -v
```
