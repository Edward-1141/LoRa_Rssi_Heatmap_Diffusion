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

## Basic API Structure

### List Available Models

**Endpoint**: `/api/models`

**Method**: `GET`

**Description**: Lists all available models.

### List Available Search Methods

**Endpoint**: `/api/search-methods`

**Method**: `GET`

**Description**: Lists all available search methods.

### Initialize Search Service

**Endpoint**: `/api/init-searce-service`

**Method**: `POST`

**Description**: Initializes the search service with the specified agent.

**Request Body**:
```json
{
  "agent": "heatmap_greedy",
  "current_loc": [22.084, 37.422],
  "grid_size": 250,
  "num_canvas": 56
}
```

### Get Next Target

**Endpoint**: `/api/next-target`

**Method**: `POST`

**Description**: Gets the next target location based on the current RSSI value.

**Request Body**:
```json
{
  "current_loc": [22.084, 37.422],
  "rssi": -110,
  "model_version": "v1", (optional)
  "search_method": "heatmap_greedy" (optional)
}
```

### Get Heatmap Image

**Endpoint**: `/api/heatmap_image`

**Method**: `GET`

**Description**: Generates and returns the last heatmap image.
