#!/bin/bash

# Step 1: Stop the running container
echo "Stopping the running container..."
sudo docker compose down

# Step 2: Build the Docker image
echo "Building the Docker image..."
sudo docker compose build

# Step 3: Start the container in detached mode
echo "Starting the container in detached mode..."
sudo docker compose up -d

# Step 4: Get the container ID of the newly started container
CONTAINER_ID=$(sudo docker ps -qf "ancestor=radar_open_space_segmentation-ross")
echo "Container ID: $CONTAINER_ID"

# Step 5: Open an interactive shell in the running container
if [ -n "$CONTAINER_ID" ]; then
    echo "Opening an interactive shell in the container..."

    sudo docker exec -it $CONTAINER_ID /bin/sh
else
    echo "No running container found. Please check if the container started correctly."
fi
