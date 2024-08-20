# Use a base image with CUDA and Python
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Define the command to run your application
CMD ["python", "main.py"]