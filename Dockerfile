# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files into the container
COPY requirements-tensorboard.txt /app/
COPY logs/runs /app/logs/runs

# Install only the necessary packages
RUN pip install --no-cache-dir -r requirements-tensorboard.txt

# Expose the port TensorBoard will run on
EXPOSE 6006

# Run TensorBoard
CMD ["tensorboard", "--logdir=logs/runs", "--host=0.0.0.0"]