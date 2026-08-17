# Smart Helmet AI Ignition System Docker Container
FROM python:3.10-slim

# Install system dependencies for OpenCV and OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port for Live Server
EXPOSE 5050

# Run live server
CMD ["python", "src/raspberry_pi/live_server.py", "--port", "5050"]
