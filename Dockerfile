FROM python:3.9-slim

# Install system dependencies needed by h5py and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    python3-dev \
    pkg-config \
    && apt-get clean

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

EXPOSE 80
EXPOSE 8080
EXPOSE 5000
# Run the application
CMD ["python", "application.py"]
