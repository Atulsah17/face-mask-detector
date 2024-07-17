FROM tensorflow/tensorflow:2.4.0

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Install Python dependencies
RUN pip install opencv-python-headless

# Set the working directory
WORKDIR /app

# Copy the script and the model weights
COPY mask_detection.py /app/mask_detection.py
COPY model_weights.h5 /app/model_weights.h5

# Copy Haar cascade file
COPY haarcascade_frontalface_default.xml /app/haarcascade_frontalface_default.xml

# Command to run the script
CMD ["python", "mask_detection.py"]
