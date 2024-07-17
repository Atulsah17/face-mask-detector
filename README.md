# Mask Detection Project

This repository contains a Python script for detecting masks on faces in input videos using TensorFlow and OpenCV. Here's how to use and run the script in a Docker environment.

## Contents

- `mask_detection.py`: Python script for mask detection on videos.
- `Face_mask`: jupyter notebook of model trainning.
- `README.md`: This file providing instructions and details about the project.
- `Dockerfile`: Docker configuration file to set up the environment.

## Instructions

### Running the Docker Container

1. Navigate to the project directory.
   
2. Build the Docker image:

3. Run the Docker container:


Replace `/path/to/Input_videos` and `/path/to/output_videos` with the actual paths on your system where input and output videos are stored.

4. The script will process the input videos, perform mask detection, and save the output videos in the specified output directory.

### Dependencies

- Python 3.6+
- TensorFlow
- OpenCV (cv2)
- Docker

### Notes

- Make sure to have sufficient disk space and permissions for input and output directories.
- Adjust the Docker volume mounts (`-v`) according to your local file paths.


