import os
import cv2
import tensorflow as tf
import numpy as np

# Ensure the save directory exists
save_dir = '/app/'
weights_path = os.path.join(save_dir, 'model_weights.h5')

# Define the model architecture
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

# Build and compile the model
model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the weights
model.load_weights(weights_path)

# Load the Haar cascade for face detection
haar_cascade_path = '/app/haarcascade_frontalface_default.xml'

if not os.path.exists(haar_cascade_path):
    raise FileNotFoundError(f"Haar cascade file not found at {haar_cascade_path}")

face_cascade = cv2.CascadeClassifier(haar_cascade_path)

if face_cascade.empty():
    raise IOError("Failed to load Haar cascade file. Please check the path and the file's existence.")

# Function to preprocess frame for model input
def preprocess_frame(frame):
    face = cv2.resize(frame, (224, 224))
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    face = face / 255.0  # Normalize to [0, 1]
    return face

# Function to perform inference on a video file
def process_video(input_path, output_path):
    print(f"Processing video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    if not out.isOpened():
        print(f"Error: Could not write to output video {output_path}")
        return

    frame_count = 0
    face_detected_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        face_detected_count += len(faces)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            preprocessed_face = preprocess_frame(face)
            prediction = model.predict(preprocessed_face)
            label = 'Mask' if np.argmax(prediction) == 1 else 'No Mask'

            color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Finished processing video: {input_path}")
    print(f"Total frames: {frame_count}, Faces detected: {face_detected_count}")
    print(f"Output video saved to: {output_path}")

# Perform inference on sample videos
input_videos_dir = '/app/input_videos'
output_videos_dir = '/app/output_videos'

videos = [
    ('Test_video1.mp4', 'output_video1.mp4'),
    ('Test_video2.mp4', 'output_video2.mp4'),
    ('Test_video3.mp4', 'output_video3.mp4')
]

for input_video, output_video in videos:
    input_path = os.path.join(input_videos_dir, input_video)
    output_path = os.path.join(output_videos_dir, output_video)
    
    if os.path.exists(input_path):
        process_video(input_path, output_path)
    else:
        print(f"Error: Input video {input_path} does not exist")

print("Inference completed.")
