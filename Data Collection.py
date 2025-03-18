import cv2
import numpy as np
import dlib
import mysql.connector
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import pickle

# Load VGGFace model
facenet_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Dlib face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("path/to/shape_predictor_68_face_landmarks.dat")  # Provide correct path

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",  # Use your MySQL password
    database="face_recognition_system"
)
cursor = db.cursor()

# Function to capture face image and generate embedding
def capture_and_store_face(name):
    cap = cv2.VideoCapture(0)  # Start video capture from webcam
    print("Starting face capture...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_image = frame[y:y+h, x:x+w]
            
            # Resize and preprocess the face image
            face_image_resized = cv2.resize(face_image, (224, 224))
            face_image_resized = face_image_resized.astype('float32') / 255.0
            face_image_resized = np.expand_dims(face_image_resized, axis=0)
            face_image_resized = preprocess_input(face_image_resized, version=2)
            
            # Generate embedding using VGGFace model
            embedding = facenet_model.predict(face_image_resized).flatten()

            # Store the embedding and name in the database
            embedding_blob = pickle.dumps(embedding)  # Convert embedding to binary

            # Insert the face data into the database
            cursor.execute("INSERT INTO faces (name, embedding) VALUES (%s, %s)", (name, embedding_blob))
            db.commit()
            print(f"Stored {name}'s face embedding in the database.")
        
        # Show the frame with detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Face Capture", frame)
        
        # If 'q' is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example: Capture face for "John Doe"
name = input("Enter name for the face: ")
capture_and_store_face(name)
