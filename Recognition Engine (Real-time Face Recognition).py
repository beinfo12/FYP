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

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",  # Use your MySQL password
    database="face_recognition_system"
)
cursor = db.cursor()

# Function to detect faces in a frame
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces

# Function to recognize the face
def recognize_face(embedding):
    cursor.execute("SELECT id, name, embedding FROM faces")
    faces = cursor.fetchall()
    
    best_match_name = "Unknown"
    best_match_distance = float('inf')
    
    for face in faces:
        db_embedding = pickle.loads(face[2])  # Deserialize the embedding
        distance = np.linalg.norm(embedding - db_embedding)
        
        if distance < best_match_distance:
            best_match_distance = distance
            best_match_name = face[1]
    
    threshold = 0.6
    if best_match_distance < threshold:
        return best_match_name
    else:
        return "Unknown"

# Function for face recognition from webcam
def start_recognition():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        faces = detect_faces(frame)
        
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_image = frame[y:y+h, x:x+w]
            
            # Resize, preprocess and generate embedding
            face_image_resized = cv2.resize(face_image, (224, 224))
            face_image_resized = face_image_resized.astype('float32') / 255.0
            face_image_resized = np.expand_dims(face_image_resized, axis=0)
            face_image_resized = preprocess_input(face_image_resized, version=2)
            
            embedding = facenet_model.predict(face_image_resized).flatten()
            
            # Recognize the face
            recognized_name = recognize_face(embedding)
            
            # Display the name on the frame
            cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        
        # Press 'q' to quit the recognition loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Start the face recognition process
start_recognition()
