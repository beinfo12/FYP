import cv2
import numpy as np
import dlib
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image
import mysql.connector

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Desktop/FYP/facial-landmarks-recognition-master/shape_predictor_68_face_landmarks.dat')

# Load VGGFace model
model = VGGFace(model='resnet50')

# Connect to MySQL Database
def connect_db():
    connection = mysql.connector.connect(
        host="localhost",
        user="your_username",
        password="your_password",
        database="FaceRecognitionDB"
    )
    return connection

# Get the face embedding from the face image
def get_face_embedding(face_image):
    img = cv2.resize(face_image, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    embedding = model.predict(img)
    return embedding

# Function to recognize face from the database
def recognize_face(face_embedding):
    connection = connect_db()
    cursor = connection.cursor()

    query = "SELECT * FROM Faces"
    cursor.execute(query)
    result = cursor.fetchall()

    # Compare the incoming face embedding with stored embeddings
    for row in result:
        stored_embedding = np.frombuffer(row[2], dtype=np.float32)
        distance = np.linalg.norm(stored_embedding - face_embedding)

        if distance < 0.6:  # Threshold for face recognition
            print(f"Recognized {row[1]} with distance {distance}")
            return row[1]  # Return name of the recognized face

    return "Unknown"  # If no match is found

# Start the webcam feed for live recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_roi = frame[y:y+h, x:x+w]

        # Get the face embedding
        embedding = get_face_embedding(face_roi)

        # Recognize the face by comparing with database embeddings
        recognized_name = recognize_face(embedding.flatten())
        
        # Draw bounding box and label the recognized person
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the live frame with recognized faces
    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
