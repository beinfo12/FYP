import cv2
import dlib
import numpy as np
import os
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

# Insert face embedding into database
def insert_face_embedding(name, embedding):
    connection = connect_db()
    cursor = connection.cursor()
    query = "INSERT INTO Faces (name, embedding) VALUES (%s, %s)"
    cursor.execute(query, (name, embedding.tobytes()))
    connection.commit()
    cursor.close()
    connection.close()

# Get the face embedding from the face image
def get_face_embedding(face_image):
    img = cv2.resize(face_image, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    embedding = model.predict(img)
    return embedding

# Function to collect faces from a folder and store embeddings
def collect_faces_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                # Get the face ROI
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                face_roi = img[y:y+h, x:x+w]

                # Get the face embedding
                embedding = get_face_embedding(face_roi)

                # Insert embedding into database
                name = filename.split('.')[0]
                insert_face_embedding(name, embedding)

            print(f"Processed {filename}")

# Collect faces from the dataset
collect_faces_from_folder("Desktop/FYP/Datasets/img_align_celeba")
