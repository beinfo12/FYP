How to Run the Code:
Install Required Libraries:
Ensure that the necessary libraries are installed:
" pip install op"


Source Code of GUI Tkinter:

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
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

# Function to detect faces in an image
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces

# Function to recognize the face from embeddings
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

    threshold = 0.6  # Recognition threshold
    if best_match_distance < threshold:
        return best_match_name
    else:
        return "Unknown"

# Function to capture a new face and store its embedding in the database
def store_face(name):
    cap = cv2.VideoCapture(0)
    print("Starting face capture...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        faces = detect_faces(frame)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_image = frame[y:y + h, x:x + w]

            # Resize, preprocess and generate embedding
            face_image_resized = cv2.resize(face_image, (224, 224))
            face_image_resized = face_image_resized.astype('float32') / 255.0
            face_image_resized = np.expand_dims(face_image_resized, axis=0)
            face_image_resized = preprocess_input(face_image_resized, version=2)

            embedding = facenet_model.predict(face_image_resized).flatten()

            # Serialize the embedding and store it in the database
            embedding_blob = pickle.dumps(embedding)  # Convert to binary

            cursor.execute("INSERT INTO faces (name, embedding) VALUES (%s, %s)", (name, embedding_blob))
            db.commit()
            print(f"Stored {name}'s face embedding in the database.")

        # Show the frame with detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Face Capture", frame)

        # If 'q' is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to update the live webcam feed in Tkinter
def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        live_feed_label.config(image=img)
        live_feed_label.image = img
    root.after(10, update_frame)

# Function to start the face recognition process
def start_recognition():
    ret, frame = cap.read()
    if ret:
        faces = detect_faces(frame)
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_image = frame[y:y + h, x:x + w]

            # Resize, preprocess and generate embedding
            face_image_resized = cv2.resize(face_image, (224, 224))
            face_image_resized = face_image_resized.astype('float32') / 255.0
            face_image_resized = np.expand_dims(face_image_resized, axis=0)
            face_image_resized = preprocess_input(face_image_resized, version=2)

            embedding = facenet_model.predict(face_image_resized).flatten()

            # Recognize the face
            recognized_name = recognize_face(embedding)

            # Update the result label with the recognized name
            result_label.config(text=f"Recognition Result: {recognized_name}")

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        live_feed_label.config(image=img)
        live_feed_label.image = img

# Function to open file dialog and upload an image for recognition
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        faces = detect_faces(image)
        
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_image = image[y:y + h, x:x + w]
            
            # Resize, preprocess and generate embedding
            face_image_resized = cv2.resize(face_image, (224, 224))
            face_image_resized = face_image_resized.astype('float32') / 255.0
            face_image_resized = np.expand_dims(face_image_resized, axis=0)
            face_image_resized = preprocess_input(face_image_resized, version=2)
            
            embedding = facenet_model.predict(face_image_resized).flatten()

            recognized_name = recognize_face(embedding)
            
            # Update the result label with the recognized name
            result_label.config(text=f"Recognition Result: {recognized_name}")

            # Draw rectangle around the face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(image))
        live_feed_label.config(image=img)
        live_feed_label.image = img

# Function to show database entries in a new window
def show_database():
    database_window = tk.Toplevel()
    database_window.title("Database Entries")
    database_window.geometry("600x400")
    
    tree = ttk.Treeview(database_window, columns=("ID", "Name"), show="headings")
    tree.heading("ID", text="ID")
    tree.heading("Name", text="Name")
    tree.pack(fill="both", expand=True)

    cursor.execute("SELECT id, name FROM faces")
    faces = cursor.fetchall()

    for face in faces:
        tree.insert("", "end", values=face)

# Tkinter GUI Setup
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("800x600")

# Live Feed Panel
live_feed_label = tk.Label(root)
live_feed_label.pack()

# Recognition Results
result_label = tk.Label(root, text="Recognition Result: None", font=("Arial", 14))
result_label.pack()

# Control Buttons
start_button = tk.Button(root, text="Start Recognition", command=start_recognition)
start_button.pack(side=tk.LEFT, padx=10)

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(side=tk.LEFT, padx=10)

add_face_button = tk.Button(root, text="Add Face to Database", command=lambda: store_face(name_entry.get()))
add_face_button.pack(side=tk.LEFT, padx=10)

database_button = tk.Button(root, text="View Database", command=show_database)
database_button.pack(side=tk.LEFT, padx=10)

# Input for new face
name_entry_label = tk.Label(root, text="Enter Name:")
name_entry_label.pack()

name_entry = tk.Entry(root)
name_entry.pack()

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Update the video feed
update_frame()

# Start the Tkinter GUI
root.mainloop()
