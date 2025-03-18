import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import dlib
import numpy as np
import mysql.connector
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# Load pre-trained models
facenet_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("path/to/shape_predictor_68_face_landmarks.dat")  # Correct path here

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",  # Change to your MySQL password
    database="face_recognition_system"
)
cursor = db.cursor()

# Function to detect faces in an image
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces

# Function to recognize a face using VGGFace
def recognize_face(face_image):
    # Resize the face image to the required input size for VGGFace (224x224)
    face_image = cv2.resize(face_image, (224, 224))
    
    # Convert the image to float32 and normalize pixel values to [0, 1]
    face_image = face_image.astype('float32') / 255.0
    
    # Expand dimensions to match the input shape expected by VGGFace (batch size of 1)
    face_image = np.expand_dims(face_image, axis=0)
    
    # Preprocess the image for VGGFace
    face_image = preprocess_input(face_image, version=2)
    
    # Generate the face embedding using the VGGFace model
    embedding = facenet_model.predict(face_image)
    return embedding.flatten()  # Flatten the embedding to 1D array

# Function to store new face embedding into the database
def store_face_embedding(name, embedding):
    # Convert embedding to a binary format for storage
    embedding_blob = embedding.tobytes()
    
    # Insert the new face's name and embedding into the database
    cursor.execute("INSERT INTO faces (name, embedding) VALUES (%s, %s)", (name, embedding_blob))
    db.commit()

    messagebox.showinfo("Success", "Face data saved successfully!")

# Data Collection Class
class DataCollection:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Data Collection")
        self.root.geometry("800x600")

        # Live Feed Panel
        self.live_feed_label = tk.Label(root)
        self.live_feed_label.pack()

        # Control Buttons
        self.start_button = tk.Button(root, text="Start Data Collection", command=self.start_data_collection)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.quit_button = tk.Button(root, text="Quit", command=self.root.quit)
        self.quit_button.pack(side=tk.LEFT, padx=10)

        # Start webcam feed
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def start_data_collection(self):
        # Prompt user for name
        name = simpledialog.askstring("Input", "Enter the name of the person:")
        if name is None or name == "":
            messagebox.showwarning("Input Error", "Name cannot be empty!")
            return

        messagebox.showinfo("Instruction", "Please position the face within the camera frame for collection.")

        # Capture face for registration
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect faces in the frame
            faces = detect_faces(frame)

            # If exactly one face is detected, process it
            if len(faces) == 1:
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    face_image = frame[y:y+h, x:x+w]

                    # Recognize the face and extract the embedding
                    embedding = recognize_face(face_image)

                    # Store the embedding in the database
                    store_face_embedding(name, embedding)

                    # Draw a rectangle around the detected face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Update the live feed display
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = ImageTk.PhotoImage(Image.fromarray(frame))
                    self.live_feed_label.config(image=img)
                    self.live_feed_label.image = img
                break

            # Display the frame with face detection
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.live_feed_label.config(image=img)
            self.live_feed_label.image = img

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.live_feed_label.config(image=img)
            self.live_feed_label.image = img
        self.root.after(10, self.update_frame)

# Create the Tkinter window
root = tk.Tk()
app = DataCollection(root)
root.mainloop()
