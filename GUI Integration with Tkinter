import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

class MainDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")

        # Live Feed Panel
        self.live_feed_label = tk.Label(root)
        self.live_feed_label.pack()

        # Recognition Results
        self.result_label = tk.Label(root, text="Recognition Results: None", font=("Arial", 14))
        self.result_label.pack()

        # Control Buttons
        self.start_button = tk.Button(root, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, padx=10)

        self.database_button = tk.Button(root, text="View Database", command=self.view_database)
        self.database_button.pack(side=tk.LEFT, padx=10)

        # Start webcam feed
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def start_recognition(self):
        print("Recognition started")

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            print(f"Image uploaded: {file_path}")

    def view_database(self):
        print("Opening Database Viewer")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = process_frame(frame)  # Process the frame for face recognition
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.live_feed_label.config(image=img)
            self.live_feed_label.image = img
        self.root.after(10, self.update_frame)

# Create the Tkinter window
root = tk.Tk()
app = MainDashboard(root)
root.mainloop()
