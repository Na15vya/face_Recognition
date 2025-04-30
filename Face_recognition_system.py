import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd
from datetime import datetime
from tkinter import *
from tkinter import ttk, messagebox

class FaceRecognitionAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("800x600")
        
        # Title
        title = Label(self.root, text="Face Recognition Attendance System", font=("Helvetica", 20, "bold"))
        title.pack(pady=20)

        # Buttons
        btn_frame = Frame(self.root)
        btn_frame.pack(pady=20)

        register_btn = Button(btn_frame, text="Register New Student", command=self.register_student, width=25)
        register_btn.grid(row=0, column=0, padx=10, pady=10)

        train_btn = Button(btn_frame, text="Train Model", command=self.train_model, width=25)
        train_btn.grid(row=0, column=1, padx=10, pady=10)

        recognize_btn = Button(btn_frame, text="Mark Attendance", command=self.mark_attendance, width=25)
        recognize_btn.grid(row=1, column=0, padx=10, pady=10)

        view_btn = Button(btn_frame, text="View Attendance", command=self.view_attendance, width=25)
        view_btn.grid(row=1, column=1, padx=10, pady=10)

        exit_btn = Button(self.root, text="Exit", command=self.root.quit, width=25)
        exit_btn.pack(pady=10)

    def register_student(self):
        def capture_images():
            user_id = "23D41A67C4"  # Hardcoded ID
            name = "Nagaraju"  # Hardcoded Name
            if not user_id or not name:
                messagebox.showerror("Error", "Please enter both ID and Name.")
                return

            # Use an existing image for registration
            face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            image_path = "C:/Users/nagar/OneDrive/Documents/WhatsApp Image 2025-04-30 at 12.29.58_f3d74bfb.jpg"
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            
            os.makedirs(f"dataset/User.{user_id}", exist_ok=True)

            count = 0
            for (x, y, w, h) in faces:
                count += 1
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                cv2.imwrite(f"dataset/User.{user_id}/{name}.{count}.jpg", face)
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(image, f"Image {count}/50", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Captured Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            messagebox.showinfo("Success", f"Images captured for ID: {user_id}, Name: {name}")
            reg_window.destroy()

        reg_window = Toplevel(self.root)
        reg_window.title("Register New Student")
        reg_window.geometry("400x200")

        Label(reg_window, text="Enter ID:", font=("Helvetica", 12)).pack(pady=5)
        id_entry = Entry(reg_window, font=("Helvetica", 12))
        id_entry.insert(0, "23D41A67C4")  # Pre-fill with ID
        id_entry.pack(pady=5)

        Label(reg_window, text="Enter Name:", font=("Helvetica", 12)).pack(pady=5)
        name_entry = Entry(reg_window, font=("Helvetica", 12))
        name_entry.insert(0, "Nagaraju")  # Pre-fill with Name
        name_entry.pack(pady=5)

        Button(reg_window, text="Capture Images", command=capture_images, font=("Helvetica", 12)).pack(pady=10)

    def train_model(self):
        data_dir = "dataset"
        faces = []
        ids = []

        for user_folder in os.listdir(data_dir):
            user_path = os.path.join(data_dir, user_folder)
            if not os.path.isdir(user_path):
                continue
            user_id = int(user_folder.split(".")[1])
            for image_file in os.listdir(user_path):
                image_path = os.path.join(user_path, image_file)
                pil_img = Image.open(image_path).convert('L')
                img_np = np.array(pil_img, 'uint8')
                faces.append(img_np)
                ids.append(user_id)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        os.makedirs("trainer", exist_ok=True)
        recognizer.save("trainer/trainer.yml")
        messagebox.showinfo("Success", "Model trained and saved successfully.")

    def mark_attendance(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainer/trainer.yml")
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        cap = cv2.VideoCapture(0)
        attendance = set()

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                id_, conf = recognizer.predict(face)
                if conf < 50:
                    name = self.get_name_by_id(id_)
                    if name and id_ not in attendance:
                        attendance.add(id_)
                        now = datetime.now()
                        dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
                        with open("attendance.csv", "a") as f:
                            f.write(f"{id_},{name},{dt_string}\n")
                        cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow("Mark Attendance", frame)
            if cv2.waitKey(1) == 13:
                break
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Success", "Attendance marked successfully.")

    def get_name_by_id(self, id_):
        data_dir = "dataset"
        for user_folder in os.listdir(data_dir):
            if not os.path.isdir(os.path.join(data_dir, user_folder)):
                continue
            folder_id = int(user_folder.split(".")[1])
            if folder_id == id_:
                return user_folder.split(".")[0]
        return None

    def view_attendance(self):
        if not os.path.exists("attendance.csv"):
            messagebox.showerror("Error", "No attendance records found.")
            return
        df = pd.read_csv("attendance.csv", names=["ID", "Name", "DateTime"])
        df["Date"] = pd.to_datetime(df["DateTime"]).dt.date
        df["Time"] = pd.to_datetime(df["DateTime"]).dt.time

        # Calculate attendance percentage
        attendance_summary = df.groupby(["ID", "Name", "Date"]).size().reset_index(name="Count")
        total_days = df["Date"].nunique()
        attendance_percentage = attendance_summary.groupby(["ID", "Name"])["Date"].count().reset_index()
        attendance_percentage["Attendance (%)"] = (attendance_percentage["Date"] / total_days) * 100

        # Display in new window
        view_window = Toplevel(self.root)
        view_window.title("Attendance Records")
        view_window.geometry("600x400")

        tree = ttk.Treeview(view_window, columns=("ID", "Name", "Attendance (%)"), show='headings')
        tree.heading("ID", text="ID")
        tree.heading("Name", text="Name")
        tree.heading("Attendance (%)", text="Attendance (%)")
        tree.pack(fill=BOTH, expand=True)

        for index, row in attendance_percentage.iterrows():
            tree.insert("", END, values=(row["ID"], row["Name"], f"{row['Attendance (%)']:.2f}"))

if __name__ == "__main__":
    root = Tk()
    app = FaceRecognitionAttendanceSystem(root)
    root.mainloop()
