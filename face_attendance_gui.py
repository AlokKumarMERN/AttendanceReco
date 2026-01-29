"""
Face Authentication Attendance System - Modern GUI (Fixed Version)
A beautiful, interactive interface for face recognition attendance
"""
import cv2
import os
import pickle
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
from datetime import datetime
import time

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DB = os.path.join(BASE_DIR, "faces_database")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance_logs")
RECOGNIZER_FILE = os.path.join(BASE_DIR, "face_recognizer.yml")
LABELS_FILE = os.path.join(BASE_DIR, "face_labels.pkl")

os.makedirs(FACES_DB, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Settings
CAMERA_INDEX = 0
SAMPLES_PER_PERSON = 20  # Reduced for faster registration
RECOGNITION_THRESHOLD = 85


class ModernStyle:
    """Modern color scheme and styling"""
    BG_DARK = "#1a1a2e"
    BG_MEDIUM = "#16213e"
    BG_LIGHT = "#0f3460"
    ACCENT = "#e94560"
    ACCENT_HOVER = "#ff6b6b"
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#a0a0a0"
    SUCCESS = "#00d26a"
    WARNING = "#ffc107"
    ERROR = "#ff4757"
    
    FONT_TITLE = ("Segoe UI", 24, "bold")
    FONT_HEADING = ("Segoe UI", 16, "bold")
    FONT_BODY = ("Segoe UI", 12)
    FONT_SMALL = ("Segoe UI", 10)
    FONT_BUTTON = ("Segoe UI", 11, "bold")


class FaceSystem:
    """Face recognition backend"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        self.labels = {}
        self.name_to_id = {}
        self.next_id = 0
        self.model_trained = False
        self.load_model()
        
        self.attendance_today = {}
        self.load_today_attendance()
    
    def load_model(self):
        if os.path.exists(RECOGNIZER_FILE) and os.path.exists(LABELS_FILE):
            try:
                self.recognizer.read(RECOGNIZER_FILE)
                with open(LABELS_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.labels = data['labels']
                    self.name_to_id = data['name_to_id']
                    self.next_id = data['next_id']
                self.model_trained = True
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model_trained = False
    
    def save_model(self):
        try:
            self.recognizer.save(RECOGNIZER_FILE)
            with open(LABELS_FILE, 'wb') as f:
                pickle.dump({
                    'labels': self.labels,
                    'name_to_id': self.name_to_id,
                    'next_id': self.next_id
                }, f)
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def get_registered_users(self):
        return list(self.name_to_id.keys())
    
    def detect_faces(self, gray_frame):
        faces = self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        return faces
    
    def get_nearest_face(self, faces):
        if len(faces) == 0:
            return None, -1
        if len(faces) == 1:
            return faces[0], 0
        areas = [(w * h, i) for i, (x, y, w, h) in enumerate(faces)]
        areas.sort(reverse=True)
        return faces[areas[0][1]], areas[0][1]
    
    def enhance_image(self, gray):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def preprocess_face(self, face_img):
        face_resized = cv2.resize(face_img, (200, 200))
        return cv2.equalizeHist(face_resized)
    
    def recognize_face(self, gray_face):
        if not self.model_trained or not self.labels:
            return "Unknown", 0
        try:
            processed = self.preprocess_face(gray_face)
            label, distance = self.recognizer.predict(processed)
            confidence = max(0, 100 - distance) / 100.0
            if distance < RECOGNITION_THRESHOLD:
                name = self.labels.get(label, "Unknown")
                return name, confidence
            return "Unknown", 0
        except:
            return "Unknown", 0
    
    def load_today_attendance(self):
        today = datetime.now().strftime("%Y-%m-%d")
        today_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
        self.attendance_today = {}
        if os.path.exists(today_file):
            with open(today_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3 and parts[0] != 'Name':
                        self.attendance_today[parts[0]] = parts[1]
    
    def record_attendance(self, name, confidence):
        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%H:%M:%S")
        
        if name not in self.attendance_today or self.attendance_today[name] == "PUNCH-OUT":
            action = "PUNCH-IN"
        else:
            action = "PUNCH-OUT"
        
        today_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
        file_exists = os.path.exists(today_file)
        
        with open(today_file, 'a') as f:
            if not file_exists:
                f.write("Name,Action,Time,Date,Confidence\n")
            f.write(f"{name},{action},{now},{today},{confidence:.2f}\n")
        
        self.attendance_today[name] = action
        return action, now
    
    def delete_user(self, name):
        if name not in self.name_to_id:
            return False
        
        user_id = self.name_to_id[name]
        del self.labels[user_id]
        del self.name_to_id[name]
        
        user_dir = os.path.join(FACES_DB, name)
        if os.path.exists(user_dir):
            import shutil
            shutil.rmtree(user_dir)
        
        if self.name_to_id:
            self.retrain_model()
        else:
            self.model_trained = False
            if os.path.exists(RECOGNIZER_FILE):
                os.remove(RECOGNIZER_FILE)
            if os.path.exists(LABELS_FILE):
                os.remove(LABELS_FILE)
        return True
    
    def retrain_model(self):
        """Retrain the model with all registered faces"""
        faces = []
        labels = []
        
        for name, user_id in self.name_to_id.items():
            user_dir = os.path.join(FACES_DB, name)
            if os.path.exists(user_dir):
                for img_name in os.listdir(user_dir):
                    img_path = os.path.join(user_dir, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        processed = self.preprocess_face(img)
                        faces.append(processed)
                        labels.append(user_id)
        
        if faces:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.train(faces, np.array(labels))
            self.model_trained = True
            self.save_model()
            return True
        return False


class FaceAttendanceGUI:
    """Modern GUI for Face Attendance System"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Attendance System")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)
        self.root.configure(bg=ModernStyle.BG_DARK)
        
        # Face system
        self.face_system = FaceSystem()
        
        # Camera
        self.cap = None
        self.is_camera_running = False
        self.current_frame = None
        self.recognized_name = None
        self.recognized_conf = 0
        
        # Registration state
        self.is_registering = False
        self.registration_name = ""
        self.registration_count = 0
        self.last_capture_time = 0
        
        # Message display
        self.message_text = ""
        self.message_color = ModernStyle.SUCCESS
        self.message_time = 0
        
        # Setup UI
        self.create_ui()
        
        # Start camera
        self.start_camera()
        
        # Update clock
        self.update_clock()
    
    def create_rounded_button(self, parent, text, command, color=None, width=20):
        """Create a modern styled button"""
        if color is None:
            color = ModernStyle.ACCENT
        
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=ModernStyle.FONT_BUTTON,
            bg=color,
            fg=ModernStyle.TEXT_PRIMARY,
            activebackground=ModernStyle.ACCENT_HOVER,
            activeforeground=ModernStyle.TEXT_PRIMARY,
            relief="flat",
            cursor="hand2",
            width=width,
            pady=12
        )
        
        original_color = color
        def on_enter(e):
            if btn['state'] != 'disabled':
                btn['background'] = ModernStyle.ACCENT_HOVER
        def on_leave(e):
            if btn['state'] != 'disabled':
                btn['background'] = original_color
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn
    
    def create_ui(self):
        """Create the main UI"""
        main_container = tk.Frame(self.root, bg=ModernStyle.BG_DARK)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        self.create_header(main_container)
        
        # Content area
        content = tk.Frame(main_container, bg=ModernStyle.BG_DARK)
        content.pack(fill="both", expand=True, pady=15)
        
        # Left panel - Camera
        self.create_camera_panel(content)
        
        # Right panel - Controls & Info
        self.create_control_panel(content)
    
    def create_header(self, parent):
        """Create header with title and clock"""
        header = tk.Frame(parent, bg=ModernStyle.BG_DARK)
        header.pack(fill="x", pady=(0, 10))
        
        # Title
        tk.Label(
            header,
            text="👤 Face Attendance System",
            font=ModernStyle.FONT_TITLE,
            bg=ModernStyle.BG_DARK,
            fg=ModernStyle.TEXT_PRIMARY
        ).pack(side="left")
        
        # Clock
        clock_frame = tk.Frame(header, bg=ModernStyle.BG_DARK)
        clock_frame.pack(side="right")
        
        self.date_label = tk.Label(
            clock_frame,
            font=ModernStyle.FONT_BODY,
            bg=ModernStyle.BG_DARK,
            fg=ModernStyle.TEXT_SECONDARY
        )
        self.date_label.pack()
        
        self.time_label = tk.Label(
            clock_frame,
            font=("Segoe UI", 22, "bold"),
            bg=ModernStyle.BG_DARK,
            fg=ModernStyle.ACCENT
        )
        self.time_label.pack()
    
    def create_camera_panel(self, parent):
        """Create camera display panel"""
        camera_frame = tk.Frame(parent, bg=ModernStyle.BG_MEDIUM, padx=3, pady=3)
        camera_frame.pack(side="left", fill="both", expand=True, padx=(0, 15))
        
        # Camera header
        cam_header = tk.Frame(camera_frame, bg=ModernStyle.BG_MEDIUM)
        cam_header.pack(fill="x", padx=10, pady=10)
        
        tk.Label(
            cam_header,
            text="📷 Live Camera Feed",
            font=ModernStyle.FONT_HEADING,
            bg=ModernStyle.BG_MEDIUM,
            fg=ModernStyle.TEXT_PRIMARY
        ).pack(side="left")
        
        self.status_indicator = tk.Label(
            cam_header,
            text="● ACTIVE",
            font=ModernStyle.FONT_SMALL,
            bg=ModernStyle.BG_MEDIUM,
            fg=ModernStyle.SUCCESS
        )
        self.status_indicator.pack(side="right")
        
        # Camera display
        self.camera_label = tk.Label(camera_frame, bg="#000000")
        self.camera_label.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Recognition info bar
        self.info_bar = tk.Frame(camera_frame, bg=ModernStyle.BG_LIGHT, height=70)
        self.info_bar.pack(fill="x", padx=10, pady=(0, 10))
        self.info_bar.pack_propagate(False)
        
        self.recognition_label = tk.Label(
            self.info_bar,
            text="👀 Looking for faces...",
            font=("Segoe UI", 14, "bold"),
            bg=ModernStyle.BG_LIGHT,
            fg=ModernStyle.TEXT_SECONDARY
        )
        self.recognition_label.pack(expand=True)
        
        # Progress bar for registration
        self.progress_frame = tk.Frame(camera_frame, bg=ModernStyle.BG_MEDIUM)
        self.progress_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="",
            font=ModernStyle.FONT_BODY,
            bg=ModernStyle.BG_MEDIUM,
            fg=ModernStyle.WARNING
        )
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            length=400,
            mode='determinate',
            maximum=SAMPLES_PER_PERSON
        )
    
    def create_control_panel(self, parent):
        """Create control panel with buttons and info"""
        control_frame = tk.Frame(parent, bg=ModernStyle.BG_DARK, width=380)
        control_frame.pack(side="right", fill="y")
        control_frame.pack_propagate(False)
        
        # Action buttons card
        actions_card = tk.Frame(control_frame, bg=ModernStyle.BG_MEDIUM, padx=20, pady=20)
        actions_card.pack(fill="x", pady=(0, 15))
        
        tk.Label(
            actions_card,
            text="⚡ Quick Actions",
            font=ModernStyle.FONT_HEADING,
            bg=ModernStyle.BG_MEDIUM,
            fg=ModernStyle.TEXT_PRIMARY
        ).pack(anchor="w", pady=(0, 15))
        
        # Attendance button
        self.attendance_btn = self.create_rounded_button(
            actions_card,
            "✓ Mark Attendance",
            self.mark_attendance,
            ModernStyle.SUCCESS
        )
        self.attendance_btn.pack(fill="x", pady=5)
        
        # Register button
        self.register_btn = self.create_rounded_button(
            actions_card,
            "➕ Register New Face",
            self.start_registration,
            ModernStyle.BG_LIGHT
        )
        self.register_btn.pack(fill="x", pady=5)
        
        # Cancel registration button (hidden initially)
        self.cancel_btn = self.create_rounded_button(
            actions_card,
            "✖ Cancel Registration",
            self.cancel_registration,
            ModernStyle.ERROR
        )
        
        # Users card
        users_card = tk.Frame(control_frame, bg=ModernStyle.BG_MEDIUM, padx=20, pady=15)
        users_card.pack(fill="x", pady=(0, 15))
        
        users_header = tk.Frame(users_card, bg=ModernStyle.BG_MEDIUM)
        users_header.pack(fill="x", pady=(0, 10))
        
        tk.Label(
            users_header,
            text="👥 Registered Users",
            font=ModernStyle.FONT_HEADING,
            bg=ModernStyle.BG_MEDIUM,
            fg=ModernStyle.TEXT_PRIMARY
        ).pack(side="left")
        
        self.user_count_label = tk.Label(
            users_header,
            text="0",
            font=ModernStyle.FONT_BODY,
            bg=ModernStyle.ACCENT,
            fg=ModernStyle.TEXT_PRIMARY,
            padx=10,
            pady=2
        )
        self.user_count_label.pack(side="right")
        
        # Users listbox
        list_frame = tk.Frame(users_card, bg=ModernStyle.BG_MEDIUM)
        list_frame.pack(fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.users_listbox = tk.Listbox(
            list_frame,
            font=ModernStyle.FONT_BODY,
            bg=ModernStyle.BG_DARK,
            fg=ModernStyle.TEXT_PRIMARY,
            selectbackground=ModernStyle.ACCENT,
            selectforeground=ModernStyle.TEXT_PRIMARY,
            relief="flat",
            height=5,
            yscrollcommand=scrollbar.set
        )
        self.users_listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.users_listbox.yview)
        
        # Delete button
        delete_btn = self.create_rounded_button(
            users_card,
            "🗑️ Delete Selected",
            self.delete_user,
            ModernStyle.ERROR,
            width=15
        )
        delete_btn.pack(pady=(10, 0))
        
        # Today's attendance card
        attendance_card = tk.Frame(control_frame, bg=ModernStyle.BG_MEDIUM, padx=20, pady=15)
        attendance_card.pack(fill="both", expand=True)
        
        att_header = tk.Frame(attendance_card, bg=ModernStyle.BG_MEDIUM)
        att_header.pack(fill="x", pady=(0, 10))
        
        tk.Label(
            att_header,
            text="📋 Today's Attendance",
            font=ModernStyle.FONT_HEADING,
            bg=ModernStyle.BG_MEDIUM,
            fg=ModernStyle.TEXT_PRIMARY
        ).pack(side="left")
        
        refresh_btn = tk.Button(
            att_header,
            text="🔄",
            font=ModernStyle.FONT_BODY,
            bg=ModernStyle.BG_LIGHT,
            fg=ModernStyle.TEXT_PRIMARY,
            relief="flat",
            command=self.refresh_attendance_list
        )
        refresh_btn.pack(side="right")
        
        # Attendance listbox
        att_frame = tk.Frame(attendance_card, bg=ModernStyle.BG_MEDIUM)
        att_frame.pack(fill="both", expand=True)
        
        att_scrollbar = tk.Scrollbar(att_frame)
        att_scrollbar.pack(side="right", fill="y")
        
        self.attendance_listbox = tk.Listbox(
            att_frame,
            font=ModernStyle.FONT_BODY,
            bg=ModernStyle.BG_DARK,
            fg=ModernStyle.TEXT_PRIMARY,
            relief="flat",
            height=6,
            yscrollcommand=att_scrollbar.set
        )
        self.attendance_listbox.pack(fill="both", expand=True)
        att_scrollbar.config(command=self.attendance_listbox.yview)
        
        # Refresh lists
        self.refresh_users_list()
        self.refresh_attendance_list()
    
    def update_clock(self):
        """Update the clock display"""
        now = datetime.now()
        self.time_label.config(text=now.strftime("%H:%M:%S"))
        self.date_label.config(text=now.strftime("%A, %B %d, %Y"))
        self.root.after(1000, self.update_clock)
    
    def start_camera(self):
        """Start the camera capture"""
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if self.cap.isOpened():
            self.is_camera_running = True
            self.update_camera()
        else:
            messagebox.showerror("Error", "Could not open camera")
    
    def update_camera(self):
        """Update camera frame - runs continuously"""
        if not self.is_camera_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            # Process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = self.face_system.enhance_image(gray)
            faces = self.face_system.detect_faces(gray)
            
            current_time = time.time()
            
            # Reset recognition if no faces
            if len(faces) == 0:
                self.recognized_name = None
                self.recognized_conf = 0
                self.recognition_label.config(
                    text="👀 Looking for faces...",
                    fg=ModernStyle.TEXT_SECONDARY
                )
            else:
                nearest_face, nearest_idx = self.face_system.get_nearest_face(faces)
                
                for idx, (x, y, w, h) in enumerate(faces):
                    is_nearest = (idx == nearest_idx)
                    face_roi = gray[y:y+h, x:x+w]
                    
                    if self.is_registering:
                        # Registration mode - capture samples quickly
                        color = (0, 165, 255)  # Orange BGR
                        
                        if is_nearest:
                            # Capture every 100ms for faster registration
                            if current_time - self.last_capture_time >= 0.1:
                                if self.registration_count < SAMPLES_PER_PERSON:
                                    # Save face sample
                                    user_dir = os.path.join(FACES_DB, self.registration_name)
                                    cv2.imwrite(
                                        os.path.join(user_dir, f"sample_{self.registration_count + 1}.jpg"),
                                        face_roi
                                    )
                                    self.registration_count += 1
                                    self.last_capture_time = current_time
                                    
                                    # Update progress
                                    self.progress_bar['value'] = self.registration_count
                                    self.progress_label.config(
                                        text=f"📸 Capturing: {self.registration_count}/{SAMPLES_PER_PERSON}"
                                    )
                                    
                                    # Check if done
                                    if self.registration_count >= SAMPLES_PER_PERSON:
                                        self.root.after(100, self.finish_registration)
                            
                            self.recognition_label.config(
                                text=f"📸 Registering: {self.registration_name} ({self.registration_count}/{SAMPLES_PER_PERSON})",
                                fg=ModernStyle.WARNING
                            )
                    else:
                        # Recognition mode
                        name, conf = self.face_system.recognize_face(face_roi)
                        
                        if is_nearest:
                            self.recognized_name = name
                            self.recognized_conf = conf
                        
                        # Set color based on recognition
                        if name == "Unknown":
                            color = (0, 0, 255)  # Red BGR
                        else:
                            color = (0, 255, 0)  # Green BGR
                    
                    # Draw face rectangle
                    thickness = 3 if is_nearest else 2
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                    
                    # Draw NEAREST tag for closest face
                    if is_nearest:
                        cv2.rectangle(frame, (x-3, y-3), (x+w+3, y+h+3), (255, 0, 0), 2)
                        cv2.putText(frame, "NEAREST", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Draw name label (only in recognition mode)
                    if not self.is_registering:
                        if name == "Unknown":
                            label = "UNKNOWN"
                            label_color = (0, 0, 255)
                        else:
                            label = f"{name} ({conf:.0%})"
                            label_color = (0, 255, 0)
                        
                        # Label background
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x, y+h+5), (x+label_w+10, y+h+label_h+15), label_color, -1)
                        cv2.putText(frame, label, (x+5, y+h+label_h+10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Update recognition label
                if not self.is_registering:
                    if self.recognized_name == "Unknown":
                        self.recognition_label.config(
                            text="❌ UNKNOWN - Not Registered",
                            fg=ModernStyle.ERROR
                        )
                    elif self.recognized_name:
                        status = self.face_system.attendance_today.get(self.recognized_name, "Not checked in")
                        next_action = "PUNCH-OUT" if status == "PUNCH-IN" else "PUNCH-IN"
                        self.recognition_label.config(
                            text=f"✓ {self.recognized_name} ({self.recognized_conf:.0%}) → Ready for {next_action}",
                            fg=ModernStyle.SUCCESS
                        )
            
            # Draw message overlay if active
            if self.message_text and current_time - self.message_time < 3:
                overlay = frame.copy()
                h, w = frame.shape[:2]
                cv2.rectangle(overlay, (w//2-200, h//2-40), (w//2+200, h//2+40), 
                             (0, 200, 0) if self.message_color == ModernStyle.SUCCESS else (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                cv2.putText(frame, self.message_text, (w//2-180, h//2+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Convert to PhotoImage and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.camera_label.config(image=photo)
            self.camera_label.image = photo
        
        # Continue camera loop (always running)
        self.root.after(30, self.update_camera)
    
    def mark_attendance(self):
        """Mark attendance for recognized face - only for known faces"""
        if self.is_registering:
            messagebox.showwarning("Warning", "Please finish registration first!")
            return
        
        if not self.recognized_name:
            messagebox.showwarning("No Face", "No face detected. Please look at the camera.")
            return
        
        if self.recognized_name == "Unknown":
            messagebox.showerror(
                "Unknown Face",
                "❌ Face not recognized!\n\nPlease register first using 'Register New Face' button."
            )
            return
        
        # Mark attendance for recognized person
        action, time_str = self.face_system.record_attendance(
            self.recognized_name, self.recognized_conf
        )
        
        self.refresh_attendance_list()
        
        # Show message on camera
        self.message_text = f"{self.recognized_name}: {action}"
        self.message_color = ModernStyle.SUCCESS
        self.message_time = time.time()
        
        messagebox.showinfo(
            "✓ Attendance Marked",
            f"Name: {self.recognized_name}\n"
            f"Action: {action}\n"
            f"Time: {time_str}"
        )
    
    def start_registration(self):
        """Start face registration process"""
        if self.is_registering:
            return
        
        name = simpledialog.askstring(
            "Register New Face",
            "Enter name for the new user:",
            parent=self.root
        )
        
        if not name or not name.strip():
            return
        
        name = name.strip()
        
        if name in self.face_system.get_registered_users():
            if not messagebox.askyesno("User Exists", f"'{name}' already exists. Overwrite?"):
                return
            self.face_system.delete_user(name)
            self.refresh_users_list()
        
        # Create user directory
        user_dir = os.path.join(FACES_DB, name)
        os.makedirs(user_dir, exist_ok=True)
        
        # Start registration
        self.is_registering = True
        self.registration_name = name
        self.registration_count = 0
        self.last_capture_time = 0
        
        # Update UI
        self.register_btn.pack_forget()
        self.cancel_btn.pack(fill="x", pady=5)
        self.attendance_btn.config(state="disabled")
        
        # Show progress bar
        self.progress_bar['value'] = 0
        self.progress_bar.pack(pady=5)
        self.progress_label.config(text=f"📸 Look at camera - Capturing faces...")
        
        self.status_indicator.config(text="● REGISTERING", fg=ModernStyle.WARNING)
    
    def cancel_registration(self):
        """Cancel ongoing registration"""
        if not self.is_registering:
            return
        
        # Clean up partial registration
        user_dir = os.path.join(FACES_DB, self.registration_name)
        if os.path.exists(user_dir):
            import shutil
            shutil.rmtree(user_dir)
        
        self.reset_registration_ui()
        messagebox.showinfo("Cancelled", "Registration cancelled.")
    
    def finish_registration(self):
        """Finish registration and train model"""
        if not self.is_registering:
            return
        
        name = self.registration_name
        
        # Assign ID
        if name not in self.face_system.name_to_id:
            user_id = self.face_system.next_id
            self.face_system.next_id += 1
            self.face_system.name_to_id[name] = user_id
            self.face_system.labels[user_id] = name
        
        # Train model in background
        self.progress_label.config(text="🔄 Training model... Please wait")
        self.root.update()
        
        success = self.face_system.retrain_model()
        
        self.reset_registration_ui()
        self.refresh_users_list()
        
        if success:
            messagebox.showinfo(
                "✓ Registration Complete",
                f"Successfully registered '{name}'!\n\n"
                f"Captured {SAMPLES_PER_PERSON} face samples.\n"
                f"You can now mark attendance."
            )
        else:
            messagebox.showerror("Error", "Failed to train model. Please try again.")
    
    def reset_registration_ui(self):
        """Reset UI after registration"""
        self.is_registering = False
        self.registration_name = ""
        self.registration_count = 0
        
        self.cancel_btn.pack_forget()
        self.register_btn.pack(fill="x", pady=5)
        self.attendance_btn.config(state="normal")
        
        self.progress_bar.pack_forget()
        self.progress_label.config(text="")
        
        self.status_indicator.config(text="● ACTIVE", fg=ModernStyle.SUCCESS)
    
    def delete_user(self):
        """Delete selected user"""
        selection = self.users_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a user to delete")
            return
        
        name = self.users_listbox.get(selection[0])
        
        if messagebox.askyesno("Confirm Delete", f"Delete user '{name}'?\n\nThis cannot be undone."):
            if self.face_system.delete_user(name):
                self.refresh_users_list()
                messagebox.showinfo("Deleted", f"User '{name}' has been deleted.")
    
    def refresh_users_list(self):
        """Refresh the users listbox"""
        self.users_listbox.delete(0, tk.END)
        users = self.face_system.get_registered_users()
        for user in sorted(users):
            self.users_listbox.insert(tk.END, user)
        self.user_count_label.config(text=str(len(users)))
    
    def refresh_attendance_list(self):
        """Refresh the attendance listbox"""
        self.attendance_listbox.delete(0, tk.END)
        self.face_system.load_today_attendance()
        
        if not self.face_system.attendance_today:
            self.attendance_listbox.insert(tk.END, "  No attendance yet")
        else:
            for name, action in self.face_system.attendance_today.items():
                status_icon = "🟢" if action == "PUNCH-IN" else "🔴"
                self.attendance_listbox.insert(tk.END, f" {status_icon} {name}: {action}")
    
    def on_closing(self):
        """Handle window close"""
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()
    
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


if __name__ == "__main__":
    app = FaceAttendanceGUI()
    app.run()
