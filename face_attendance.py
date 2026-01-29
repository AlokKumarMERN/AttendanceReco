"""
Face Authentication Attendance System - Simple OpenCV Implementation
Uses Haar Cascades for detection and LBPH for recognition
No external model downloads required!
"""
import cv2
import os
import pickle
import numpy as np
from datetime import datetime
from collections import defaultdict

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
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SAMPLES_PER_PERSON = 30
RECOGNITION_THRESHOLD = 80  # Lower = stricter (LBPH returns distance, lower is better)


class FaceSystem:
    def __init__(self):
        print("="*55)
        print("  Face Authentication Attendance System")
        print("="*55)
        
        # Load Haar cascade for face detection (built into OpenCV)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # LBPH Face Recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Labels mapping
        self.labels = {}  # id -> name
        self.name_to_id = {}  # name -> id
        self.next_id = 0
        
        # Load existing model
        self.load_model()
        
        # Attendance tracking
        self.attendance_today = {}
        self.load_today_attendance()
        
        print(f"✓ Loaded {len(self.labels)} registered users")
    
    def load_model(self):
        """Load the trained recognizer and labels"""
        if os.path.exists(RECOGNIZER_FILE) and os.path.exists(LABELS_FILE):
            try:
                self.recognizer.read(RECOGNIZER_FILE)
                with open(LABELS_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.labels = data['labels']
                    self.name_to_id = data['name_to_id']
                    self.next_id = data['next_id']
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
    
    def save_model(self):
        """Save the trained recognizer and labels"""
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
        """Get list of registered users"""
        return list(self.name_to_id.keys())
    
    def detect_faces(self, gray_frame):
        """Detect faces and return list of (x, y, w, h) tuples"""
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        return faces
    
    def get_nearest_face(self, faces, frame_shape):
        """Get the nearest face (largest area)"""
        if len(faces) == 0:
            return None, -1
        
        if len(faces) == 1:
            return faces[0], 0
        
        # Find largest face (closest to camera)
        areas = [(w * h, i) for i, (x, y, w, h) in enumerate(faces)]
        areas.sort(reverse=True)
        
        return faces[areas[0][1]], areas[0][1]
    
    def enhance_image(self, gray):
        """Enhance image for better detection"""
        # Apply CLAHE for lighting normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced
    
    def preprocess_face(self, face_img):
        """Preprocess face for recognition"""
        # Resize to consistent size
        face_resized = cv2.resize(face_img, (200, 200))
        # Normalize
        face_normalized = cv2.equalizeHist(face_resized)
        return face_normalized
    
    def register_face(self, name):
        """Register a new face with multiple samples"""
        print(f"\n--- Registering: {name} ---")
        print("Look at the camera and move your head slowly")
        print("Press 'Q' to cancel\n")
        
        # Create folder for user
        user_dir = os.path.join(FACES_DB, name)
        os.makedirs(user_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        samples = []
        sample_count = 0
        last_capture = 0
        
        while sample_count < SAMPLES_PER_PERSON:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = self.enhance_image(gray)
            
            faces = self.detect_faces(gray)
            display = frame.copy()
            
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Capture sample every 0.2 seconds
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                if current_time - last_capture >= 0.2:
                    face_img = gray[y:y+h, x:x+w]
                    processed = self.preprocess_face(face_img)
                    samples.append(processed)
                    
                    # Save image
                    cv2.imwrite(os.path.join(user_dir, f"sample_{sample_count}.jpg"), face_img)
                    
                    sample_count += 1
                    last_capture = current_time
                    
                    # Visual feedback
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 3)
                
                cv2.putText(display, f"Capturing: {sample_count}/{SAMPLES_PER_PERSON}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif len(faces) == 0:
                cv2.putText(display, "No face - Look at camera",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Multiple faces - Only one person",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Progress bar
            progress = int((sample_count / SAMPLES_PER_PERSON) * (FRAME_WIDTH - 20))
            cv2.rectangle(display, (10, FRAME_HEIGHT-30), (10+progress, FRAME_HEIGHT-20), (0, 255, 0), -1)
            cv2.rectangle(display, (10, FRAME_HEIGHT-30), (FRAME_WIDTH-10, FRAME_HEIGHT-20), (100, 100, 100), 1)
            
            cv2.putText(display, f"Registering: {name} | Press Q to cancel",
                       (10, FRAME_HEIGHT-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Face Registration", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Registration cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        
        if sample_count >= SAMPLES_PER_PERSON:
            # Assign ID to this person
            if name in self.name_to_id:
                user_id = self.name_to_id[name]
            else:
                user_id = self.next_id
                self.next_id += 1
                self.name_to_id[name] = user_id
                self.labels[user_id] = name
            
            # Train or update recognizer
            self._retrain_model()
            
            print(f"\n✓ Successfully registered '{name}' with {sample_count} samples")
            return True
        
        return False
    
    def _retrain_model(self):
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
            self.save_model()
            print(f"Model trained with {len(faces)} samples from {len(self.name_to_id)} users")
    
    def delete_user(self, name):
        """Delete a user"""
        if name not in self.name_to_id:
            print(f"User '{name}' not found")
            return
        
        # Remove from labels
        user_id = self.name_to_id[name]
        del self.labels[user_id]
        del self.name_to_id[name]
        
        # Remove images
        user_dir = os.path.join(FACES_DB, name)
        if os.path.exists(user_dir):
            import shutil
            shutil.rmtree(user_dir)
        
        # Retrain model
        if self.name_to_id:
            self._retrain_model()
        else:
            # No users left, remove model files
            if os.path.exists(RECOGNIZER_FILE):
                os.remove(RECOGNIZER_FILE)
            if os.path.exists(LABELS_FILE):
                os.remove(LABELS_FILE)
        
        print(f"✓ Deleted user '{name}'")
    
    def recognize_face(self, gray_face):
        """Recognize a face, return (name, confidence)"""
        if not self.labels:
            return "Unknown", 0
        
        try:
            processed = self.preprocess_face(gray_face)
            label, distance = self.recognizer.predict(processed)
            
            # Convert distance to confidence (LBPH uses distance, lower = better)
            confidence = max(0, 100 - distance) / 100.0
            
            if distance < RECOGNITION_THRESHOLD:
                name = self.labels.get(label, "Unknown")
                return name, confidence
            else:
                return "Unknown", 0
        except:
            return "Unknown", 0
    
    def check_liveness(self, gray, face_rect):
        """Basic liveness check using eye detection"""
        (x, y, w, h) = face_rect
        roi = gray[y:y+h, x:x+w]
        
        eyes = self.eye_cascade.detectMultiScale(roi, 1.1, 5, minSize=(20, 20))
        
        # Real face should have 2 eyes detected most of the time
        return len(eyes) >= 1
    
    def load_today_attendance(self):
        """Load today's attendance"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
        
        self.attendance_today = {}
        if os.path.exists(today_file):
            with open(today_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3 and parts[0] != 'Name':
                        name = parts[0]
                        action = parts[1]
                        self.attendance_today[name] = action
    
    def record_attendance(self, name, confidence):
        """Record attendance and return (success, message, action)"""
        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%H:%M:%S")
        
        # Determine action
        if name not in self.attendance_today or self.attendance_today[name] == "PUNCH-OUT":
            action = "PUNCH-IN"
        else:
            action = "PUNCH-OUT"
        
        # Save to file
        today_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
        file_exists = os.path.exists(today_file)
        
        with open(today_file, 'a') as f:
            if not file_exists:
                f.write("Name,Action,Time,Date,Confidence\n")
            f.write(f"{name},{action},{now},{today},{confidence:.2f}\n")
        
        self.attendance_today[name] = action
        
        return True, f"{action} at {now}", action
    
    def run_recognition(self):
        """Main recognition loop"""
        print("\n--- Recognition Mode ---")
        print("Controls: Q=Quit | A=Attendance | R=Register | V=View Users")
        print("The NEAREST face (closest to camera) will be prioritized\n")
        
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        message = ""
        message_time = 0
        message_color = (0, 255, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = self.enhance_image(gray)
            
            faces = self.detect_faces(gray)
            
            recognized_name = None
            recognized_conf = 0
            nearest_idx = -1
            
            if len(faces) > 0:
                nearest_face, nearest_idx = self.get_nearest_face(faces, frame.shape)
            
            # Process all faces
            for idx, (x, y, w, h) in enumerate(faces):
                is_nearest = (idx == nearest_idx)
                face_roi = gray[y:y+h, x:x+w]
                
                # Recognize face
                name, conf = self.recognize_face(face_roi)
                
                # Check liveness for nearest face
                is_live = True
                if is_nearest:
                    is_live = self.check_liveness(gray, (x, y, w, h))
                    recognized_name = name
                    recognized_conf = conf
                
                # Colors
                if name == "Unknown":
                    color = (0, 0, 255)  # Red
                else:
                    color = (0, 255, 0)  # Green
                
                # Draw rectangle
                thickness = 3 if is_nearest else 2
                cv2.rectangle(display, (x, y), (x+w, y+h), color, thickness)
                
                # Add "NEAREST" tag
                if is_nearest:
                    cv2.rectangle(display, (x-2, y-2), (x+w+2, y+h+2), (255, 0, 0), 2)
                    cv2.putText(display, "NEAREST", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Label
                label = f"{name}"
                if name != "Unknown":
                    label += f" ({conf:.0%})"
                cv2.putText(display, label, (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Header
            cv2.rectangle(display, (0, 0), (FRAME_WIDTH, 35), (0, 0, 0), -1)
            now = datetime.now().strftime("%H:%M:%S")
            cv2.putText(display, "Face Attendance System", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, now, (FRAME_WIDTH-80, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Status
            if recognized_name:
                if recognized_name != "Unknown":
                    expected = "PUNCH-OUT" if self.attendance_today.get(recognized_name) == "PUNCH-IN" else "PUNCH-IN"
                    status = f"{recognized_name} | Press 'A' for {expected}"
                    status_color = (0, 255, 0)
                else:
                    status = "Unknown Face - Please register first"
                    status_color = (0, 0, 255)
            else:
                status = "No face detected"
                status_color = (150, 150, 150)
            
            cv2.putText(display, status, (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Message display
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if message and current_time - message_time < 3:
                cv2.rectangle(display, (FRAME_WIDTH//2-150, FRAME_HEIGHT//2-25),
                             (FRAME_WIDTH//2+150, FRAME_HEIGHT//2+25), message_color, -1)
                cv2.putText(display, message, (FRAME_WIDTH//2-140, FRAME_HEIGHT//2+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Controls
            cv2.rectangle(display, (0, FRAME_HEIGHT-30), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
            cv2.putText(display, "Q=Quit | A=Attendance | R=Register | V=View Users | H=History",
                       (10, FRAME_HEIGHT-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.imshow("Face Attendance", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('a'):
                if recognized_name and recognized_name != "Unknown":
                    success, msg, action = self.record_attendance(recognized_name, recognized_conf)
                    message = f"{recognized_name}: {action}"
                    message_color = (0, 200, 0)
                    message_time = current_time
                    print(f"✓ {recognized_name}: {action} at {datetime.now().strftime('%H:%M:%S')}")
                else:
                    message = "No recognized face!"
                    message_color = (0, 0, 200)
                    message_time = current_time
            
            elif key == ord('r'):
                cap.release()
                cv2.destroyAllWindows()
                name = input("\nEnter name for registration: ").strip()
                if name:
                    if name in self.name_to_id:
                        overwrite = input(f"'{name}' exists. Overwrite? (y/n): ").lower()
                        if overwrite == 'y':
                            self.delete_user(name)
                            self.register_face(name)
                    else:
                        self.register_face(name)
                cap = cv2.VideoCapture(CAMERA_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                print("\nResuming recognition mode...")
            
            elif key == ord('v'):
                users = self.get_registered_users()
                print("\n--- Registered Users ---")
                for u in users:
                    status = self.attendance_today.get(u, "Not checked in")
                    print(f"  • {u}: {status}")
                print("------------------------\n")
            
            elif key == ord('h'):
                print("\n--- Today's Attendance ---")
                for name, action in self.attendance_today.items():
                    print(f"  • {name}: {action}")
                if not self.attendance_today:
                    print("  No attendance records today")
                print("--------------------------\n")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def show_menu(self):
        """Display main menu"""
        print("\n" + "="*45)
        print("  MAIN MENU")
        print("="*45)
        print("  1. Start Recognition Mode")
        print("  2. Register New Face")
        print("  3. View Registered Users")
        print("  4. View Today's Attendance")
        print("  5. Delete User")
        print("  6. Exit")
        print("="*45)
    
    def run(self):
        """Main application loop"""
        while True:
            self.show_menu()
            choice = input("Enter choice (1-6): ").strip()
            
            if choice == '1':
                self.run_recognition()
            
            elif choice == '2':
                name = input("\nEnter name: ").strip()
                if name:
                    if name in self.name_to_id:
                        overwrite = input(f"'{name}' exists. Overwrite? (y/n): ").lower()
                        if overwrite == 'y':
                            self.delete_user(name)
                            self.register_face(name)
                    else:
                        self.register_face(name)
            
            elif choice == '3':
                users = self.get_registered_users()
                print("\n--- Registered Users ---")
                if users:
                    for u in users:
                        print(f"  • {u}")
                else:
                    print("  No users registered")
                print("------------------------")
            
            elif choice == '4':
                self.load_today_attendance()
                print("\n--- Today's Attendance ---")
                if self.attendance_today:
                    for name, action in self.attendance_today.items():
                        print(f"  • {name}: {action}")
                else:
                    print("  No attendance records today")
                print("--------------------------")
            
            elif choice == '5':
                users = self.get_registered_users()
                if users:
                    print("\nRegistered:", ", ".join(users))
                    name = input("Enter name to delete: ").strip()
                    if name in users:
                        confirm = input(f"Delete '{name}'? (y/n): ").lower()
                        if confirm == 'y':
                            self.delete_user(name)
                    else:
                        print(f"User '{name}' not found")
                else:
                    print("\nNo users to delete")
            
            elif choice == '6':
                print("\nGoodbye!")
                break
            
            else:
                print("\nInvalid choice")


if __name__ == "__main__":
    app = FaceSystem()
    app.run()
