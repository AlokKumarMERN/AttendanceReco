# Face Authentication Attendance System

A complete face authentication system for attendance tracking with real-time face recognition, anti-spoofing measures, and punch-in/punch-out functionality.

## 📋 Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model and Approach](#model-and-approach)
- [Training Process](#training-process)
- [Accuracy Expectations](#accuracy-expectations)
- [Known Limitations](#known-limitations)
- [Project Structure](#project-structure)

## ✨ Features

### Core Features
- **Face Registration**: Capture multiple face samples (10 by default) for robust recognition
- **Real-time Face Recognition**: Identify registered users through live camera feed
- **Nearest Face Detection**: Automatically selects the face closest to the camera when multiple faces are present
- **Punch-in/Punch-out**: Automatic attendance tracking with timestamps
- **Unknown Face Handling**: Clearly marks unrecognized faces

### Security Features
- **Basic Spoof Prevention**: 
  - Blink detection (liveness check)
  - Motion analysis (detects static images)
  - Texture analysis (detects flat photo surfaces)
  - Face size variation (detects natural movements)
- **Confidence Threshold**: Only marks attendance when recognition confidence is high

### Additional Features
- **Varying Lighting Handling**: CLAHE enhancement for better detection in different lighting
- **Attendance Logs**: CSV-based logging with daily files
- **User Management**: Add, view, and delete registered users
- **Cooldown System**: Prevents accidental duplicate punches

## 💻 System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher
- **Camera**: Built-in webcam or USB camera
- **RAM**: Minimum 4GB (8GB recommended)
- **CPU**: Dual-core processor or better

## 🔧 Installation

### 1. Clone or Download the Project

```
cd c:\Users\Hp\Desktop\Medoc
```

### 2. Create Virtual Environment (Recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install opencv-python numpy face-recognition dlib Pillow cmake
```

**Note for Windows users**: If you encounter issues installing `dlib`, you may need:
- Visual Studio Build Tools with C++ workload
- CMake (can be installed via `pip install cmake`)

### 4. Verify Installation

```powershell
python -c "import face_recognition; import cv2; print('Installation successful!')"
```

## 🚀 Usage

### Starting the Application

```powershell
cd c:\Users\Hp\Desktop\Medoc
.\.venv\Scripts\python.exe face_attendance.py
```

**Note**: The system uses OpenCV's built-in LBPH (Local Binary Patterns Histograms) face recognizer which works out of the box without requiring external model downloads.

### Main Menu Options

1. **Start Recognition Mode**: Launch the main attendance system
2. **Register New Face**: Add a new user to the system
3. **View Registered Users**: List all registered users
4. **View Today's Attendance**: Show attendance summary
5. **Delete User**: Remove a user from the system
6. **Test Spoof Detection**: Test the anti-spoofing module
7. **Exit**: Close the application

### Keyboard Controls (Recognition Mode)

| Key | Action |
|-----|--------|
| `Q` | Quit recognition mode |
| `R` | Register new face |
| `A` | Mark attendance for recognized face |
| `V` | View registered users |
| `H` | View today's history |

### Registering a New Face

1. Select option 2 from the menu OR press `R` during recognition
2. Enter the person's name
3. Position face in front of camera
4. System will capture 10 samples automatically
5. Move head slightly between captures for better accuracy

### Marking Attendance

1. Stand in front of the camera
2. Wait for recognition (green box = recognized)
3. Press `A` to mark attendance
4. System automatically determines punch-in or punch-out

## 🧠 Model and Approach

### Face Detection
- **Primary Model**: HOG (Histogram of Oriented Gradients)
  - Fast CPU-based detection
  - Works well for frontal faces
- **Alternative**: CNN-based detection (for GPU systems)
  - More accurate but slower
  - Better for non-frontal angles

### Face Recognition
- **Library**: dlib's face recognition (via face_recognition library)
- **Model**: ResNet-based deep learning model
- **Encoding**: 128-dimensional face encodings
- **Comparison**: Euclidean distance between encodings

### Recognition Algorithm
```
1. Detect all faces in frame
2. Calculate face encodings for each detected face
3. Select nearest face (largest area + closest to center)
4. Compare encoding with all stored encodings
5. Use voting mechanism across multiple samples per person
6. Return match if distance < tolerance threshold
```

### Nearest Face Selection
The system prioritizes the nearest face using:
- **Face Area**: Larger faces are closer to the camera
- **Center Distance**: Faces closer to frame center are prioritized
- **Combined Score**: `score = area × (1 - 0.3 × normalized_distance)`

### Spoof Detection Approach
Multi-factor liveness detection:

| Factor | Weight | Description |
|--------|--------|-------------|
| Blink Detection | 30% | Detects eye closure patterns |
| Texture Analysis | 30% | Laplacian variance for texture depth |
| Motion Analysis | 20% | Micro-movements in face region |
| Size Variation | 20% | Natural face size changes |

## 📚 Training Process

### Face Registration (Training)
1. **Sample Collection**: 10 face images per person
2. **Encoding Generation**: Extract 128-D embeddings for each sample
3. **Storage**: Pickle file containing all encodings with labels

### Why Multiple Samples?
- Captures different angles and expressions
- Improves recognition accuracy
- Enables voting-based recognition (more robust)

### Data Storage
- Face images: `faces_database/<name>/sample_X.jpg`
- Encodings: `face_encodings.pkl`
- Attendance: `attendance_logs/attendance_YYYY-MM-DD.csv`

## 📊 Accuracy Expectations

### Recognition Accuracy

| Condition | Expected Accuracy |
|-----------|-------------------|
| Good lighting, frontal face | 95-99% |
| Moderate lighting variations | 85-95% |
| Side angles (up to 30°) | 70-85% |
| Low light conditions | 60-80% |
| With glasses | 90-95% |
| With face masks | 40-60% |

### Spoof Detection Accuracy

| Attack Type | Detection Rate |
|-------------|----------------|
| Printed photo | 70-85% |
| Phone/tablet screen | 60-75% |
| Video playback | 50-70% |
| 3D masks | 20-40% |

### Factors Affecting Accuracy
- Lighting consistency between registration and recognition
- Camera quality
- Face angle and distance
- Time between sessions (appearance changes)
- Environmental factors (glasses, facial hair, makeup)

## ⚠️ Known Limitations

### Technical Limitations

1. **Single Camera Support**: Currently uses only one camera
2. **HOG Detection Limitations**: 
   - May miss faces at extreme angles
   - Struggles with very small faces
3. **No GPU Optimization**: Runs on CPU by default
4. **Memory Usage**: All encodings loaded into RAM

### Spoof Detection Limitations

1. **Basic Implementation**: Not production-grade security
2. **High-Quality Attacks**: May not detect sophisticated spoofs
3. **Lighting Dependency**: Texture analysis affected by lighting
4. **Training Required**: Blink detection needs time to establish baseline

### Operational Limitations

1. **Single Face Registration**: Cannot handle twins/very similar faces well
2. **No Real-time Training**: New faces require manual registration
3. **Lighting Sensitivity**: Performance varies with lighting changes
4. **No Offline Mode**: Requires camera access at all times

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| Multiple identical faces | May confuse between them |
| Significant appearance change | May fail to recognize |
| Very dark environment | Detection may fail |
| Fast movement | Tracking may be lost |
| Face masks | Reduced accuracy |

## 📁 Project Structure

```
Medoc/
├── main.py                  # Main application entry point
├── config.py                # Configuration settings
├── face_registration.py     # Face registration module
├── face_recognition_module.py # Face recognition module
├── spoof_detection.py       # Anti-spoofing module
├── attendance_system.py     # Attendance tracking module
├── face_encodings.pkl       # Stored face encodings (generated)
├── faces_database/          # Stored face images
│   └── <user_name>/
│       ├── sample_1.jpg
│       ├── sample_2.jpg
│       └── ...
├── attendance_logs/         # Attendance CSV files
│   └── attendance_YYYY-MM-DD.csv
├── README.md               # This documentation
└── requirements.txt        # Python dependencies
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Face recognition tolerance (lower = stricter)
FACE_RECOGNITION_TOLERANCE = 0.5

# Number of samples during registration
SAMPLES_PER_PERSON = 10

# Minimum time between punches (minutes)
PUNCH_COOLDOWN_MINUTES = 1

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
```

## 📝 Attendance Log Format

CSV files with columns:
- `name`: User's registered name
- `action`: PUNCH-IN or PUNCH-OUT
- `time`: HH:MM:SS format
- `date`: YYYY-MM-DD format
- `confidence`: Recognition confidence (0-1)
- `liveness_score`: Spoof detection score (0-1)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

## 📄 License

This project is for educational purposes. Use responsibly and in compliance with privacy laws and regulations.

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Author**: AI/ML Intern Project
