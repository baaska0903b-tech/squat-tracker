# Squat Tracker

A real-time squat tracking application built with Python, OpenCV, and MediaPipe Pose Landmarker.

The application uses a webcam to detect body landmarks, estimate the knee angle, count squat repetitions, and save exercise measurements to a CSV file.

## Features

- Real-time webcam video
- Human pose and skeleton landmark detection
- Knee angle estimation
- Target angle detection
- Automatic squat repetition counting
- CSV logging of exercise measurements

## Technologies

- Python 3
- OpenCV
- MediaPipe
- NumPy

## Project Structure

```text
body_tracking/
├── main.py
├── pose_landmarker_lite.task
├── README.md
├── .gitignore
└── squat_results.csv
```

The virtual environment folder (`.venv/`) and generated CSV results are not committed to GitHub.

## Requirements

- A MacBook with a working webcam
- Python 3
- Internet connection for downloading the MediaPipe model
- macOS camera permission enabled for Terminal or your code editor

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Replace the example values with your actual GitHub repository URL and folder name.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the required packages

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python mediapipe numpy
```

### 4. Download the MediaPipe model

Run this command inside the project folder:

```bash
curl -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" -o pose_landmarker_lite.task
```

The model file must be in the same folder as `main.py`.

## How to Run

Activate the virtual environment and run:

```bash
source .venv/bin/activate
python main.py
```

Allow camera access if macOS asks for permission.

Stand sideways to the camera and ensure your hip, knee, and ankle are visible. Press `q` in the camera window to stop the application.

## How It Works

1. OpenCV reads video frames from the webcam.
2. MediaPipe detects body landmarks.
3. The application estimates the knee angle using the hip, knee, and ankle landmarks.
4. When the angle reaches the configured target threshold, the target position is recorded.
5. When the user returns to a standing position, the repetition counter increases.

### Configuration

The current thresholds are configured in `main.py`:

```python
TARGET_ANGLE = 95
STANDING_ANGLE = 160
VISIBILITY_LIMIT = 0.60
```

These are experimental thresholds and may require adjustment based on camera placement and pose detection accuracy. The measured angle is a 2D estimate, not a medical assessment.

## Output and Results

After running the program, the results are saved in:

```text
squat_results.csv
```

The file is created in the current working directory from which `python main.py` is run. If you run the program from the project folder, the CSV will be saved there.

The CSV contains:

| Column | Description |
|---|---|
| `timestamp` | Date and time of the measurement |
| `elapsed_seconds` | Time since the application started |
| `knee_angle` | Estimated knee angle in degrees |
| `target_reached` | Whether the target-angle state is active |
| `repetitions` | Current squat repetition count |
| `status` | Current exercise status |

The file can be opened in Microsoft Excel or imported into Python for analysis.

**Note:** Measurements are recorded for each processed frame where a valid knee angle is available. The repetition count is updated when a squat reaches the target angle and the user returns to the standing threshold.

## Troubleshooting

**Camera does not open**

- Check macOS System Settings → Privacy & Security → Camera.
- Allow access for Terminal or the code editor.
- Close other applications that may be using the webcam.

**Model file not found**

- Confirm that `pose_landmarker_lite.task` is in the same folder as `main.py`.
- Download the model using the command above.

**Python module not found**

Activate the virtual environment and reinstall the dependencies.

**Body landmarks are unstable**

- Position the camera to the side.
- Keep the full body inside the frame.
- Improve lighting and avoid blocking the hip, knee, and ankle.

## Limitations

- Knee angle is estimated from a 2D camera image.
- Camera position and landmark visibility affect accuracy.
- A target angle is an experimental threshold, not a universal exercise or medical recommendation.
- The application is a prototype and does not guarantee correct exercise technique or injury prevention.

## License

Add a license file if you intend to distribute this project publicly.
