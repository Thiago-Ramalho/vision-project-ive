## Updated Camera Selection Guide

I've enhanced your vision project to support multiple cameras! Here's what I added:

### New Features:

1. **Camera Detection**: Automatically scans for all available cameras
2. **Interactive Selection**: Choose which camera to use
3. **Camera Preview**: Test cameras before calibration/rectification
4. **Cross-platform XML**: Calibration data now saves in XML format

### How to Use:

#### 1. **Detect Available Cameras:**
```bash
python3 detect_cameras.py
```
This script will:
- Scan for all connected cameras
- Show their resolution and FPS
- Let you preview each camera
- Help you identify which camera is your USB camera

#### 2. **Run Calibration with Camera Selection:**
```bash
python3 src/calibration.py
```
Now the calibration script will:
- Automatically detect available cameras
- Ask you to choose which one to use
- Optionally preview the selected camera
- Save calibration data as `calibration_data.xml`

#### 3. **Run Main Application with Camera Selection:**
```bash
python3 src/main.py
```
The main app will also let you select which camera to use.

### What Changed:

1. **camera_capture.py**: Added camera detection functions
2. **calibration.py**: Added camera selection on startup
3. **main.py**: Added camera selection on startup
4. **utils.py**: Added XML format support for calibration data
5. **detect_cameras.py**: New utility script for camera testing

### Typical Workflow:

1. Connect your USB camera
2. Run `python3 detect_cameras.py` to see all cameras
3. Note the index of your USB camera (usually 1 or 2)
4. Run calibration: `python3 src/calibration.py`
5. Select your USB camera when prompted
6. Proceed with calibration as normal

The calibration data will now be saved as `calibration_data.xml` which is cross-platform compatible and can be used by other OpenCV applications written in C++, Python, or other languages.
