# Camera Calibration and Image Rectification Project

This project implements a camera calibration and image rectification system using OpenCV. The application captures images from your camera, removes radial distortion, and allows you to perform perspective correction by selecting four corner points.

## Features

- Real-time camera capture
- Radial distortion correction using camera calibration
- Interactive point selection for perspective transformation
- Image rectification based on selected points
- Save calibration data and rectified images

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone this repository or download the project files
2. Navigate to the project directory
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Camera Calibration

First, you need to calibrate your camera to remove radial distortion:

```bash
python src/calibration.py
```

This will:
- Capture multiple images using your camera
- Detect chessboard patterns for calibration
- Calculate camera matrix and distortion coefficients
- Save calibration data to `calibration_data.xml` (cross-platform format)

### 2. Image Rectification

Once calibration is complete, run the main application:

```bash
python src/main.py
```

This will:
- Load your camera calibration data
- Start the camera feed with distortion correction applied
- Allow you to capture an image and select 4 corner points
- Generate a rectified (perspective-corrected) image
- Save the result

### 3. Testing Calibration Quality

To verify your calibration data is working correctly:

```bash
python test_calibration.py
```

This comprehensive testing tool will:
- Validate calibration file integrity
- Show before/after distortion correction comparison
- Analyze distortion patterns and quality metrics
- Provide real-time correction preview
- Give quality assessment and recommendations

## Project Structure

```
src/
├── main.py              # Main application entry point
├── calibration.py       # Camera calibration module
├── camera_capture.py    # Camera handling and image capture
├── image_processor.py   # Image processing and rectification
└── utils.py            # Utility functions

detect_cameras.py        # Camera detection and testing utility
test_calibration.py      # Calibration quality testing tool
```

## How to Use

1. **Camera Detection**: First, identify available cameras:
   ```bash
   python detect_cameras.py
   ```

2. **Calibration Phase**: Run the calibration script and capture 10-15 images of a chessboard pattern from different angles and distances:
   ```bash
   python src/calibration.py
   ```

3. **Test Calibration**: Verify your calibration quality:
   ```bash
   python test_calibration.py
   ```

4. **Rectification Phase**: 
   - Run the main application
   - Position your camera to capture the document/surface you want to rectify
   - Press 'c' to capture an image
   - Click on the four corners of the area you want to rectify (in order: top-left, top-right, bottom-right, bottom-left)
   - The rectified image will be displayed and saved

## Controls

- **Space**: Capture image for calibration
- **'c'**: Capture image for rectification
- **'q'**: Quit application
- **Mouse Click**: Select corner points for rectification

## Output Files

- `calibration_data.xml`: Camera calibration parameters (OpenCV XML format - cross-platform)
- `calibration_data_summary.txt`: Human-readable calibration summary
- `rectified_image.jpg`: Final rectified image
- `captured_images/`: Directory containing captured calibration images

## Notes

- Ensure good lighting conditions for better results
- Use a printed chessboard pattern (9x6 or 8x6 internal corners work well)
- For best rectification results, ensure the four selected points form a quadrilateral
- The application assumes the target rectangle should be oriented upright in the final image
- Run `test_calibration.py` to verify calibration quality before using for rectification
- Calibration quality indicators:
  - **Excellent**: < 0.5 pixels reprojection error
  - **Good**: 0.5 - 1.0 pixels reprojection error  
  - **Acceptable**: 1.0 - 2.0 pixels reprojection error
  - **Poor**: > 2.0 pixels reprojection error (recalibrate recommended)
