"""
Utility functions for the camera calibration and image rectification project.
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime


def save_calibration_data(calibration_data, filename='calibration_data.npz'):
    """
    Save camera calibration data to a file.
    
    Args:
        calibration_data: Dictionary containing calibration results
        filename: Name of the file to save data to
    """
    try:
        # Convert any lists to numpy arrays for consistency
        data_to_save = {}
        for key, value in calibration_data.items():
            if isinstance(value, list):
                data_to_save[key] = np.array(value)
            else:
                data_to_save[key] = value
        
        # Add timestamp
        data_to_save['timestamp'] = datetime.now().isoformat()
        
        # Save to compressed numpy format
        np.savez_compressed(filename, **data_to_save)
        print(f"Calibration data saved to {filename}")
        
        # Also save a human-readable summary
        save_calibration_summary(calibration_data, filename.replace('.npz', '_summary.txt'))
        
    except Exception as e:
        print(f"Error saving calibration data: {e}")


def load_calibration_data(filename='calibration_data.npz'):
    """
    Load camera calibration data from a file.
    
    Args:
        filename: Name of the file to load data from
        
    Returns:
        dict: Calibration data or None if loading failed
    """
    try:
        if not os.path.exists(filename):
            print(f"Calibration file {filename} not found")
            return None
        
        # Load data
        data = np.load(filename)
        
        # Convert to dictionary
        calibration_data = {}
        for key in data.files:
            calibration_data[key] = data[key]
        
        print(f"Calibration data loaded from {filename}")
        return calibration_data
        
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return None


def save_calibration_summary(calibration_data, filename='calibration_summary.txt'):
    """
    Save a human-readable summary of calibration data.
    
    Args:
        calibration_data: Dictionary containing calibration results
        filename: Name of the summary file
    """
    try:
        with open(filename, 'w') as f:
            f.write("Camera Calibration Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if 'camera_matrix' in calibration_data:
                f.write("Camera Matrix:\n")
                f.write(str(calibration_data['camera_matrix']) + "\n\n")
            
            if 'dist_coeffs' in calibration_data:
                f.write("Distortion Coefficients:\n")
                f.write(str(calibration_data['dist_coeffs']) + "\n\n")
            
            if 'mean_error' in calibration_data:
                f.write(f"Mean Reprojection Error: {calibration_data['mean_error']:.6f}\n\n")
            
            if 'image_size' in calibration_data:
                f.write(f"Calibration Image Size: {calibration_data['image_size']}\n\n")
            
            f.write(f"Number of calibration images: {len(calibration_data.get('rvecs', []))}\n")
        
        print(f"Calibration summary saved to {filename}")
        
    except Exception as e:
        print(f"Error saving calibration summary: {e}")


def save_image(filename, image, quality=95):
    """
    Save an image with optional quality settings.
    
    Args:
        filename: Name of the file to save
        image: Image array to save
        quality: JPEG quality (0-100)
    """
    try:
        # Determine file extension
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in ['.jpg', '.jpeg']:
            # Set JPEG quality
            cv2.imwrite(filename, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == '.png':
            # Set PNG compression
            cv2.imwrite(filename, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            # Default save
            cv2.imwrite(filename, image)
        
        print(f"Image saved: {filename}")
        
    except Exception as e:
        print(f"Error saving image {filename}: {e}")


def load_image(filename):
    """
    Load an image from file.
    
    Args:
        filename: Name of the file to load
        
    Returns:
        numpy.ndarray: Loaded image or None if loading failed
    """
    try:
        if not os.path.exists(filename):
            print(f"Image file {filename} not found")
            return None
        
        image = cv2.imread(filename)
        
        if image is None:
            print(f"Failed to load image: {filename}")
            return None
        
        return image
        
    except Exception as e:
        print(f"Error loading image {filename}: {e}")
        return None


def create_output_directory(directory='output'):
    """
    Create an output directory if it doesn't exist.
    
    Args:
        directory: Name of the directory to create
        
    Returns:
        str: Path to the created directory
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return directory
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        return None


def get_timestamp_string():
    """
    Get a timestamp string for file naming.
    
    Returns:
        str: Timestamp string in YYYYMMDD_HHMMSS format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_points(points, image_shape):
    """
    Validate that points are within image boundaries.
    
    Args:
        points: List or array of (x, y) points
        image_shape: Shape of the image (height, width, channels)
        
    Returns:
        bool: True if all points are valid
    """
    try:
        if len(points) != 4:
            return False
        
        height, width = image_shape[:2]
        
        for point in points:
            x, y = point
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
        
        return True
        
    except Exception:
        return False


def calculate_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        float: Distance between points
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def resize_image_maintain_aspect(image, max_width=800, max_height=600):
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        numpy.ndarray: Resized image
    """
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized


def draw_text_with_background(image, text, position, font_scale=0.7, 
                            text_color=(255, 255, 255), bg_color=(0, 0, 0),
                            thickness=2, padding=5):
    """
    Draw text with a background rectangle.
    
    Args:
        image: Image to draw on
        text: Text to draw
        position: (x, y) position for the text
        font_scale: Size of the font
        text_color: Color of the text (BGR)
        bg_color: Color of the background (BGR)
        thickness: Text thickness
        padding: Padding around text
        
    Returns:
        numpy.ndarray: Image with text drawn
    """
    result = image.copy()
    
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate background rectangle coordinates
    x, y = position
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + baseline + padding
    
    # Draw background rectangle
    cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    # Draw text
    cv2.putText(result, text, position, font, font_scale, text_color, thickness)
    
    return result


def log_message(message, log_file='application.log'):
    """
    Log a message to file with timestamp.
    
    Args:
        message: Message to log
        log_file: Log file name
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
    except Exception as e:
        print(f"Error writing to log: {e}")


def check_opencv_version():
    """
    Check OpenCV version and print information.
    
    Returns:
        str: OpenCV version string
    """
    version = cv2.__version__
    print(f"OpenCV version: {version}")
    
    # Check for required features
    print("Available capture backends:")
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_FFMPEG, "FFmpeg"),
    ]
    
    for backend_id, backend_name in backends:
        try:
            cap = cv2.VideoCapture(backend_id)
            if cap.isOpened():
                print(f"  - {backend_name}: Available")
                cap.release()
            else:
                print(f"  - {backend_name}: Not available")
        except:
            print(f"  - {backend_name}: Error checking")
    
    return version


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test OpenCV version check
    check_opencv_version()
    
    # Test timestamp generation
    print(f"Timestamp: {get_timestamp_string()}")
    
    # Test directory creation
    test_dir = create_output_directory('test_output')
    if test_dir:
        print(f"Created directory: {test_dir}")
    
    print("Utility functions test completed!")
