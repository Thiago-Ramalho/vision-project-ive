"""
Utility functions for the camera calibration and image rectification project.
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime


def save_calibration_data(calibration_data, filename='calibration_data.xml', format='xml'):
    """
    Save camera calibration data to a file.
    
    Args:
        calibration_data: Dictionary containing calibration results
        filename: Name of the file to save data to
        format: File format ('xml' or 'npz')
    """
    try:
        if format.lower() == 'xml' or filename.endswith('.xml'):
            save_calibration_data_xml(calibration_data, filename)
        else:
            # Fallback to NPZ format
            save_calibration_data_npz(calibration_data, filename)
        
        # Also save a human-readable summary
        base_name = os.path.splitext(filename)[0]
        save_calibration_summary(calibration_data, f"{base_name}_summary.txt")
        
    except Exception as e:
        print(f"Error saving calibration data: {e}")


def save_calibration_data_xml(calibration_data, filename='calibration_data.xml'):
    """
    Save camera calibration data to XML format using OpenCV FileStorage.
    
    Args:
        calibration_data: Dictionary containing calibration results
        filename: Name of the XML file to save data to
    """
    try:
        # Create FileStorage object for writing
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        
        # Add timestamp
        fs.write('timestamp', datetime.now().isoformat())
        
        # Write camera matrix
        if 'camera_matrix' in calibration_data:
            fs.write('camera_matrix', calibration_data['camera_matrix'])
        
        # Write distortion coefficients
        if 'dist_coeffs' in calibration_data:
            fs.write('distortion_coefficients', calibration_data['dist_coeffs'])
        
        # Write image size
        if 'image_size' in calibration_data:
            fs.write('image_width', int(calibration_data['image_size'][0]))
            fs.write('image_height', int(calibration_data['image_size'][1]))
        
        # Write reprojection error
        if 'mean_error' in calibration_data:
            fs.write('mean_reprojection_error', float(calibration_data['mean_error']))
        
        # Write number of calibration images
        if 'rvecs' in calibration_data:
            fs.write('num_calibration_images', len(calibration_data['rvecs']))
        
        # Write rotation vectors (optional, for advanced use)
        if 'rvecs' in calibration_data:
            for i, rvec in enumerate(calibration_data['rvecs']):
                fs.write(f'rotation_vector_{i}', rvec)
        
        # Write translation vectors (optional, for advanced use)
        if 'tvecs' in calibration_data:
            for i, tvec in enumerate(calibration_data['tvecs']):
                fs.write(f'translation_vector_{i}', tvec)
        
        # Release the file
        fs.release()
        
        print(f"Calibration data saved to XML: {filename}")
        
    except Exception as e:
        print(f"Error saving calibration data to XML: {e}")


def save_calibration_data_npz(calibration_data, filename='calibration_data.npz'):
    """
    Save camera calibration data to NPZ format (original method).
    
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
        print(f"Calibration data saved to NPZ: {filename}")
        
    except Exception as e:
        print(f"Error saving calibration data to NPZ: {e}")


def load_calibration_data(filename='calibration_data.xml'):
    """
    Load camera calibration data from a file (XML or NPZ format).
    
    Args:
        filename: Name of the file to load data from
        
    Returns:
        dict: Calibration data or None if loading failed
    """
    try:
        if not os.path.exists(filename):
            print(f"Calibration file {filename} not found")
            return None
        
        if filename.endswith('.xml'):
            return load_calibration_data_xml(filename)
        else:
            return load_calibration_data_npz(filename)
        
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return None


def load_calibration_data_xml(filename='calibration_data.xml'):
    """
    Load camera calibration data from XML format.
    
    Args:
        filename: Name of the XML file to load data from
        
    Returns:
        dict: Calibration data or None if loading failed
    """
    try:
        # Create FileStorage object for reading
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        
        if not fs.isOpened():
            print(f"Could not open XML file: {filename}")
            return None
        
        calibration_data = {}
        
        # Read camera matrix
        camera_matrix_node = fs.getNode('camera_matrix')
        if not camera_matrix_node.empty():
            calibration_data['camera_matrix'] = camera_matrix_node.mat()
        
        # Read distortion coefficients
        dist_coeffs_node = fs.getNode('distortion_coefficients')
        if not dist_coeffs_node.empty():
            calibration_data['dist_coeffs'] = dist_coeffs_node.mat()
        
        # Read image size
        width_node = fs.getNode('image_width')
        height_node = fs.getNode('image_height')
        if not width_node.empty() and not height_node.empty():
            calibration_data['image_size'] = (int(width_node.real()), int(height_node.real()))
        
        # Read reprojection error
        error_node = fs.getNode('mean_reprojection_error')
        if not error_node.empty():
            calibration_data['mean_error'] = error_node.real()
        
        # Read timestamp
        timestamp_node = fs.getNode('timestamp')
        if not timestamp_node.empty():
            calibration_data['timestamp'] = timestamp_node.string()
        
        # Read number of calibration images
        num_images_node = fs.getNode('num_calibration_images')
        if not num_images_node.empty():
            num_images = int(num_images_node.real())
            
            # Read rotation and translation vectors
            rvecs = []
            tvecs = []
            for i in range(num_images):
                rvec_node = fs.getNode(f'rotation_vector_{i}')
                tvec_node = fs.getNode(f'translation_vector_{i}')
                
                if not rvec_node.empty():
                    rvecs.append(rvec_node.mat())
                if not tvec_node.empty():
                    tvecs.append(tvec_node.mat())
            
            if rvecs:
                calibration_data['rvecs'] = rvecs
            if tvecs:
                calibration_data['tvecs'] = tvecs
        
        # Release the file
        fs.release()
        
        print(f"Calibration data loaded from XML: {filename}")
        return calibration_data
        
    except Exception as e:
        print(f"Error loading calibration data from XML: {e}")
        return None


def load_calibration_data_npz(filename='calibration_data.npz'):
    """
    Load camera calibration data from NPZ format (original method).
    
    Args:
        filename: Name of the NPZ file to load data from
        
    Returns:
        dict: Calibration data or None if loading failed
    """
    try:
        # Load data
        data = np.load(filename)
        
        # Convert to dictionary
        calibration_data = {}
        for key in data.files:
            calibration_data[key] = data[key]
        
        print(f"Calibration data loaded from NPZ: {filename}")
        return calibration_data
        
    except Exception as e:
        print(f"Error loading calibration data from NPZ: {e}")
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
