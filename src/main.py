"""
Main application for camera calibration and image rectification.
This module provides the main entry point for the application.
"""

import cv2
import numpy as np
import os

# Fix Qt platform plugin issue on Linux
os.environ['QT_QPA_PLATFORM'] = 'xcb'

from camera_capture import CameraCapture, select_camera_interactive, preview_camera
from image_processor import ImageProcessor
from point_selector import PointSelector
from utils import load_calibration_data, save_image


class RectificationApp:
    def __init__(self, camera_index=None):
        # Select camera
        if camera_index is None:
            camera_index = select_camera_interactive()
        
        # Ask if user wants to preview the camera
        preview_choice = input(f"\nDo you want to preview camera {camera_index} before starting? (y/N): ").strip().lower()
        if preview_choice in ['y', 'yes']:
            preview_camera(camera_index)
        
        self.camera = CameraCapture(camera_index)
        self.processor = ImageProcessor()
        self.captured_image = None
        self.calibration_data = None
        
    def load_calibration(self):
        """Load camera calibration data."""
        try:
            # Try to load XML format first (preferred), then fallback to NPZ
            if os.path.exists('calibration_data.xml'):
                self.calibration_data = load_calibration_data('calibration_data.xml')
            elif os.path.exists('calibration_data.npz'):
                self.calibration_data = load_calibration_data('calibration_data.npz')
            else:
                self.calibration_data = None
            
            if self.calibration_data is None:
                print("No calibration data found. Please run calibration.py first.")
                return False
            
            # Extract camera matrix and distortion coefficients
            self.camera_matrix = self.calibration_data['camera_matrix']
            self.distortion_coeffs = self.calibration_data['dist_coeffs']
            
            return True
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return False
    
    def select_points_interactive(self, image):
        """Use the interactive point selector interface."""
        print("Opening interactive point selection interface...")
        
        selector = PointSelector(image, "Select 4 Corner Points")
        success, points = selector.select_points()
        
        if success and points:
            print("Points selected successfully:")
            for i, point in enumerate(points):
                print(f"  Point {i+1}: {point}")
            return list(points)
        else:
            print("Point selection cancelled or failed")
            return None
    
    def process_rectification(self, image, points):
        """Process the rectification based on selected points."""
        if len(points) != 4:
            print("Need exactly 4 points for rectification")
            return
        
        # Convert points to numpy array
        src_points = np.array(points, dtype=np.float32)
        
        # Process rectification
        rectified = self.processor.rectify_image(image, src_points)
        
        if rectified is not None:
            # Display rectified image
            cv2.imshow('Rectified Image', rectified)
            
            # Save the rectified image
            save_image('rectified_image.jpg', rectified)
            print("Rectified image saved as 'rectified_image.jpg'")
            
            # Reset for next capture
            self.captured_image = None
    
    def run(self):
        """Main application loop."""
        # Load calibration data
        if not self.load_calibration():
            print("Cannot proceed without calibration data")
            return

        print("Camera started. Press 'c' to capture image for rectification, 'q' to quit")
        
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                print("Failed to read from camera")
                break
            
            # Apply distortion correction
            corrected_frame = cv2.undistort(frame, self.camera_matrix, 
                                          self.distortion_coeffs)
            
            cv2.imshow('Camera (Distortion Corrected)', corrected_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Capture image and start point selection
                self.captured_image = corrected_frame.copy()
                
                # Use interactive point selector
                points = self.select_points_interactive(self.captured_image)
                
                if points:
                    # Process rectification
                    self.process_rectification(self.captured_image, points)
                
            elif key == ord('q'):
                break
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        # Try to test OpenCV GUI functionality
        test_window = "OpenCV Test"
        cv2.namedWindow(test_window, cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow(test_window)
        
        print("OpenCV GUI test passed. Starting application...")
        app = RectificationApp()
        app.run()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        print("\nTroubleshooting:")
        print("1. Try running: export QT_QPA_PLATFORM=xcb")
        print("2. Or try: export QT_QPA_PLATFORM=x11") 
        print("3. Or use the provided run_main.sh script")
        print("4. Make sure you have a display available (not running headless)")
        
        # Try alternative: run without GUI for testing
        print("\nAlternatively, you can run individual modules:")
        print("- python src/calibration.py (for camera calibration)")
        print("- python src/detect_cameras.py (to list cameras)")
