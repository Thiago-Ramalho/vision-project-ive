"""
Main application for camera calibration and image rectification.
This module provides the main entry point for the application.
"""

import cv2
import numpy as np
import os
from camera_capture import CameraCapture, select_camera_interactive, preview_camera
from image_processor import ImageProcessor
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
        self.points = []
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
            return True
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN and self.captured_image is not None:
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"Point {len(self.points)}: ({x}, {y})")
                
                # Draw the point on the image
                cv2.circle(self.captured_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.captured_image, str(len(self.points)), 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                if len(self.points) == 4:
                    print("All 4 points selected. Processing rectification...")
                    self.process_rectification()
    
    def process_rectification(self):
        """Process the rectification based on selected points."""
        if len(self.points) != 4:
            print("Need exactly 4 points for rectification")
            return
        
        # Convert points to numpy array
        src_points = np.array(self.points, dtype=np.float32)
        
        # Create the original image copy for rectification
        original_image = self.captured_image.copy()
        
        # Process rectification
        rectified = self.processor.rectify_image(original_image, src_points)
        
        if rectified is not None:
            # Display rectified image
            cv2.imshow('Rectified Image', rectified)
            
            # Save the rectified image
            save_image('rectified_image.jpg', rectified)
            print("Rectified image saved as 'rectified_image.jpg'")
            
            # Reset for next capture
            self.points = []
            self.captured_image = None
    
    def run(self):
        """Run the main application loop."""
        if not self.load_calibration():
            print("Cannot start application without calibration data.")
            return
        
        print("Camera Rectification Application")
        print("Controls:")
        print("  'c' - Capture image for rectification")
        print("  'q' - Quit")
        print("  Click 4 points on captured image to rectify")
        print("  Points order: top-left, top-right, bottom-right, bottom-left")
        
        # Set up window and mouse callback
        cv2.namedWindow('Camera Feed')
        cv2.setMouseCallback('Camera Feed', self.mouse_callback)
        
        try:
            while True:
                # Get frame from camera
                frame = self.camera.get_frame()
                if frame is None:
                    print("Failed to get frame from camera")
                    break
                
                # Apply distortion correction if calibration data is available
                if self.calibration_data is not None:
                    camera_matrix = self.calibration_data['camera_matrix']
                    dist_coeffs = self.calibration_data['dist_coeffs']
                    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
                
                # Display the frame
                display_frame = frame.copy()
                if self.captured_image is None:
                    cv2.putText(display_frame, "Press 'c' to capture", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                
                cv2.imshow('Camera Feed', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Capture image for rectification
                    self.captured_image = frame.copy()
                    self.points = []
                    cv2.imshow('Camera Feed', self.captured_image)
                    print("Image captured. Click 4 corner points in order:")
                    print("1. Top-left, 2. Top-right, 3. Bottom-right, 4. Bottom-left")
        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        
        finally:
            self.camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = RectificationApp()
    app.run()
