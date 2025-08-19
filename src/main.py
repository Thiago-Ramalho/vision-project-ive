"""
Real-time camera calibration and image rectification application.
Features live point selection and real-time rectified video output.
"""

import cv2
import numpy as np
import os
from camera_capture import CameraCapture, select_camera_interactive, preview_camera
from point_selector import RealtimePointSelector
from realtime_rectifier import RealtimeRectifier
from utils import load_calibration_data


class RectificationApp:
    def __init__(self, camera_index=None):
        """Initialize the real-time rectification application."""
        # Select camera
        if camera_index is None:
            camera_index = select_camera_interactive()
        
        # Ask if user wants to preview the camera
        preview_choice = input(f"\nDo you want to preview camera {camera_index} before starting? (y/N): ").strip().lower()
        if preview_choice in ['y', 'yes']:
            preview_camera(camera_index)
        
        # Initialize components
        self.camera = CameraCapture(camera_index)
        self.calibration_data = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.rectifier = None
        
        # Get frame dimensions
        test_frame = self.camera.get_frame()
        if test_frame is not None:
            self.frame_height, self.frame_width = test_frame.shape[:2]
            print(f"Camera resolution: {self.frame_width}x{self.frame_height}")
        else:
            raise RuntimeError("Cannot get frame from camera")
    
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
            
            print("Calibration data loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return False
    
    def setup_rectification(self):
        """Setup real-time rectification with point selection."""
        print("\n=== Setting up Real-time Rectification ===")
        print("You will now select 4 corner points on the live video feed.")
        print("These points will define the area to be rectified.")
        
        # Initialize rectifier
        self.rectifier = RealtimeRectifier(self.frame_width, self.frame_height)
        
        # Create point selector with live camera feed
        point_selector = RealtimePointSelector(self.camera, "Select 4 Corner Points - Live Video")
        
        # Get points from user
        success, points = point_selector.select_points()
        
        if not success or len(points) != 4:
            print("Point selection failed or cancelled")
            return False
        
        print("Points selected:")
        for i, point in enumerate(points):
            print(f"  Point {i+1}: {point}")
        
        # Ask for output dimensions
        print("\nOutput dimensions for rectified image:")
        try:
            width = input(f"Width (default: auto-calculate): ").strip()
            height = input(f"Height (default: auto-calculate): ").strip()
            
            output_width = int(width) if width else None
            output_height = int(height) if height else None
        except ValueError:
            output_width = output_height = None
        
        # Compute transformation
        try:
            self.rectifier.compute_transformation(points, output_width, output_height)
            
            # Save transformation to XML
            transform_file = "rectification_transform.xml"
            self.rectifier.save_transformation_xml(transform_file)
            print(f"\nTransformation saved to {transform_file}")
            
            return True
        except Exception as e:
            print(f"Error computing transformation: {e}")
            return False
    
    def load_existing_transformation(self):
        """Load existing rectification transformation."""
        transform_file = "rectification_transform.xml"
        if os.path.exists(transform_file):
            self.rectifier = RealtimeRectifier(self.frame_width, self.frame_height)
            if self.rectifier.load_transformation_xml(transform_file):
                print(f"Loaded existing transformation from {transform_file}")
                return True
        return False
    
    def run_realtime_rectification(self):
        """Run the real-time rectification loop."""
        if not self.rectifier or not self.rectifier.is_initialized:
            print("Rectifier not initialized")
            return
        
        print("\n=== Real-time Rectification Started ===")
        print("Press 'q' to quit, 's' to save current rectified frame")
        print("Press 'r' to reconfigure rectification points")
        
        # Create windows
        cv2.namedWindow('Original (Distortion Corrected)', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Rectified (Real-time)', cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        try:
            while True:
                # Get frame
                frame = self.camera.get_frame()
                if frame is None:
                    print("Failed to get frame from camera")
                    break
                
                # Apply distortion correction
                corrected_frame = cv2.undistort(frame, self.camera_matrix, self.distortion_coeffs)
                
                # Apply real-time rectification
                rectified_frame = self.rectifier.rectify_frame(corrected_frame)
                
                # Display both frames
                cv2.imshow('Original (Distortion Corrected)', corrected_frame)
                cv2.imshow('Rectified (Real-time)', rectified_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current rectified frame
                    filename = f"rectified_realtime_{frame_count:04d}.jpg"
                    cv2.imwrite(filename, rectified_frame)
                    print(f"Saved frame: {filename}")
                elif key == ord('r'):
                    # Reconfigure rectification
                    cv2.destroyWindow('Original (Distortion Corrected)')
                    cv2.destroyWindow('Rectified (Real-time)')
                    if self.setup_rectification():
                        cv2.namedWindow('Original (Distortion Corrected)', cv2.WINDOW_AUTOSIZE)
                        cv2.namedWindow('Rectified (Real-time)', cv2.WINDOW_AUTOSIZE)
                    else:
                        break
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cv2.destroyAllWindows()
    
    def run(self):
        """Main application loop."""
        print("=== Real-time Camera Rectification Application ===")
        
        # Load calibration data
        if not self.load_calibration():
            print("Cannot proceed without calibration data")
            return
        
        # Check for existing transformation
        use_existing = False
        if os.path.exists("rectification_transform.xml"):
            choice = input("\nFound existing rectification transformation. Use it? (y/N): ").strip().lower()
            if choice in ['y', 'yes']:
                use_existing = self.load_existing_transformation()
        
        # Setup rectification if needed
        if not use_existing:
            if not self.setup_rectification():
                print("Rectification setup failed")
                return
        
        # Run real-time rectification
        self.run_realtime_rectification()
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("\nApplication closed")


def main():
    """Main entry point."""
    try:
        # Set Qt platform for Linux compatibility
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        
        app = RectificationApp()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")


if __name__ == "__main__":
    main()
