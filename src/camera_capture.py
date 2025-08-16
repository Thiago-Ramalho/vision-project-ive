"""
Camera capture module for handling camera input and video capture.
"""

import cv2
import numpy as np


class CameraCapture:
    def __init__(self, camera_index=0):
        """
        Initialize camera capture.
        
        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_initialized = False
        self.initialize_camera()
    
    def initialize_camera(self):
        """Initialize the camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties for better quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_initialized = True
            print(f"Camera {self.camera_index} initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def get_frame(self):
        """
        Capture a frame from the camera.
        
        Returns:
            numpy.ndarray: The captured frame, or None if capture failed
        """
        if not self.is_initialized or self.cap is None:
            print("Camera not initialized")
            return None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to capture frame")
                return None
            
            return frame
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def get_camera_properties(self):
        """
        Get current camera properties.
        
        Returns:
            dict: Dictionary containing camera properties
        """
        if not self.is_initialized or self.cap is None:
            return None
        
        properties = {
            'width': self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
        }
        
        return properties
    
    def set_camera_property(self, property_id, value):
        """
        Set a camera property.
        
        Args:
            property_id: OpenCV property ID (e.g., cv2.CAP_PROP_BRIGHTNESS)
            value: Value to set
            
        Returns:
            bool: True if property was set successfully
        """
        if not self.is_initialized or self.cap is None:
            return False
        
        try:
            return self.cap.set(property_id, value)
        except Exception as e:
            print(f"Error setting camera property: {e}")
            return False
    
    def capture_image(self, filename=None):
        """
        Capture and optionally save an image.
        
        Args:
            filename: If provided, save the image to this file
            
        Returns:
            numpy.ndarray: The captured image, or None if capture failed
        """
        frame = self.get_frame()
        
        if frame is not None and filename is not None:
            try:
                cv2.imwrite(filename, frame)
                print(f"Image saved as {filename}")
            except Exception as e:
                print(f"Error saving image: {e}")
        
        return frame
    
    def is_camera_available(self):
        """
        Check if the camera is available and working.
        
        Returns:
            bool: True if camera is available
        """
        return self.is_initialized and self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.is_initialized = False
            print("Camera released")
    
    def __del__(self):
        """Destructor to ensure camera is released."""
        self.release()


def test_camera():
    """Test function to verify camera functionality."""
    print("Testing camera capture...")
    
    camera = CameraCapture()
    
    if not camera.is_camera_available():
        print("Camera is not available")
        return
    
    print("Camera properties:", camera.get_camera_properties())
    
    print("Press 'q' to quit, 's' to save image")
    
    try:
        while True:
            frame = camera.get_frame()
            
            if frame is None:
                print("Failed to get frame")
                break
            
            cv2.imshow('Camera Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                camera.capture_image('test_capture.jpg')
    
    except KeyboardInterrupt:
        print("Test interrupted")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()
