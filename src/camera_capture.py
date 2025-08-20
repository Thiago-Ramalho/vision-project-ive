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


def list_available_cameras(max_cameras=10):
    """
    List all available cameras connected to the system.
    
    Args:
        max_cameras: Maximum number of camera indices to check
        
    Returns:
        list: List of available camera indices
    """
    available_cameras = []
    
    print("Scanning for available cameras...")
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to confirm the camera works
            ret, frame = cap.read()
            if ret and frame is not None:
                # Get camera properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                available_cameras.append({
                    'index': i,
                    'width': int(width),
                    'height': int(height),
                    'fps': fps
                })
                
                print(f"Camera {i}: {int(width)}x{int(height)} @ {fps:.1f} FPS")
            
            cap.release()
    
    if not available_cameras:
        print("No cameras found!")
    else:
        print(f"Found {len(available_cameras)} camera(s)")
    
    return available_cameras


def select_camera_interactive():
    """
    Interactive camera selection.
    
    Returns:
        int: Selected camera index, or 0 if no selection made
    """
    cameras = list_available_cameras()
    
    if not cameras:
        print("No cameras available. Using default camera index 0.")
        return 0
    
    if len(cameras) == 1:
        print(f"Only one camera found. Using camera {cameras[0]['index']}")
        return cameras[0]['index']
    
    print("\nAvailable cameras:")
    for i, cam in enumerate(cameras):
        print(f"  {i}: Camera {cam['index']} - {cam['width']}x{cam['height']} @ {cam['fps']:.1f} FPS")
    
    while True:
        try:
            choice = input(f"\nSelect camera (0-{len(cameras)-1}) or press Enter for camera 0: ").strip()
            
            if choice == "":
                return cameras[0]['index']
            
            choice_idx = int(choice)
            if 0 <= choice_idx < len(cameras):
                selected_camera = cameras[choice_idx]['index']
                print(f"Selected camera {selected_camera}")
                return selected_camera
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(cameras)-1}")
                
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nUsing default camera 0")
            return 0


def preview_camera(camera_index):
    """
    Preview a specific camera to verify it's working.
    
    Args:
        camera_index: Index of the camera to preview
    """
    print(f"Previewing camera {camera_index}...")
    print("Press 'q' to close preview")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame")
                break
            
            # Add camera info overlay
            cv2.putText(frame, f"Camera {camera_index}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to close", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(f'Camera {camera_index} Preview', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Preview interrupted")
    
    finally:
        cap.release()
        cv2.destroyWindow(f'Camera {camera_index} Preview')


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
