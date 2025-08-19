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
        self.mouse_action = None  # For handling mouse clicks
        
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
        
        # Use input video resolution as default output resolution
        # But the rectifier will calculate optimal size respecting field proportions
        field_aspect_ratio = 170.0 / 130.0  # Mini soccer field: 170cm x 130cm
        
        # Compute transformation
        try:
            self.rectifier.compute_transformation(points, field_aspect_ratio=field_aspect_ratio)
            
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
    
    def draw_control_buttons(self, frame, show_rectified, has_rectification):
        """Draw control buttons on the frame."""
        height, width = frame.shape[:2]
        button_width = 120
        button_height = 40
        margin = 10
        
        # Button positions (bottom of frame)
        y_base = height - button_height - margin
        
        # Switch view button
        switch_x = margin
        switch_color = (0, 150, 0) if has_rectification else (100, 100, 100)
        switch_text = "Normal View" if show_rectified else "Rectified View"
        
        cv2.rectangle(frame, (switch_x, y_base), (switch_x + button_width, y_base + button_height), switch_color, -1)
        cv2.rectangle(frame, (switch_x, y_base), (switch_x + button_width, y_base + button_height), (255, 255, 255), 2)
        
        # Center text
        text_size = cv2.getTextSize(switch_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = switch_x + (button_width - text_size[0]) // 2
        text_y = y_base + (button_height + text_size[1]) // 2
        cv2.putText(frame, switch_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Configure points button
        config_x = switch_x + button_width + margin
        config_color = (150, 100, 0)
        config_text = "Configure Points"
        
        cv2.rectangle(frame, (config_x, y_base), (config_x + button_width, y_base + button_height), config_color, -1)
        cv2.rectangle(frame, (config_x, y_base), (config_x + button_width, y_base + button_height), (255, 255, 255), 2)
        
        text_size = cv2.getTextSize(config_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = config_x + (button_width - text_size[0]) // 2
        text_y = y_base + (button_height + text_size[1]) // 2
        cv2.putText(frame, config_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Status text
        status_y = y_base - 10
        status_text = f"Mode: {'Rectified' if show_rectified else 'Normal'}"
        if not has_rectification:
            status_text += " - No rectification configured"
        else:
            status_text += " - Soccer field (170×130cm)"
        
        cv2.putText(frame, status_text, (margin, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Field info when in rectified mode
        if show_rectified and has_rectification:
            field_info_y = status_y - 25
            field_info = f"Real-world proportions maintained | Aspect ratio: {170.0/130.0:.3f}"
            cv2.putText(frame, field_info, (margin, field_info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Instructions
        instructions = [
            "Press 'v' to toggle view, 'c' to configure points, 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = 30 + i * 25
            cv2.putText(frame, instruction, (margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return {
            'switch': (switch_x, y_base, button_width, button_height),
            'config': (config_x, y_base, button_width, button_height)
        }
    
    def handle_mouse_click(self, event, x, y, flags, param):
        """Handle mouse clicks on control buttons."""
        if event == cv2.EVENT_LBUTTONDOWN:
            buttons, show_rectified, has_rectification = param
            
            # Check switch view button
            if self.point_in_rect((x, y), buttons['switch']) and has_rectification:
                # We need to use a class variable to communicate with main loop
                self.mouse_action = 'switch'
            
            # Check configure points button
            elif self.point_in_rect((x, y), buttons['config']):
                self.mouse_action = 'config'
    
    def point_in_rect(self, point, rect):
        """Check if point is inside rectangle."""
        x, y = point
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh
    
    def run_unified_interface(self):
        """Run the unified interface with normal/rectified view switching."""
        print("\n=== Unified Camera Interface ===")
        print("Controls:")
        print("- Press 'v' or click button to toggle between normal/rectified view")
        print("- Press 'c' or click button to configure rectification points")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        
        # Interface state
        show_rectified = False
        has_rectification = self.rectifier and self.rectifier.is_initialized
        mouse_action = None
        
        # Create window
        window_name = "Camera Interface"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
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
                
                # Apply rectification if needed and available
                if show_rectified and has_rectification:
                    display_frame = self.rectifier.rectify_frame(corrected_frame)
                else:
                    display_frame = corrected_frame.copy()
                
                # Draw control buttons
                buttons = self.draw_control_buttons(display_frame, show_rectified, has_rectification)
                
                # Set mouse callback with current state
                cv2.setMouseCallback(window_name, self.handle_mouse_click, (buttons, show_rectified, has_rectification))
                
                # Display frame
                cv2.imshow(window_name, display_frame)
                
                # Handle mouse actions
                if self.mouse_action == 'switch' and has_rectification:
                    show_rectified = not show_rectified
                    print(f"Switched to {'rectified' if show_rectified else 'normal'} view")
                    self.mouse_action = None
                elif self.mouse_action == 'config':
                    cv2.destroyWindow(window_name)
                    if self.setup_rectification():
                        has_rectification = True
                        print("Rectification configured successfully")
                    else:
                        print("Rectification configuration failed")
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    self.mouse_action = None
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('v') and has_rectification:
                    # Toggle view
                    show_rectified = not show_rectified
                    print(f"Switched to {'rectified' if show_rectified else 'normal'} view")
                elif key == ord('c'):
                    # Configure points
                    cv2.destroyWindow(window_name)
                    if self.setup_rectification():
                        has_rectification = True
                        print("Rectification configured successfully")
                    else:
                        print("Rectification configuration failed")
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                elif key == ord('s'):
                    # Save current frame
                    prefix = "rectified" if (show_rectified and has_rectification) else "normal"
                    filename = f"{prefix}_frame_{frame_count:04d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved frame: {filename}")
                
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
        if os.path.exists("rectification_transform.xml"):
            self.rectifier = RealtimeRectifier(self.frame_width, self.frame_height)
            if self.rectifier.load_transformation_xml("rectification_transform.xml"):
                print("Loaded existing rectification transformation")
            else:
                print("Failed to load existing transformation")
                self.rectifier = None
        else:
            print("No existing rectification found")
            self.rectifier = None
        
        # Run unified interface
        self.run_unified_interface()
        
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
