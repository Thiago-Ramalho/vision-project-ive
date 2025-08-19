"""
Real-time point selector that works with live camera feed.
Allows users to select and adjust 4 corner points while viewing live video.
"""

import cv2
import numpy as np
import threading
import time


class RealtimePointSelector:
    def __init__(self, camera, window_name="Real-time Point Selection"):
        """
        Initialize real-time point selector.
        
        Args:
            camera: Camera object that provides get_frame() method
            window_name: Name of the OpenCV window
        """
        self.camera = camera
        self.window_name = window_name
        self.points = []
        self.selected_point = -1  # Index of currently selected point (-1 = none)
        self.is_dragging = False
        self.running = False
        
        # Visual settings
        self.point_radius = 8
        self.point_color = (0, 255, 0)  # Green
        self.selected_color = (0, 255, 255)  # Yellow
        self.line_color = (255, 0, 0)  # Blue
        self.line_thickness = 2
        
        # Button areas (will be set based on frame size)
        self.button_ok = None
        self.button_cancel = None
        self.button_reset = None
        self.button_undo = None
        
        # State
        self.confirmed = False
        self.cancelled = False
        
        # Threading
        self.frame_lock = threading.Lock()
        self.current_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point selection and manipulation."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on buttons first
            if self.button_ok and self._point_in_rect((x, y), self.button_ok):
                if len(self.points) == 4:
                    self.confirmed = True
                return
            elif self.button_cancel and self._point_in_rect((x, y), self.button_cancel):
                self.cancelled = True
                return
            elif self.button_reset and self._point_in_rect((x, y), self.button_reset):
                self.points = []
                self.selected_point = -1
                return
            elif self.button_undo and self._point_in_rect((x, y), self.button_undo):
                if self.points:
                    self.points.pop()
                    self.selected_point = -1
                return
            
            # Check if clicking near existing point
            for i, point in enumerate(self.points):
                if self._distance((x, y), point) < self.point_radius * 2:
                    self.selected_point = i
                    self.is_dragging = True
                    return
            
            # Add new point if we have less than 4
            if len(self.points) < 4:
                self.points.append((x, y))
                self.selected_point = len(self.points) - 1
        
        elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
            # Move the selected point
            if 0 <= self.selected_point < len(self.points):
                self.points[self.selected_point] = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False
    
    def _distance(self, p1, p2):
        """Calculate distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _point_in_rect(self, point, rect):
        """Check if point is inside rectangle."""
        x, y = point
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh
    
    def _setup_ui_elements(self, frame_height, frame_width):
        """Setup button positions based on frame size."""
        button_width = 80
        button_height = 30
        margin = 10
        
        # Position buttons at the bottom
        y_pos = frame_height - button_height - margin
        
        self.button_ok = (margin, y_pos, button_width, button_height)
        self.button_cancel = (margin + button_width + 10, y_pos, button_width, button_height)
        self.button_reset = (margin + 2 * (button_width + 10), y_pos, button_width, button_height)
        self.button_undo = (margin + 3 * (button_width + 10), y_pos, button_width, button_height)
    
    def _draw_button(self, frame, rect, text, color=(100, 100, 100)):
        """Draw a button on the frame."""
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Center text in button
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_overlay(self, frame):
        """Draw the interactive overlay on the frame."""
        overlay = frame.copy()
        
        # Draw points
        for i, point in enumerate(self.points):
            color = self.selected_color if i == self.selected_point else self.point_color
            cv2.circle(overlay, point, self.point_radius, color, -1)
            cv2.circle(overlay, point, self.point_radius + 2, (255, 255, 255), 2)
            
            # Label points
            label = str(i + 1)
            cv2.putText(overlay, label, (point[0] + 15, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw polygon if we have at least 3 points
        if len(self.points) >= 3:
            pts = np.array(self.points + ([self.points[0]] if len(self.points) == 4 else []), 
                          dtype=np.int32)
            cv2.polylines(overlay, [pts], len(self.points) == 4, self.line_color, self.line_thickness)
        
        # Setup UI elements if not done
        if self.button_ok is None:
            self._setup_ui_elements(frame.shape[0], frame.shape[1])
        
        # Draw buttons
        ok_color = (0, 150, 0) if len(self.points) == 4 else (100, 100, 100)
        self._draw_button(overlay, self.button_ok, "OK", ok_color)
        self._draw_button(overlay, self.button_cancel, "Cancel", (0, 0, 150))
        self._draw_button(overlay, self.button_reset, "Reset", (150, 150, 0))
        
        undo_color = (150, 100, 0) if self.points else (100, 100, 100)
        self._draw_button(overlay, self.button_undo, "Undo", undo_color)
        
        # Instructions
        instructions = [
            "Click to add points (max 4)",
            "Drag points to move them",
            f"Points: {len(self.points)}/4"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = 30 + i * 25
            cv2.putText(overlay, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay
    
    def select_points(self):
        """
        Start the real-time point selection interface.
        
        Returns:
            tuple: (success, points) where success is bool and points is list of (x,y) tuples
        """
        self.running = True
        self.confirmed = False
        self.cancelled = False
        self.points = []
        self.selected_point = -1
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Real-time point selection started...")
        print("- Click to add up to 4 corner points")
        print("- Drag points to adjust their position")
        print("- Use OK button when done, Cancel to abort")
        
        try:
            while self.running and not self.confirmed and not self.cancelled:
                # Get current frame
                frame = self.camera.get_frame()
                if frame is None:
                    print("Failed to get frame from camera")
                    break
                
                # Draw overlay
                display_frame = self._draw_overlay(frame)
                
                # Show frame
                cv2.imshow(self.window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Esc key
                    self.cancelled = True
                elif key == ord(' ') and len(self.points) == 4:  # Space to confirm
                    self.confirmed = True
                elif key == ord('r'):  # R to reset
                    self.points = []
                    self.selected_point = -1
                elif key == ord('u') and self.points:  # U to undo
                    self.points.pop()
                    self.selected_point = -1
        
        except KeyboardInterrupt:
            self.cancelled = True
        
        finally:
            cv2.destroyWindow(self.window_name)
            self.running = False
        
        if self.confirmed and len(self.points) == 4:
            print("Points selected successfully!")
            return True, self.points.copy()
        else:
            print("Point selection cancelled or incomplete")
            return False, []
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        cv2.destroyWindow(self.window_name)
