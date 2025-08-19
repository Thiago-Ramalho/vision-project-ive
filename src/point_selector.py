"""
Enhanced Point Selection Interface for Image Rectification

This module provides an interactive interface for selecting and adjusting
the four corner points for perspective rectification. Features include:
- Drag and drop point selection
- Real-time polygon visualization
- Undo/redo functionality
- Visual feedback and validation
- OK/Cancel buttons
"""

import cv2
import numpy as np
import math


class PointSelector:
    def __init__(self, image, window_name="Point Selection"):
        """
        Initialize the point selector interface.
        
        Args:
            image: The image to select points on
            window_name: Name of the OpenCV window
        """
        self.original_image = image.copy()
        self.display_image = image.copy()
        self.window_name = window_name
        
        # Point management
        self.points = [None, None, None, None]  # [TL, TR, BR, BL]
        self.point_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        self.point_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR
        
        # Interaction state
        self.selected_point = None
        self.dragging = False
        self.mouse_pos = (0, 0)
        
        # UI elements
        self.point_radius = 8
        self.line_thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        # Button areas (will be calculated based on image size)
        self.button_height = 40
        self.button_width = 80
        self.buttons = {}
        self.button_states = {"OK": False, "Cancel": False, "Reset": False}
        
        # History for undo/redo
        self.history = []
        self.history_index = -1
        
        # Result
        self.result = None
        self.cancelled = False
        
        self._setup_interface()
    
    def _setup_interface(self):
        """Setup the interface elements."""
        h, w = self.original_image.shape[:2]
        
        # Calculate button positions
        button_y = h + 10
        self.buttons = {
            "OK": (w - 200, button_y, self.button_width, self.button_height),
            "Cancel": (w - 110, button_y, self.button_width, self.button_height),
            "Reset": (10, button_y, self.button_width, self.button_height),
            "Undo": (100, button_y, self.button_width, self.button_height)
        }
        
        # Create extended canvas for buttons
        self.canvas_height = h + 60
        self.canvas = np.ones((self.canvas_height, w, 3), dtype=np.uint8) * 240
        
        # Copy original image to canvas
        self.canvas[:h, :] = self.original_image
        
        # Save initial state
        self._save_state()
    
    def _save_state(self):
        """Save current state for undo functionality."""
        state = [p.copy() if p is not None else None for p in self.points]
        self.history = self.history[:self.history_index + 1]  # Remove redo history
        self.history.append(state)
        self.history_index = len(self.history) - 1
    
    def _undo(self):
        """Undo last action."""
        if self.history_index > 0:
            self.history_index -= 1
            self.points = [p.copy() if p is not None else None for p in self.history[self.history_index]]
            self._update_display()
    
    def _reset_points(self):
        """Reset all points."""
        self.points = [None, None, None, None]
        self._save_state()
        self._update_display()
    
    def _point_distance(self, p1, p2):
        """Calculate distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _find_nearest_point(self, pos):
        """Find the nearest point to the given position."""
        min_dist = float('inf')
        nearest_idx = None
        
        for i, point in enumerate(self.points):
            if point is not None:
                dist = self._point_distance(pos, point)
                if dist < min_dist and dist < 20:  # Within 20 pixels
                    min_dist = dist
                    nearest_idx = i
        
        return nearest_idx
    
    def _find_empty_point_slot(self):
        """Find the first empty point slot."""
        for i, point in enumerate(self.points):
            if point is None:
                return i
        return None
    
    def _point_in_button(self, pos, button_name):
        """Check if position is inside a button."""
        if button_name not in self.buttons:
            return False
        
        x, y, w, h = self.buttons[button_name]
        px, py = pos
        return x <= px <= x + w and y <= py <= y + h
    
    def _draw_button(self, button_name, pressed=False):
        """Draw a button on the canvas."""
        if button_name not in self.buttons:
            return
        
        x, y, w, h = self.buttons[button_name]
        
        # Button colors
        if pressed:
            color = (200, 200, 200)
            text_color = (0, 0, 0)
        else:
            color = (220, 220, 220)
            text_color = (50, 50, 50)
        
        # Draw button background
        cv2.rectangle(self.canvas, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(self.canvas, (x, y), (x + w, y + h), (100, 100, 100), 2)
        
        # Draw button text
        text_size = cv2.getTextSize(button_name, self.font, self.font_scale, self.font_thickness)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(self.canvas, button_name, (text_x, text_y), 
                   self.font, self.font_scale, text_color, self.font_thickness)
    
    def _draw_polygon(self):
        """Draw the polygon formed by the four points."""
        valid_points = [p for p in self.points if p is not None]
        
        if len(valid_points) >= 3:
            # Draw polygon outline
            if len(valid_points) == 4:
                # Complete quadrilateral
                pts = np.array(self.points, dtype=np.int32)
                cv2.polylines(self.canvas, [pts], True, (0, 255, 255), self.line_thickness)
                
                # Fill with semi-transparent overlay
                overlay = self.canvas.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 255))
                cv2.addWeighted(self.canvas, 0.9, overlay, 0.1, 0, self.canvas)
            else:
                # Partial polygon
                for i in range(len(valid_points) - 1):
                    cv2.line(self.canvas, tuple(valid_points[i]), tuple(valid_points[i + 1]), 
                            (128, 128, 255), self.line_thickness)
    
    def _draw_points(self):
        """Draw all points on the canvas."""
        for i, point in enumerate(self.points):
            if point is not None:
                color = self.point_colors[i]
                
                # Highlight selected point
                if i == self.selected_point:
                    cv2.circle(self.canvas, tuple(point), self.point_radius + 3, (255, 255, 255), -1)
                
                # Draw point
                cv2.circle(self.canvas, tuple(point), self.point_radius, color, -1)
                cv2.circle(self.canvas, tuple(point), self.point_radius, (0, 0, 0), 2)
                
                # Draw point label
                label = f"{i+1}: {self.point_names[i]}"
                label_pos = (point[0] + 15, point[1] - 10)
                cv2.putText(self.canvas, label, label_pos, self.font, 
                           self.font_scale, color, self.font_thickness)
    
    def _draw_instructions(self):
        """Draw instruction text."""
        h, w = self.original_image.shape[:2]
        instructions = [
            "Instructions:",
            "1. Click to place points (order: TL, TR, BR, BL)",
            "2. Drag points to adjust position",
            "3. Right-click point to remove it",
            "4. Click OK when satisfied"
        ]
        
        y_start = h + 5
        for i, instruction in enumerate(instructions):
            y_pos = y_start + i * 15
            if i == 0:
                cv2.putText(self.canvas, instruction, (w - 400, y_pos), 
                           self.font, 0.7, (0, 0, 0), 2)
            else:
                cv2.putText(self.canvas, instruction, (w - 400, y_pos), 
                           self.font, 0.5, (50, 50, 50), 1)
    
    def _draw_point_count(self):
        """Draw point count information."""
        valid_count = sum(1 for p in self.points if p is not None)
        status_text = f"Points: {valid_count}/4"
        
        if valid_count == 4:
            status_text += " - Ready for rectification!"
            color = (0, 150, 0)
        else:
            color = (0, 0, 150)
        
        cv2.putText(self.canvas, status_text, (10, 25), 
                   self.font, 0.7, color, 2)
    
    def _update_display(self):
        """Update the display with current state."""
        # Reset canvas
        h, w = self.original_image.shape[:2]
        self.canvas = np.ones((self.canvas_height, w, 3), dtype=np.uint8) * 240
        self.canvas[:h, :] = self.original_image
        
        # Draw polygon first (so it's behind points)
        self._draw_polygon()
        
        # Draw points
        self._draw_points()
        
        # Draw UI elements
        self._draw_instructions()
        self._draw_point_count()
        
        # Draw buttons
        for button_name in self.buttons:
            self._draw_button(button_name, self.button_states.get(button_name, False))
        
        # Show the canvas
        cv2.imshow(self.window_name, self.canvas)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        self.mouse_pos = (x, y)
        
        # Only handle clicks on the image area
        h, w = self.original_image.shape[:2]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check button clicks first
            for button_name in self.buttons:
                if self._point_in_button((x, y), button_name):
                    self.button_states[button_name] = True
                    self._update_display()
                    return
            
            # Handle image area clicks
            if y < h:
                # Find nearest point
                nearest_idx = self._find_nearest_point((x, y))
                
                if nearest_idx is not None:
                    # Start dragging existing point
                    self.selected_point = nearest_idx
                    self.dragging = True
                else:
                    # Place new point
                    empty_slot = self._find_empty_point_slot()
                    if empty_slot is not None:
                        self.points[empty_slot] = [x, y]
                        self.selected_point = empty_slot
                        self._save_state()
                
                self._update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Handle button releases
            for button_name in self.buttons:
                if self.button_states.get(button_name, False):
                    self.button_states[button_name] = False
                    if self._point_in_button((x, y), button_name):
                        # Button was clicked
                        if button_name == "OK":
                            valid_count = sum(1 for p in self.points if p is not None)
                            if valid_count == 4:
                                self.result = [tuple(p) for p in self.points]
                                return
                        elif button_name == "Cancel":
                            self.cancelled = True
                            return
                        elif button_name == "Reset":
                            self._reset_points()
                        elif button_name == "Undo":
                            self._undo()
                    
                    self._update_display()
                    return
            
            # Stop dragging
            if self.dragging:
                self.dragging = False
                self._save_state()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to remove point
            if y < h:
                nearest_idx = self._find_nearest_point((x, y))
                if nearest_idx is not None:
                    self.points[nearest_idx] = None
                    self.selected_point = None
                    self._save_state()
                    self._update_display()
        
        elif event == cv2.EVENT_MOUSEMOVE:
            # Handle dragging
            if self.dragging and self.selected_point is not None and y < h:
                self.points[self.selected_point] = [x, y]
                self._update_display()
    
    def select_points(self):
        """
        Start the point selection interface.
        
        Returns:
            tuple: (success, points) where success is bool and points is list of 4 (x,y) tuples
                  Returns (False, None) if cancelled
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self._update_display()
        
        print("Point Selection Interface")
        print("========================")
        print("Instructions:")
        print("- Click to place points in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
        print("- Drag points to adjust their position")
        print("- Right-click a point to remove it")
        print("- Click OK when all 4 points are placed correctly")
        print("- Press ESC or click Cancel to abort")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC key
                self.cancelled = True
                break
            elif key == ord('u'):  # U key for undo
                self._undo()
            elif key == ord('r'):  # R key for reset
                self._reset_points()
            
            if self.result is not None:
                break
            
            if self.cancelled:
                break
        
        cv2.destroyWindow(self.window_name)
        
        if self.cancelled:
            return False, None
        elif self.result is not None:
            return True, self.result
        else:
            return False, None


def test_point_selector():
    """Test the point selector with a sample image."""
    # Create a test image
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a test pattern
    cv2.rectangle(test_image, (100, 100), (540, 380), (200, 200, 200), -1)
    cv2.rectangle(test_image, (150, 150), (490, 330), (100, 100, 100), 2)
    cv2.putText(test_image, "Test Document", (200, 250), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Test the selector
    selector = PointSelector(test_image)
    success, points = selector.select_points()
    
    if success:
        print(f"Selected points: {points}")
    else:
        print("Selection cancelled")


if __name__ == "__main__":
    test_point_selector()
