"""
Image processing module for handling image rectification and perspective transformation.
"""

import cv2
import numpy as np


class ImageProcessor:
    def __init__(self):
        """Initialize the image processor."""
        pass
    
    def rectify_image(self, image, src_points, target_width=800, target_height=600):
        """
        Rectify an image based on four corner points.
        
        Args:
            image: Input image to rectify
            src_points: Four source points as numpy array [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                       Order: top-left, top-right, bottom-right, bottom-left
            target_width: Width of the output rectified image
            target_height: Height of the output rectified image
            
        Returns:
            numpy.ndarray: Rectified image
        """
        if len(src_points) != 4:
            print("Error: Need exactly 4 points for rectification")
            return None
        
        # Define destination points for a rectangle
        dst_points = np.array([
            [0, 0],                           # top-left
            [target_width - 1, 0],            # top-right
            [target_width - 1, target_height - 1],  # bottom-right
            [0, target_height - 1]            # bottom-left
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transformation
        rectified = cv2.warpPerspective(image, matrix, (target_width, target_height))
        
        return rectified
    
    def auto_rectify_document(self, image, min_area=1000):
        """
        Automatically detect and rectify a document in the image.
        
        Args:
            image: Input image
            min_area: Minimum area for contour detection
            
        Returns:
            tuple: (rectified_image, corners) or (None, None) if detection failed
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Find the largest contour with 4 corners
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < min_area:
                continue
            
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If the approximated contour has 4 points, we found our document
            if len(approx) == 4:
                # Order the points
                corners = self.order_points(approx.reshape(4, 2))
                
                # Calculate target dimensions
                width, height = self.calculate_target_dimensions(corners)
                
                # Rectify the image
                rectified = self.rectify_image(image, corners, int(width), int(height))
                
                return rectified, corners
        
        print("Could not find a rectangular document in the image")
        return None, None
    
    def order_points(self, points):
        """
        Order points in the order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            points: Array of 4 points
            
        Returns:
            numpy.ndarray: Ordered points
        """
        # Initialize ordered points
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference of coordinates
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        
        # Top-left point has the smallest sum
        rect[0] = points[np.argmin(s)]
        
        # Bottom-right point has the largest sum
        rect[2] = points[np.argmax(s)]
        
        # Top-right point has the smallest difference
        rect[1] = points[np.argmin(diff)]
        
        # Bottom-left point has the largest difference
        rect[3] = points[np.argmax(diff)]
        
        return rect
    
    def calculate_target_dimensions(self, corners):
        """
        Calculate target dimensions for rectification based on corner points.
        
        Args:
            corners: Ordered corner points
            
        Returns:
            tuple: (width, height)
        """
        # Calculate width
        width_top = np.linalg.norm(corners[0] - corners[1])
        width_bottom = np.linalg.norm(corners[3] - corners[2])
        width = max(width_top, width_bottom)
        
        # Calculate height
        height_left = np.linalg.norm(corners[0] - corners[3])
        height_right = np.linalg.norm(corners[1] - corners[2])
        height = max(height_left, height_right)
        
        return width, height
    
    def enhance_image(self, image):
        """
        Enhance the image quality (contrast, brightness, etc.).
        
        Args:
            image: Input image
            
        Returns:
            numpy.ndarray: Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def remove_perspective_distortion(self, image, corners, target_aspect_ratio=None):
        """
        Remove perspective distortion while maintaining aspect ratio.
        
        Args:
            image: Input image
            corners: Four corner points
            target_aspect_ratio: Desired aspect ratio (width/height)
            
        Returns:
            numpy.ndarray: Corrected image
        """
        if target_aspect_ratio is None:
            # Calculate original aspect ratio
            width, height = self.calculate_target_dimensions(corners)
            target_aspect_ratio = width / height
        
        # Set target dimensions
        target_height = 800
        target_width = int(target_height * target_aspect_ratio)
        
        return self.rectify_image(image, corners, target_width, target_height)
    
    def draw_points(self, image, points, color=(0, 255, 0), radius=5):
        """
        Draw points on an image.
        
        Args:
            image: Input image
            points: List of points to draw
            color: Color of the points (BGR)
            radius: Radius of the circles
            
        Returns:
            numpy.ndarray: Image with points drawn
        """
        result = image.copy()
        
        for i, point in enumerate(points):
            cv2.circle(result, tuple(map(int, point)), radius, color, -1)
            cv2.putText(result, str(i + 1), 
                       (int(point[0]) + radius + 5, int(point[1]) - radius - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return result
    
    def draw_quadrilateral(self, image, points, color=(0, 255, 0), thickness=2):
        """
        Draw a quadrilateral connecting four points.
        
        Args:
            image: Input image
            points: Four corner points
            color: Color of the lines (BGR)
            thickness: Line thickness
            
        Returns:
            numpy.ndarray: Image with quadrilateral drawn
        """
        result = image.copy()
        
        if len(points) == 4:
            points_int = np.array(points, dtype=np.int32)
            cv2.polylines(result, [points_int], True, color, thickness)
        
        return result


def test_image_processor():
    """Test the image processor functionality."""
    print("Testing image processor...")
    
    # Create a test image with a simple rectangle
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (500, 300), (255, 255, 255), -1)
    
    # Define source points (corners of the rectangle)
    src_points = np.array([
        [100, 100],  # top-left
        [500, 100],  # top-right
        [500, 300],  # bottom-right
        [100, 300]   # bottom-left
    ], dtype=np.float32)
    
    processor = ImageProcessor()
    
    # Test rectification
    rectified = processor.rectify_image(test_image, src_points)
    
    if rectified is not None:
        cv2.imshow('Original', test_image)
        cv2.imshow('Rectified', rectified)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Test completed successfully!")
    else:
        print("Test failed!")


if __name__ == "__main__":
    test_image_processor()
