"""
Real-time image rectification system with pre-computed transformation maps.
Supports both homography matrix and pixel lookup table approaches for maximum performance.
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime


class RealtimeRectifier:
    def __init__(self, frame_width, frame_height):
        """
        Initialize real-time rectifier.
        
        Args:
            frame_width: Width of input frames
            frame_height: Height of input frames
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.homography_matrix = None
        self.map_x = None
        self.map_y = None
        self.output_width = None
        self.output_height = None
        self.field_mask = None  # Mask to hide corner areas outside the field
        self.is_initialized = False
        
    def compute_transformation(self, src_points, output_width=None, output_height=None, field_aspect_ratio=None):
        """
        Compute transformation matrix and maps from source points.
        
        Args:
            src_points: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            output_width: Width of output rectified image (default: auto-calculate)
            output_height: Height of output rectified image (default: auto-calculate)
            field_aspect_ratio: Real-world aspect ratio (width/height) of the field (default: 170/130)
        """
        if len(src_points) != 4:
            raise ValueError("Need exactly 4 points for rectification")
        
        # Convert to numpy array
        src_pts = np.array(src_points, dtype=np.float32)
        
        # Default to soccer field proportions: 170cm x 130cm
        if field_aspect_ratio is None:
            field_aspect_ratio = 170.0 / 130.0  # ≈ 1.308
        
        # Calculate output dimensions respecting field proportions and input frame size
        if output_width is None or output_height is None:
            # Calculate the largest rectangle with correct aspect ratio that fits in input frame
            input_aspect_ratio = self.frame_width / self.frame_height
            
            if field_aspect_ratio > input_aspect_ratio:
                # Field is wider than input frame - constrain by width
                self.output_width = self.frame_width
                self.output_height = int(self.frame_width / field_aspect_ratio)
            else:
                # Field is taller than input frame - constrain by height
                self.output_height = self.frame_height
                self.output_width = int(self.frame_height * field_aspect_ratio)
        else:
            self.output_width = output_width
            self.output_height = output_height
        
        print(f"Output dimensions: {self.output_width}x{self.output_height} (aspect ratio: {self.output_width/self.output_height:.3f})")
        print(f"Field aspect ratio: {field_aspect_ratio:.3f} (170cm x 130cm)")
        
        # Define destination points for a perfect rectangle
        dst_pts = np.array([
            [0, 0],                                    # Top-left
            [self.output_width - 1, 0],                # Top-right
            [self.output_width - 1, self.output_height - 1],  # Bottom-right
            [0, self.output_height - 1]               # Bottom-left
        ], dtype=np.float32)
        
        # Compute homography matrix
        self.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Pre-compute transformation maps for maximum performance
        self._compute_maps()
        
        self.is_initialized = True
        print(f"Transformation computed for output size: {self.output_width}x{self.output_height}")
    
    def _compute_maps(self):
        """Pre-compute pixel mapping for real-time rectification."""
        # Create coordinate matrices
        x_coords, y_coords = np.meshgrid(
            np.arange(self.output_width),
            np.arange(self.output_height)
        )
        
        # Stack coordinates
        coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones(x_coords.size)])
        
        # Apply inverse transformation
        inv_homography = np.linalg.inv(self.homography_matrix)
        src_coords = inv_homography @ coords
        
        # Convert to pixel coordinates
        src_x = (src_coords[0] / src_coords[2]).reshape(self.output_height, self.output_width)
        src_y = (src_coords[1] / src_coords[2]).reshape(self.output_height, self.output_width)
        
        # Store maps for cv2.remap
        self.map_x = src_x.astype(np.float32)
        self.map_y = src_y.astype(np.float32)
        
        # Create field mask to hide corner areas
        self._create_field_mask()
    
    def _create_field_mask(self):
        """
        Create a mask for the soccer field that excludes the corner areas.
        Field: 170cm x 130cm
        Corner areas to exclude: 10cm x 45cm each (red areas in corners)
        """
        # Create mask with same dimensions as output
        self.field_mask = np.ones((self.output_height, self.output_width), dtype=np.uint8) * 255
        
        # Calculate corner area dimensions in pixels
        # Field real dimensions: 170cm x 130cm
        # Corner areas: 10cm x 45cm each
        corner_width_cm = 10.0
        corner_height_cm = 45.0
        field_width_cm = 170.0
        field_height_cm = 130.0
        
        # Convert to pixel coordinates
        corner_width_px = int((corner_width_cm / field_width_cm) * self.output_width)
        corner_height_px = int((corner_height_cm / field_height_cm) * self.output_height)
        
        print(f"Field mask: excluding corner areas of {corner_width_px}×{corner_height_px} pixels")
        
        # Black out the four corner areas
        # Top-left corner
        self.field_mask[0:corner_height_px, 0:corner_width_px] = 0
        
        # Top-right corner  
        self.field_mask[0:corner_height_px, self.output_width-corner_width_px:self.output_width] = 0
        
        # Bottom-left corner
        self.field_mask[self.output_height-corner_height_px:self.output_height, 0:corner_width_px] = 0
        
        # Bottom-right corner
        self.field_mask[self.output_height-corner_height_px:self.output_height, 
                       self.output_width-corner_width_px:self.output_width] = 0
    
    def rectify_frame(self, frame):
        """
        Apply real-time rectification to a frame with proper centering, black borders, and field masking.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            numpy array: Rectified frame with same dimensions as input, centered with black borders,
                        and corner areas outside the soccer field masked out
        """
        if not self.is_initialized:
            raise RuntimeError("Rectifier not initialized. Call compute_transformation first.")
        
        # Use pre-computed maps for maximum performance
        rectified = cv2.remap(frame, self.map_x, self.map_y, 
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Apply field mask to hide corner areas
        if self.field_mask is not None:
            # Convert mask to 3-channel for color images
            if len(rectified.shape) == 3:
                mask_3ch = cv2.cvtColor(self.field_mask, cv2.COLOR_GRAY2BGR)
                rectified = cv2.bitwise_and(rectified, mask_3ch)
            else:
                rectified = cv2.bitwise_and(rectified, self.field_mask)
        
        # If rectified image is smaller than input frame, center it with black borders
        if self.output_width != self.frame_width or self.output_height != self.frame_height:
            # Create black frame with input dimensions
            output_frame = np.zeros((self.frame_height, self.frame_width, frame.shape[2]), dtype=frame.dtype)
            
            # Calculate centering offsets
            offset_x = (self.frame_width - self.output_width) // 2
            offset_y = (self.frame_height - self.output_height) // 2
            
            # Place rectified image in center
            output_frame[offset_y:offset_y + self.output_height, 
                        offset_x:offset_x + self.output_width] = rectified
            
            return output_frame
        else:
            return rectified
    
    def save_transformation_xml(self, filename="rectification_transform.xml"):
        """
        Save transformation data to XML file.
        
        Args:
            filename: Output XML filename
        """
        if not self.is_initialized:
            raise RuntimeError("No transformation to save. Call compute_transformation first.")
        
        # Create XML structure
        root = ET.Element("RectificationTransform")
        
        # Add metadata
        metadata = ET.SubElement(root, "Metadata")
        ET.SubElement(metadata, "timestamp").text = datetime.now().isoformat()
        ET.SubElement(metadata, "input_width").text = str(self.frame_width)
        ET.SubElement(metadata, "input_height").text = str(self.frame_height)
        ET.SubElement(metadata, "output_width").text = str(self.output_width)
        ET.SubElement(metadata, "output_height").text = str(self.output_height)
        ET.SubElement(metadata, "field_aspect_ratio").text = f"{170.0/130.0:.6f}"
        ET.SubElement(metadata, "field_description").text = "Mini soccer field: 170cm x 130cm"
        ET.SubElement(metadata, "corner_masking").text = "true"
        ET.SubElement(metadata, "corner_areas").text = "10cm x 45cm each (excluded from output)"
        
        # Add homography matrix
        homography_elem = ET.SubElement(root, "HomographyMatrix")
        homography_elem.set("rows", "3")
        homography_elem.set("cols", "3")
        homography_elem.set("type", "float32")
        
        # Flatten matrix and convert to string
        matrix_data = self.homography_matrix.ravel()
        homography_elem.text = " ".join([f"{val:.10f}" for val in matrix_data])
        
        # Add transformation maps (optional, for very high performance)
        maps_elem = ET.SubElement(root, "TransformationMaps")
        maps_elem.set("width", str(self.output_width))
        maps_elem.set("height", str(self.output_height))
        
        # Save map_x
        map_x_elem = ET.SubElement(maps_elem, "MapX")
        map_x_elem.set("type", "float32")
        map_x_data = self.map_x.ravel()
        map_x_elem.text = " ".join([f"{val:.6f}" for val in map_x_data])
        
        # Save map_y
        map_y_elem = ET.SubElement(maps_elem, "MapY")
        map_y_elem.set("type", "float32")
        map_y_data = self.map_y.ravel()
        map_y_elem.text = " ".join([f"{val:.6f}" for val in map_y_data])
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(filename, encoding="utf-8", xml_declaration=True)
        
        print(f"Transformation saved to {filename}")
    
    def load_transformation_xml(self, filename="rectification_transform.xml"):
        """
        Load transformation data from XML file.
        
        Args:
            filename: Input XML filename
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            tree = ET.parse(filename)
            root = tree.getroot()
            
            # Load metadata
            metadata = root.find("Metadata")
            if metadata is not None:
                self.frame_width = int(metadata.find("input_width").text)
                self.frame_height = int(metadata.find("input_height").text)
                self.output_width = int(metadata.find("output_width").text)
                self.output_height = int(metadata.find("output_height").text)
            
            # Load homography matrix
            homography_elem = root.find("HomographyMatrix")
            if homography_elem is not None:
                matrix_data = list(map(float, homography_elem.text.split()))
                self.homography_matrix = np.array(matrix_data).reshape(3, 3)
            
            # Load transformation maps if available
            maps_elem = root.find("TransformationMaps")
            if maps_elem is not None:
                map_x_elem = maps_elem.find("MapX")
                map_y_elem = maps_elem.find("MapY")
                
                if map_x_elem is not None and map_y_elem is not None:
                    map_x_data = list(map(float, map_x_elem.text.split()))
                    map_y_data = list(map(float, map_y_elem.text.split()))
                    
                    self.map_x = np.array(map_x_data).reshape(self.output_height, self.output_width).astype(np.float32)
                    self.map_y = np.array(map_y_data).reshape(self.output_height, self.output_width).astype(np.float32)
                else:
                    # Recompute maps from homography matrix
                    self._compute_maps()
            else:
                # Recompute maps from homography matrix
                self._compute_maps()
            
            self.is_initialized = True
            print(f"Transformation loaded from {filename}")
            print(f"Output size: {self.output_width}x{self.output_height}")
            return True
            
        except Exception as e:
            print(f"Error loading transformation: {e}")
            return False
    
    def get_transformation_info(self):
        """
        Get information about the current transformation.
        
        Returns:
            dict: Transformation information
        """
        if not self.is_initialized:
            return None
        
        return {
            "input_size": (self.frame_width, self.frame_height),
            "output_size": (self.output_width, self.output_height),
            "homography_matrix": self.homography_matrix.copy(),
            "has_maps": self.map_x is not None and self.map_y is not None
        }
