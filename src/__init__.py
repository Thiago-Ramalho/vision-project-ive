"""
Camera Calibration and Image Rectification Package

This package provides tools for camera calibration and image rectification using OpenCV.

Modules:
- main: Main application entry point
- calibration: Camera calibration functionality
- camera_capture: Camera input handling
- image_processor: Image processing and rectification
- utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "Vision Project"

# Import main classes for easier access
from .camera_capture import CameraCapture
from .image_processor import ImageProcessor
from .utils import (
    save_calibration_data,
    load_calibration_data,
    save_image,
    load_image
)

__all__ = [
    'CameraCapture',
    'ImageProcessor',
    'save_calibration_data',
    'load_calibration_data',
    'save_image',
    'load_image'
]
