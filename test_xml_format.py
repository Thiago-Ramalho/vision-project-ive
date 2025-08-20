#!/usr/bin/env python3
"""
Test script to demonstrate XML calibration data format.
This script shows how the XML format is cross-platform compatible.
"""

import sys
import os
sys.path.append('src')

from utils import save_calibration_data, load_calibration_data
import numpy as np

def test_xml_format():
    """Test the XML calibration data format."""
    print("Testing XML calibration data format...")
    
    # Create sample calibration data
    sample_data = {
        'camera_matrix': np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        'dist_coeffs': np.array([[-0.1, 0.02, 0.001, 0.002, 0.0]], dtype=np.float64),
        'image_size': (640, 480),
        'mean_error': 0.045
    }
    
    # Save to XML format
    print("Saving sample calibration data to XML...")
    save_calibration_data(sample_data, 'test_calibration.xml', 'xml')
    
    # Load from XML format
    print("Loading calibration data from XML...")
    loaded_data = load_calibration_data('test_calibration.xml')
    
    if loaded_data is not None:
        print("✅ XML format test successful!")
        print("\nLoaded data:")
        print(f"Camera matrix shape: {loaded_data['camera_matrix'].shape}")
        print(f"Distortion coefficients shape: {loaded_data['dist_coeffs'].shape}")
        print(f"Image size: {loaded_data['image_size']}")
        print(f"Mean error: {loaded_data['mean_error']}")
        
        # Clean up test file
        if os.path.exists('test_calibration.xml'):
            os.remove('test_calibration.xml')
        if os.path.exists('test_calibration_summary.txt'):
            os.remove('test_calibration_summary.txt')
    else:
        print("❌ XML format test failed!")

if __name__ == "__main__":
    test_xml_format()
