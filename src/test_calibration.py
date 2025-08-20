#!/usr/bin/env python3
"""
Camera Calibration Testing Script

This script provides comprehensive testing for camera calibration data including:
1. Loading and validating calibration files
2. Visual distortion correction comparison
3. Reprojection error analysis
4. Real-time distortion correction preview
5. Calibration quality assessment
"""

import sys
import os
sys.path.append('src')

import cv2
import numpy as np
from camera_capture import CameraCapture, select_camera_interactive
from utils import load_calibration_data
import matplotlib.pyplot as plt


class CalibrationTester:
    def __init__(self):
        self.calibration_data = None
        self.camera = None
    
    def load_and_validate_calibration(self, filename=None):
        """Load and validate calibration data."""
        print("=== Loading Calibration Data ===")
        
        # Try different calibration files
        files_to_try = []
        if filename:
            files_to_try.append(filename)
        files_to_try.extend(['calibration_data.xml', 'calibration_data.npz'])
        
        for cal_file in files_to_try:
            if os.path.exists(cal_file):
                print(f"Found calibration file: {cal_file}")
                self.calibration_data = load_calibration_data(cal_file)
                if self.calibration_data:
                    break
        
        if not self.calibration_data:
            print("❌ No valid calibration data found!")
            print("Please run calibration.py first to generate calibration data.")
            return False
        
        # Validate calibration data
        required_keys = ['camera_matrix', 'dist_coeffs']
        for key in required_keys:
            if key not in self.calibration_data:
                print(f"❌ Missing required calibration parameter: {key}")
                return False
        
        print("✅ Calibration data loaded successfully!")
        self.print_calibration_info()
        return True
    
    def print_calibration_info(self):
        """Print detailed calibration information."""
        print("\n=== Calibration Parameters ===")
        
        camera_matrix = self.calibration_data['camera_matrix']
        dist_coeffs = self.calibration_data['dist_coeffs']
        
        print(f"Camera Matrix:")
        print(f"  fx = {camera_matrix[0, 0]:.2f}")
        print(f"  fy = {camera_matrix[1, 1]:.2f}")
        print(f"  cx = {camera_matrix[0, 2]:.2f}")
        print(f"  cy = {camera_matrix[1, 2]:.2f}")
        
        print(f"\nDistortion Coefficients:")
        print(f"  k1 = {dist_coeffs[0, 0]:.6f}")
        print(f"  k2 = {dist_coeffs[0, 1]:.6f}")
        print(f"  p1 = {dist_coeffs[0, 2]:.6f}")
        print(f"  p2 = {dist_coeffs[0, 3]:.6f}")
        if dist_coeffs.shape[1] > 4:
            print(f"  k3 = {dist_coeffs[0, 4]:.6f}")
        
        if 'mean_error' in self.calibration_data:
            print(f"\nReprojection Error: {self.calibration_data['mean_error']:.6f} pixels")
            if self.calibration_data['mean_error'] < 0.5:
                print("✅ Excellent calibration quality!")
            elif self.calibration_data['mean_error'] < 1.0:
                print("✅ Good calibration quality")
            elif self.calibration_data['mean_error'] < 2.0:
                print("⚠️  Acceptable calibration quality")
            else:
                print("❌ Poor calibration quality - consider recalibrating")
        
        if 'image_size' in self.calibration_data:
            print(f"\nCalibration Image Size: {self.calibration_data['image_size']}")
    
    def test_with_test_image(self, image_path=None):
        """Test calibration with a static test image."""
        print("\n=== Testing with Static Image ===")
        
        if image_path and os.path.exists(image_path):
            test_image = cv2.imread(image_path)
        else:
            # Try to use a captured calibration image
            cal_images_dir = 'captured_images'
            if os.path.exists(cal_images_dir):
                images = [f for f in os.listdir(cal_images_dir) if f.endswith(('.jpg', '.png'))]
                if images:
                    test_image = cv2.imread(os.path.join(cal_images_dir, images[0]))
                    print(f"Using calibration image: {images[0]}")
                else:
                    print("No test images available")
                    return
            else:
                print("No test images available")
                return
        
        if test_image is None:
            print("Failed to load test image")
            return
        
        # Apply distortion correction
        camera_matrix = self.calibration_data['camera_matrix']
        dist_coeffs = self.calibration_data['dist_coeffs']
        
        undistorted = cv2.undistort(test_image, camera_matrix, dist_coeffs)
        
        # Create side-by-side comparison
        h, w = test_image.shape[:2]
        comparison = np.hstack((test_image, undistorted))
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "Corrected", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Resize for display if too large
        if comparison.shape[1] > 1400:
            scale = 1400 / comparison.shape[1]
            new_w = int(comparison.shape[1] * scale)
            new_h = int(comparison.shape[0] * scale)
            comparison = cv2.resize(comparison, (new_w, new_h))
        
        cv2.imshow('Calibration Test - Before/After', comparison)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow('Calibration Test - Before/After')
    
    def test_real_time_correction(self):
        """Test real-time distortion correction with camera."""
        print("\n=== Real-Time Correction Test ===")
        
        # Select camera
        camera_index = select_camera_interactive()
        self.camera = CameraCapture(camera_index)
        
        if not self.camera.is_camera_available():
            print("❌ Camera not available")
            return
        
        camera_matrix = self.calibration_data['camera_matrix']
        dist_coeffs = self.calibration_data['dist_coeffs']
        
        print("Real-time distortion correction test")
        print("Controls:")
        print("  's' - Save current frame comparison")
        print("  'q' - Quit")
        
        frame_count = 0
        
        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    break
                
                # Apply distortion correction
                undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
                
                # Create side-by-side comparison
                h, w = frame.shape[:2]
                
                # Resize frames for display
                display_scale = 0.6
                display_w = int(w * display_scale)
                display_h = int(h * display_scale)
                
                frame_small = cv2.resize(frame, (display_w, display_h))
                undistorted_small = cv2.resize(undistorted, (display_w, display_h))
                
                comparison = np.hstack((frame_small, undistorted_small))
                
                # Add labels and info
                cv2.putText(comparison, "Original", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(comparison, "Corrected", (display_w + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Add instructions
                cv2.putText(comparison, "Press 's' to save, 'q' to quit", 
                           (10, comparison.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Real-time Calibration Test', comparison)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save comparison
                    full_comparison = np.hstack((frame, undistorted))
                    filename = f'calibration_test_{frame_count:03d}.jpg'
                    cv2.imwrite(filename, full_comparison)
                    print(f"Saved comparison as {filename}")
                    frame_count += 1
        
        except KeyboardInterrupt:
            print("\nTest interrupted")
        
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
    
    def analyze_distortion_pattern(self):
        """Analyze the distortion pattern from calibration coefficients."""
        print("\n=== Distortion Pattern Analysis ===")
        
        dist_coeffs = self.calibration_data['dist_coeffs']
        camera_matrix = self.calibration_data['camera_matrix']
        
        # Get image size
        if 'image_size' in self.calibration_data:
            w, h = self.calibration_data['image_size']
        else:
            w, h = 640, 480  # Default
        
        # Create distortion visualization
        print("Analyzing distortion coefficients...")
        
        # Analyze distortion coefficients directly
        k1, k2, p1, p2 = dist_coeffs[0][:4]
        k3 = dist_coeffs[0][4] if dist_coeffs.shape[1] > 4 else 0
        
        print(f"Radial distortion coefficients:")
        print(f"  k1 = {k1:.6f}")
        print(f"  k2 = {k2:.6f}")
        print(f"  k3 = {k3:.6f}")
        
        print(f"Tangential distortion coefficients:")
        print(f"  p1 = {p1:.6f}")
        print(f"  p2 = {p2:.6f}")
        
        # Calculate theoretical distortion at different radial distances
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        
        # Test points at different distances from center
        test_radii = [0.1, 0.3, 0.5, 0.7, 0.9]  # Normalized radii
        max_radius = min(w/2, h/2)
        
        print(f"\nDistortion analysis at different distances from center:")
        max_distortion = 0
        
        for norm_r in test_radii:
            # Convert normalized radius to pixels
            r_pixels = norm_r * max_radius
            
            # Normalize coordinates
            x_norm = r_pixels / fx
            y_norm = 0  # Test along horizontal axis
            
            r2 = x_norm**2 + y_norm**2
            r4 = r2**2
            r6 = r2**3
            
            # Calculate radial distortion factor
            radial_factor = 1 + k1*r2 + k2*r4 + k3*r6
            
            # Calculate distortion in pixels
            distortion_x = x_norm * (radial_factor - 1) * fx
            distortion_magnitude = abs(distortion_x)
            
            print(f"  At {norm_r*100:3.0f}% from center: {distortion_magnitude:.2f} pixels distortion")
            max_distortion = max(max_distortion, distortion_magnitude)
        
        print(f"\nEstimated maximum distortion: {max_distortion:.2f} pixels")
        
        # Visual assessment based on coefficients
        if abs(k1) < 0.01 and abs(k2) < 0.001:
            print("✅ Very low distortion - excellent lens quality")
        elif abs(k1) < 0.05 and abs(k2) < 0.01:
            print("✅ Low distortion - good lens quality")
        elif abs(k1) < 0.2 and abs(k2) < 0.05:
            print("⚠️  Moderate distortion - typical for consumer cameras")
        else:
            print("❌ High distortion - calibration is important for this lens")
        
        # Check for barrel vs pincushion distortion
        if k1 < 0:
            print("📷 Barrel distortion detected (negative k1)")
        elif k1 > 0:
            print("📷 Pincushion distortion detected (positive k1)")
        else:
            print("📷 No significant radial distortion")
    
    def create_distortion_visualization(self):
        """Create a visual representation of the distortion correction."""
        print("\n=== Creating Distortion Visualization ===")
        
        camera_matrix = self.calibration_data['camera_matrix']
        dist_coeffs = self.calibration_data['dist_coeffs']
        
        # Get image size
        if 'image_size' in self.calibration_data:
            w, h = self.calibration_data['image_size']
        else:
            w, h = 640, 480
        
        # Create a test pattern with grid lines
        test_image = np.zeros((h, w, 3), dtype=np.uint8)
        test_image.fill(255)  # White background
        
        # Draw grid lines
        grid_spacing = 50
        for x in range(0, w, grid_spacing):
            cv2.line(test_image, (x, 0), (x, h-1), (0, 0, 0), 1)
        for y in range(0, h, grid_spacing):
            cv2.line(test_image, (0, y), (w-1, y), (0, 0, 0), 1)
        
        # Draw circles at different radii
        center = (w//2, h//2)
        for radius in range(50, min(w, h)//2, 50):
            cv2.circle(test_image, center, radius, (128, 128, 128), 1)
        
        # Add center point
        cv2.circle(test_image, center, 5, (0, 0, 255), -1)
        
        # Apply distortion correction
        undistorted = cv2.undistort(test_image, camera_matrix, dist_coeffs)
        
        # Create comparison
        comparison = np.hstack((test_image, undistorted))
        
        # Add labels
        cv2.putText(comparison, "Distorted Grid", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "Corrected Grid", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Scale down if too large for display
        if comparison.shape[1] > 1200:
            scale = 1200 / comparison.shape[1]
            new_w = int(comparison.shape[1] * scale)
            new_h = int(comparison.shape[0] * scale)
            comparison = cv2.resize(comparison, (new_w, new_h))
        
        cv2.imshow('Distortion Visualization', comparison)
        print("Distortion visualization created. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow('Distortion Visualization')
        
        # Save the visualization
        cv2.imwrite('distortion_visualization.jpg', comparison)
        print("Visualization saved as 'distortion_visualization.jpg'")
    
    def run_comprehensive_test(self):
        """Run all calibration tests."""
        print("=== Comprehensive Calibration Test ===\n")
        
        # Load calibration data
        if not self.load_and_validate_calibration():
            return
        
        # Analyze distortion pattern
        self.analyze_distortion_pattern()
        
        # Create distortion visualization
        create_viz = input("\nDo you want to see distortion grid visualization? (Y/n): ").strip().lower()
        if create_viz not in ['n', 'no']:
            self.create_distortion_visualization()
        
        # Test with static image
        self.test_with_test_image()
        
        # Ask for real-time test
        choice = input("\nDo you want to run real-time correction test? (y/N): ").strip().lower()
        if choice in ['y', 'yes']:
            self.test_real_time_correction()
        
        print("\n✅ Calibration testing complete!")


def main():
    """Main function."""
    print("Camera Calibration Testing Tool")
    print("=" * 40)
    
    tester = CalibrationTester()
    
    # Menu
    while True:
        print("\nTesting Options:")
        print("1. Comprehensive test (recommended)")
        print("2. Load and validate calibration only")
        print("3. Real-time correction test only")
        print("4. Distortion pattern analysis only")
        print("5. Create distortion grid visualization")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            tester.run_comprehensive_test()
        elif choice == "2":
            tester.load_and_validate_calibration()
        elif choice == "3":
            if tester.load_and_validate_calibration():
                tester.test_real_time_correction()
        elif choice == "4":
            if tester.load_and_validate_calibration():
                tester.analyze_distortion_pattern()
        elif choice == "5":
            if tester.load_and_validate_calibration():
                tester.create_distortion_visualization()
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select 1-6.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
