"""
Camera calibration module for calculating camera matrix and distortion coefficients.
This module helps remove radial distortion from camera images.
"""

import cv2
import numpy as np
import os
from camera_capture import CameraCapture, select_camera_interactive, preview_camera
from utils import save_calibration_data


class CameraCalibrator:
    def __init__(self, chessboard_size=(9, 6), camera_index=None):
        """
        Initialize the calibrator.
        
        Args:
            chessboard_size: Tuple of (width, height) of internal chessboard corners
            camera_index: Index of camera to use (None for interactive selection)
        """
        self.chessboard_size = chessboard_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane
        
        # Select camera
        if camera_index is None:
            camera_index = select_camera_interactive()
        
        # Ask if user wants to preview the camera
        preview_choice = input(f"\nDo you want to preview camera {camera_index} before calibration? (y/N): ").strip().lower()
        if preview_choice in ['y', 'yes']:
            preview_camera(camera_index)
        
        self.camera = CameraCapture(camera_index)
        self.calibration_images = []
        
    def capture_calibration_images(self, min_images=10):
        """
        Capture images for calibration.
        
        Args:
            min_images: Minimum number of images needed for calibration
        """
        print("Camera Calibration - Image Capture")
        print(f"Capture at least {min_images} images of a chessboard pattern")
        print("Controls:")
        print("  Space - Capture image")
        print("  'q' - Finish capture")
        print(f"Looking for {self.chessboard_size[0]}x{self.chessboard_size[1]} internal corners")
        
        # Create directory for captured images
        os.makedirs('captured_images', exist_ok=True)
        
        captured_count = 0
        
        try:
            while captured_count < min_images:
                frame = self.camera.get_frame()
                if frame is None:
                    print("Failed to get frame from camera")
                    break
                
                # Convert to grayscale for chessboard detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
                
                # Draw corners if found
                display_frame = frame.copy()
                if ret:
                    cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret)
                    cv2.putText(display_frame, "Chessboard detected - Press SPACE to capture", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No chessboard detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(display_frame, f"Captured: {captured_count}/{min_images}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.imshow('Calibration - Capture Images', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and ret:
                    # Refine corners
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    
                    # Store points
                    self.objpoints.append(self.objp)
                    self.imgpoints.append(corners2)
                    
                    # Save image
                    image_filename = f'captured_images/calibration_{captured_count:03d}.jpg'
                    cv2.imwrite(image_filename, frame)
                    self.calibration_images.append(frame.copy())
                    
                    captured_count += 1
                    print(f"Captured image {captured_count}")
                    
                elif key == ord('q'):
                    if captured_count >= min_images:
                        break
                    else:
                        print(f"Need at least {min_images} images. Currently have {captured_count}")
        
        except KeyboardInterrupt:
            print("\nCapture interrupted by user")
        
        finally:
            cv2.destroyWindow('Calibration - Capture Images')
        
        return captured_count >= min_images
    
    def calibrate_camera(self):
        """
        Perform camera calibration using captured images.
        
        Returns:
            Dictionary containing calibration results
        """
        if len(self.objpoints) == 0 or len(self.imgpoints) == 0:
            print("No calibration data available")
            return None
        
        print(f"Performing calibration with {len(self.objpoints)} images...")
        
        # Get image size from the first calibration image
        if self.calibration_images:
            h, w = self.calibration_images[0].shape[:2]
        else:
            # Fallback to getting a frame
            frame = self.camera.get_frame()
            if frame is not None:
                h, w = frame.shape[:2]
            else:
                print("Cannot determine image size")
                return None
        
        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, (w, h), None, None
        )
        
        if ret:
            print("Calibration successful!")
            print(f"Camera matrix:\n{camera_matrix}")
            print(f"Distortion coefficients:\n{dist_coeffs}")
            
            # Calculate reprojection error
            total_error = 0
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(
                    self.objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
            
            mean_error = total_error / len(self.objpoints)
            print(f"Mean reprojection error: {mean_error}")
            
            calibration_data = {
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs,
                'rvecs': rvecs,
                'tvecs': tvecs,
                'mean_error': mean_error,
                'image_size': (w, h)
            }
            
            return calibration_data
        
        else:
            print("Calibration failed!")
            return None
    
    def run_calibration(self):
        """Run the complete calibration process."""
        print("Starting camera calibration process...")
        
        # Capture images
        if self.capture_calibration_images():
            # Perform calibration
            calibration_data = self.calibrate_camera()
            
            if calibration_data is not None:
                # Save calibration data in XML format (cross-platform compatible)
                save_calibration_data(calibration_data, 'calibration_data.xml', 'xml')
                print("Calibration data saved successfully in XML format!")
                
                # Also save in NPZ format for backward compatibility (optional)
                # save_calibration_data(calibration_data, 'calibration_data.npz', 'npz')
                
                # Demonstrate distortion correction
                self.demonstrate_correction(calibration_data)
                
                return True
            else:
                print("Calibration failed")
                return False
        else:
            print("Not enough images captured for calibration")
            return False
    
    def demonstrate_correction(self, calibration_data):
        """Demonstrate the distortion correction."""
        print("Demonstrating distortion correction...")
        print("Press any key to continue")
        
        camera_matrix = calibration_data['camera_matrix']
        dist_coeffs = calibration_data['dist_coeffs']
        
        frame = self.camera.get_frame()
        if frame is not None:
            # Show original and corrected side by side
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
            
            # Resize images for display
            h, w = frame.shape[:2]
            frame_resized = cv2.resize(frame, (w//2, h//2))
            undistorted_resized = cv2.resize(undistorted, (w//2, h//2))
            
            # Create side-by-side comparison
            comparison = np.hstack((frame_resized, undistorted_resized))
            
            # Add labels
            cv2.putText(comparison, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(comparison, "Corrected", (w//2 + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Distortion Correction Comparison', comparison)
            cv2.waitKey(0)
            cv2.destroyWindow('Distortion Correction Comparison')
    
    def __del__(self):
        """Clean up resources."""
        self.camera.release()


def main():
    """Main function for calibration."""
    calibrator = CameraCalibrator()
    calibrator.run_calibration()


if __name__ == "__main__":
    main()
