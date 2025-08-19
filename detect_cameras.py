#!/usr/bin/env python3
"""
Camera detection and testing script.
Use this script to identify and test available cameras before running calibration.
"""

import sys
import os
sys.path.append('src')

from camera_capture import list_available_cameras, select_camera_interactive, preview_camera

def main():
    """Main function to detect and test cameras."""
    print("=== Camera Detection and Testing ===\n")
    
    # List all available cameras
    cameras = list_available_cameras()
    
    if not cameras:
        print("No cameras detected. Please check your connections.")
        return
    
    print("\n=== Camera Testing ===")
    
    while True:
        print("\nOptions:")
        print("1. Preview a specific camera")
        print("2. Interactive camera selection")
        print("3. List cameras again")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            try:
                camera_idx = int(input("Enter camera index to preview: "))
                # Check if this camera index exists in our detected cameras
                available_indices = [cam['index'] for cam in cameras]
                if camera_idx in available_indices:
                    preview_camera(camera_idx)
                else:
                    print(f"Camera {camera_idx} not found. Available cameras: {available_indices}")
            except ValueError:
                print("Invalid camera index. Please enter a number.")
        
        elif choice == "2":
            selected = select_camera_interactive()
            print(f"You selected camera {selected}")
            
            test_preview = input("Do you want to test this camera? (y/N): ").strip().lower()
            if test_preview in ['y', 'yes']:
                preview_camera(selected)
        
        elif choice == "3":
            cameras = list_available_cameras()
        
        elif choice == "4":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
