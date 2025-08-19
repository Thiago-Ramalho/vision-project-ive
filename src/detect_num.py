import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from camera_capture import CameraCapture, select_camera_interactive, preview_camera
from point_selector import RealtimePointSelector
from realtime_rectifier import RealtimeRectifier
from utils import load_calibration_data
import easyocr

# Define the same model architecture as used for training
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 11) # 10 digits + 1 for "no single digit"

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
loaded_model = Net()

# Define the path to the saved model file
model_save_path = 'mnist_cnn_11_classes.pth'

# Load the saved state dictionary into the model
loaded_model.load_state_dict(torch.load(model_save_path))

# Set the model to evaluation mode
loaded_model.eval()

reader = easyocr.Reader(['en'], gpu=False)

if __name__ == '__main__':
    cv2.namedWindow('Original (Distortion Corrected)', cv2.WINDOW_AUTOSIZE)

    camera_index = 0
    camera = CameraCapture(camera_index)

    patch_size = 28
    stride = 7

    try:
        frame_count = 0

        while True:
            # Get frame
            frame = camera.get_frame()
            if frame is None:
                print("Failed to get frame from camera")
                break
                
            # frame_gray = 255 - cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (320, 180))
            frame_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (640, 360))

            # cropped_h = (frame_gray.shape[0] - patch_size) // stride * stride + patch_size
            # cropped_w = (frame_gray.shape[1] - patch_size) // stride * stride + patch_size
            
            # fgt = torch.tensor(frame_gray).unsqueeze(0).unsqueeze(0).float()/255.0
            # print(fgt)

            # # fgt_cropped = fgt[:, :, :cropped_h, :cropped_w]
            # # print(fgt_cropped.shape)
            # # patches = torch.nn.functional.unfold(fgt_cropped, kernel_size=(patch_size, patch_size), stride=(stride, stride))
            # # print(patches.shape)

            # # patches = torch.tensor(frame_gray).unfold(0,patch_size,stride) # torch.nn.functional.unfold(fgt_cropped, kernel_size=(patch_size, patch_size), stride=(stride, stride))
            # patches = torch.nn.functional.unfold(fgt, kernel_size=patch_size, stride=stride, padding=patch_size)
            # patches = patches.transpose(1, 2).reshape(-1, 1, patch_size, patch_size)
            # print(patches.shape)

            reads = reader.readtext(frame_gray)
            for read in reads:
                if read[1] >= '0' and read[1] <= '9':
                    cv2.rectangle(frame_gray, [int(x) for x in read[0][0]], [int(x) for x in read[0][2]], 0, 5)


            # Apply distortion correction
            corrected_frame = frame_gray
            
            # Display both frames
            cv2.imshow('Original (Distortion Corrected)', corrected_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current rectified frame
                filename = f"rectified_realtime_{frame_count:04d}.jpg"
                cv2.imwrite(filename, rectified_frame)
                print(f"Saved frame: {filename}")

            frame_count += 1
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cv2.destroyAllWindows()
    
    camera.release()
