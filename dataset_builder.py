import os
import csv
import cv2
import mediapipe as mp

# MediaPipe Tasks API Imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup paths and configurations
DATASET_DIR = "asl_alphabet_train"  # Points to your renamed folder
OUTPUT_CSV = "data.csv"
MAX_SAMPLES_PER_LETTER = 50  # 🏎️ Speed cap: 50 images per letter keeps it incredibly fast!

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3, 
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

headers = [f"point_{i}_{dim}" for i in range(21) for dim in ['x', 'y', 'z']] + ["label"]

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers) # Writes the headers to ensure the file isn't empty!

    print("⚡ Starting high-speed dataset extraction...")
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Cannot find folder '{DATASET_DIR}'! Make sure you renamed it.")
        exit()

    for label in sorted(os.listdir(DATASET_DIR)):
        label_path = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_path):
            continue
            
        print(f"📦 Processing letter: {label}...", end="", flush=True)
        sample_count = 0
        
        for img_name in os.listdir(label_path):
            if sample_count >= MAX_SAMPLES_PER_LETTER:
                break
                
            img_path = os.path.join(label_path, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)
            
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    row_features = []
                    for lm in hand_landmarks:
                        row_features.extend([lm.x, lm.y, lm.z])
                    
                    writer.writerow(row_features + [label])
                    sample_count += 1

        print(f" ✅ Saved {sample_count} samples.")

print(f"\n🎉 Done! 'data.csv' is successfully filled with coordinate values.")