import os
import numpy as np

folder_path = "/home/nrashvan/STG-NF/data/ABC_poselift/gt/test_frame_mask"

npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

total_files = len(npy_files)
total_frames = 0
total_anomalous_frames = 0

print(f"Total .npy files: {total_files}\n")

for file in npy_files:
    file_path = os.path.join(folder_path, file)
    
    data = np.load(file_path)   # shape: (num_frames,)
    
    num_frames = len(data)
    num_anomalies = np.sum(data)   # counts 1s
    
    total_frames += num_frames
    total_anomalous_frames += num_anomalies
    
    print(f"{file}:")
    print(f"  Frames: {num_frames}")
    print(f"  Anomalous frames: {num_anomalies}")
    print(f"  Normal frames: {num_frames - num_anomalies}\n")

print("===== TOTAL =====")
print(f"Total files: {total_files}")
print(f"Total frames: {total_frames}")
print(f"Total anomalous frames: {total_anomalous_frames}")
print(f"Total normal frames: {total_frames - total_anomalous_frames}")