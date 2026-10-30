import numpy as np
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
# Point this to your ground truth or prediction .npy folder
root_dir = Path("/home/nrashvan/STG-NF/ShanghaiTech_Cleaned_npy")    

#root_dir = Path("/home/nrashvan/STG-NF/data/ShanghaiTech/gt/test_frame_mask/")    
total_dataset_events = 0
total_short_events = 0
short_event_details = []

print(f"{'File Name':<25} | {'Total Events':<12} | {'Short (<15)':<12} | {'Short Durs'}")
print("-" * 80)

for file_path in sorted(root_dir.glob('*.npy')):
    # Load the array (Consecutive 1s are events)
    array = np.load(file_path)
    
    # 1. Detect Events using transitions
    # We pad with 0 to catch events that start at frame 0 or end at the last frame
    padded = np.concatenate([[0], array, [0]])
    diff = np.diff(padded)

    # Starts where value goes from 0 to 1; Ends where it goes from 1 to 0
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1 
    
    num_events = len(starts)
    file_short_count = 0
    file_short_durations = []

    # 2. Analyze Durations
    if num_events > 0:
        durations = ends - starts + 1
        
        for dur in durations:
            if dur < 90:
                file_short_count += 1
                file_short_durations.append(dur)
    
    # Update Global Totals
    total_dataset_events += num_events
    total_short_events += file_short_count
    
    # 3. Print Row for File
    short_label = ",".join(map(str, file_short_durations)) if file_short_durations else "None"
    print(f"{file_path.name:<25} | {num_events:<12} | {file_short_count:<12} | {short_label}")

# --- Final Summary ---
print("\n" + "="*50)
print("DATASET EVENT SUMMARY")
print("="*50)
print(f"Total Events Detected:       {total_dataset_events}")
print(f"Short Events (< 20 frames):  {total_short_events}")
if total_dataset_events > 0:
    percentage = (total_short_events / total_dataset_events) * 100
    print(f"Short Event Ratio:           {percentage:.2f}%")
print("="*50)