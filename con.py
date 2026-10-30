import numpy as np
import pandas as pd
import os

# Set your folder paths
source_folder = '/home/nrashvan/STG-NF/Shanghai_Cleaned'
output_folder = '/home/nrashvan/STG-NF/ShanghaiTech_Cleaned_npy'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through CSV files
for filename in os.listdir(source_folder):
    if filename.endswith('.csv'):
        csv_path = os.path.join(source_folder, filename)

        # Read CSV
        df = pd.read_csv(csv_path)

        # Extract label column
        data = df['Label'].to_numpy()

        # Convert filename
        npy_filename = filename.replace('.csv', '.npy')
        npy_path = os.path.join(output_folder, npy_filename)

        # Save as .npy
        np.save(npy_path, data)

        print(f"Converted: {filename} -> {npy_filename}")

print("\nAll CSV files have been converted to NPY successfully!")