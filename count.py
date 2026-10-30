import os

# Define the directory path
directory_path = '/home/nrashvan/STG-NF/data/ABC/pose/test'

# Get the list of all files and directories in the specified directory
files_and_directories = os.listdir(directory_path)

# Count the number of files (excluding directories)
file_count = len([f for f in files_and_directories if os.path.isfile(os.path.join(directory_path, f))])

print(f"Number of files in '{directory_path}': {file_count}")



directory_path = '/home/nrashvan/STG-NF/data/ABC/pose/train'
# Get the list of all files and directories in the specified directory
files_and_directories = os.listdir(directory_path)

# Count the number of files (excluding directories)
file_count = len([f for f in files_and_directories if os.path.isfile(os.path.join(directory_path, f))])

print(f"Number of files in '{directory_path}': {file_count}")