import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

# Load the data from the CSV file
file_path = 'results/csv_files/01_222.csv'  # Update with your CSV file path
data = pd.read_csv(file_path, header=None, delim_whitespace=True)

# Split the data
split_data = data[0].str.split(',', expand=True)

# Prepare DataFrame
if split_data.shape[1] == 2:
    split_data.columns = ['Ground_Truth', 'Score']
    split_data['Frame'] = split_data.index + 1
    data = split_data
else:
    print(f"Warning: Expected 2 columns after split, but got {split_data.shape[1]} columns.")

# Convert columns to numeric
data['Ground_Truth'] = pd.to_numeric(data['Ground_Truth'], errors='coerce')
data['Score'] = pd.to_numeric(data['Score'], errors='coerce')

# Calculate AUC-ROC
auc_roc = roc_auc_score(data['Ground_Truth'], data['Score'])
print(f"AUC-ROC: {auc_roc:.4f}")

# Calculate AUC-PR
auc_pr = average_precision_score(data['Ground_Truth'], data['Score'])
print(f"AUC-PR: {auc_pr:.4f}")

# Plotting the scores and ground truth against frames
plt.figure(figsize=(10, 5))

# Plot Scores
plt.plot(data['Frame'], data['Score'], label='Score', marker='o', color='blue', linestyle='-', alpha=0.7)

# Plot Ground Truth
plt.plot(data['Frame'], data['Ground_Truth'], label='Ground Truth', marker='x', color='red', linestyle='--', alpha=0.7)

plt.xlabel('Frame Number')
plt.ylabel('Score / Ground Truth')
plt.title('Scores and Ground Truth Based on Each Frame')
plt.grid()
plt.legend()

# Save the plot to an image file
plt.savefig('scores_and_ground_truth_plot.png', format='png')

# Optionally, display the plot
plt.show()