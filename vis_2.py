import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score, average_precision_score

# Directory containing CSV files
directory = 'results/csv_files/'  
results = []
skipped_data = []

# Iterate over all CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        print(f"Processing file: {filename}")

        # Load the data from the CSV file
        data = pd.read_csv(file_path, header=None, sep='\s+')

        # Split the data
        split_data = data[0].str.split(',', expand=True)

        # Prepare DataFrame
        if split_data.shape[1] == 2:
            split_data.columns = ['Ground_Truth', 'Score']
            split_data['Frame'] = split_data.index + 1
            data = split_data
        else:
            print(f"Warning: Expected 2 columns after split, but got {split_data.shape[1]} columns.")
            continue

        # Convert columns to numeric
        data['Ground_Truth'] = pd.to_numeric(data['Ground_Truth'], errors='coerce')
        data['Score'] = pd.to_numeric(data['Score'], errors='coerce')

        # Check for NaN values
        print(data['Ground_Truth'].isna().sum(), "NaN values in Ground Truth")
        print(data['Score'].isna().sum(), "NaN values in Score")

        if data['Ground_Truth'].nunique() < 2:
            print(f"Insufficient classes in Ground Truth for {filename}. Skipping.")
            # Store skipped data for plotting
            skipped_data.append((filename, data))
            continue

        # Calculate AUC-ROC and AUC-PR
        if not data.empty:
            auc_roc = roc_auc_score(data['Ground_Truth'], data['Score'])
            auc_pr = average_precision_score(data['Ground_Truth'], data['Score'])
            print(f"AUC-ROC for {filename}: {auc_roc:.4f}")
            print(f"AUC-PR for {filename}: {auc_pr:.4f}")
        else:
            print(f"No valid data for {filename}. Skipping.")
            continue

        # Save the results
        results.append((filename, auc_roc, auc_pr))

        # Plotting the scores and ground truth
        plt.figure(figsize=(10, 5))

        # Plot Scores
        plt.plot(data['Frame'], data['Score'], label='Score', marker='o', color='blue', linestyle='-', alpha=0.7)

        # Plot Ground Truth
        plt.plot(data['Frame'], data['Ground_Truth'], label='Ground Truth', marker='x', color='red', linestyle='--', alpha=0.7)

        plt.xlabel('Frame Number')
        plt.ylabel('Score / Ground Truth')
        plt.title(f'Scores and Ground Truth for {filename}')
        plt.grid()
        plt.legend()

        # Save the plot to an image file
        plot_file_path = os.path.join(directory, f"{filename.split('.')[0]}_plot.png")
        plt.savefig(plot_file_path, format='png')
        plt.close()  # Close the plot to free memory

# Convert results to DataFrame for easier sorting
results_df = pd.DataFrame(results, columns=['Filename', 'AUC_ROC', 'AUC_PR'])

# Find the top 3 and bottom 3 files based on AUC-ROC and AUC-PR
best_auc_roc = results_df.nlargest(6, 'AUC_ROC')
worst_auc_roc = results_df.nsmallest(6, 'AUC_ROC')
best_auc_pr = results_df.nlargest(6, 'AUC_PR')
worst_auc_pr = results_df.nsmallest(6, 'AUC_PR')

# Print results
print("\nBest AUC-ROC files:")
print(best_auc_roc)

print("\nWorst AUC-ROC files:")
print(worst_auc_roc)

print("\nBest AUC-PR files:")
print(best_auc_pr)

print("\nWorst AUC-PR files:")
print(worst_auc_pr)

# Plot skipped files
for filename, data in skipped_data:
    plt.figure(figsize=(10, 5))

    # Plot Scores
    plt.plot(data['Frame'], data['Score'], label='Score', marker='o', color='blue', linestyle='-', alpha=0.7)

    # Plot Ground Truth
    plt.plot(data['Frame'], data['Ground_Truth'], label='Ground Truth', marker='x', color='red', linestyle='--', alpha=0.7)

    plt.xlabel('Frame Number')
    plt.ylabel('Score / Ground Truth')
    plt.title(f'Scores and Ground Truth (Skipped) for {filename}')
    plt.grid()
    plt.legend()

    # Save the plot to an image file
    plot_file_path = os.path.join(directory, f"{filename.split('.')[0]}_skipped_plot.png")
    plt.savefig(plot_file_path, format='png')
    plt.close()  # Close the plot to free memory