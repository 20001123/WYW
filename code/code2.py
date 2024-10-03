import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define IMC model function
def imc(A, Sm, Sd, r, lambda_w=1.0, lambda_h=1.0):
    np.random.seed(42)
    W = np.random.rand(Sm.shape[0], r)  # miRNA feature matrix
    H = np.random.rand(Sd.shape[0], r)  # disease feature matrix

    for k in range(1000):
        # Update H matrix (disease feature matrix)
        numerator = Sd.T @ A.T @ Sm @ W  # 64*383 383*495 495*64
        denominator = Sd.T @ Sd @ H @ W.T @ Sm.T @ Sm @ W + lambda_h * H + 1e-9
        H *= numerator / denominator

        # Update W matrix (miRNA feature matrix)
        numerator = Sm.T @ A @ Sd @ H
        denominator = Sm.T @ Sm @ W @ H.T @ Sd.T @ Sd @ H + lambda_w * W + 1e-9
        W *= numerator / denominator

    score = Sm @ W @ H.T @ Sd.T
    return score, W, H


# Load miRNA and disease similarity matrices
ms = pd.read_csv("...//data//05fusionMS.csv", header=None).values
ds = pd.read_csv("...//data//04fusionDS.csv", header=None).values
A = pd.read_csv('...//03knownassociation.csv', header=None).values

# Set IMC model parameters
r = 50  # Latent feature space dimension
lambda_w = 1.0  # Regularization coefficient for W matrix
lambda_h = 1.0  # Regularization coefficient for H matrix

# Compute association score matrix using IMC model
score_matrix, W_opt, H_opt = imc(A, ms, ds, r, lambda_w, lambda_h)

# Define the indices and names of the three diseases
disease_indices = [49, 60, 363]  # Disease indices (50th, 61st, and 364th)
disease_names = ['Breast Neoplasms', 'Non-Small-Cell Lung', 'Stomach Neoplasms']  # Disease names

# Track the number of correct and incorrect predictions for each disease
true_preds = []
false_preds = []

# Output top 50 miRNAs for each disease and count incorrect predictions
for idx, disease_index in enumerate(disease_indices):
    print(f"\nFor {disease_names[idx]} (Disease Index {disease_index + 1}):")
    disease_scores = score_matrix[:, disease_index]

    # Get the top 50 miRNAs with the highest association scores
    top_50_mirnas = np.argsort(disease_scores)[-50:]

    # Count correct and incorrect predictions
    correct_count = 0
    mispred_count = 0

    for mirna_idx in top_50_mirnas[::-1]:  # Sort in descending order
        print(f"miRNA {mirna_idx + 1} has association score: {disease_scores[mirna_idx]:.4f}")

        # Check the A matrix for known associations, 0 indicates a possible false positive
        if A[mirna_idx, disease_index] == 0:
            print(f"miRNA {mirna_idx + 1} has no known association (Possible False Positive)")
            mispred_count += 1
        else:
            correct_count += 1

    # Save the count of correct and incorrect predictions
    true_preds.append(correct_count)
    false_preds.append(mispred_count)

# Plot the bar chart showing true vs. false predictions for each disease
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size

# Stack the correct and incorrect predictions
bars1 = ax.bar(disease_names, true_preds, label='True Predictions', color='skyblue')
bars2 = ax.bar(disease_names, false_preds, bottom=true_preds, label='False Predictions', color='salmon')

# Add labels on the True Predictions bars
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{true_preds[i]}', ha='center', va='center', color='black')

# Add labels on the False Predictions bars
for i, bar in enumerate(bars2):
    height = bar.get_height() + true_preds[i]
    ax.text(bar.get_x() + bar.get_width() / 2, height - bar.get_height() / 2, f'{false_preds[i]}', ha='center',
            va='center', color='black')


ax.set_ylabel('miRNA Number')
ax.set_title('miRNA Prediction Results for Three Diseases (True vs. False Predictions)')

# Adjust legend position
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)

# Ensure layout is tight and margins are correct
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)
plt.tight_layout()
plt.show()
