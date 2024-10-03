import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc

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

# Calculate Precision, Recall, F1, and AUPR
def calculate_metrics(y_true, disease_scores, k, total_mirnas):
    top_k_indices = np.argsort(disease_scores)[-k:]
    y_pred = np.zeros_like(y_true)
    y_pred[top_k_indices] = 1

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = fp
    tn = total_mirnas - k - fp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, disease_scores)
    aupr = auc(recall_curve, precision_curve)

    return precision, recall, f1, aupr


# Evaluate performance metrics for each disease at different percentiles
def evaluate_precision_at_different_percentiles(score_matrix, A):
    percentiles = [0.1, 0.3, 0.5, 0.8]
    total_mirnas = score_matrix.shape[0]

    results = {perc: [] for perc in percentiles}  # Store results for each percentile

    for perc in percentiles:
        for i in range(A.shape[1]):
            known_mirnas = A[:, i]
            known_associations_count = np.sum(known_mirnas)

            if known_associations_count < 10:
                continue

            disease_scores = score_matrix[:, i]
            k = int(np.ceil(perc * known_associations_count))

            precision, recall, f1, aupr = calculate_metrics(known_mirnas, disease_scores, k, total_mirnas)

            # Ensure metrics are in the range [0, 1]
            aupr = max(0, min(aupr, 1))
            f1 = max(0, min(f1, 1))
            precision = max(0, min(precision, 1))
            recall = max(0, min(recall, 1))

            results[perc].append([known_associations_count, aupr, f1, precision, recall])

    return results


# Select the best performance metrics for diseases with the same number of known associations
def select_best_metrics(results):
    best_metrics = {}

    for perc, values in results.items():
        df = pd.DataFrame(values, columns=['known_associations', 'AUPR', 'F1', 'Precision', 'Recall'])
        grouped = df.groupby('known_associations').agg({
            'AUPR': 'max',
            'F1': 'max',
            'Precision': 'max',
            'Recall': 'max'
        }).reset_index()
        best_metrics[perc] = grouped

    return best_metrics


# Plot heatmap for performance metrics
def plot_heatmap(best_metrics):
    metrics = ['AUPR', 'F1', 'Precision', 'Recall']
    labels = ['10%', '30%', '50%', '80%']

    # Set color palette for the heatmap
    cmap_choice = 'YlGnBu'

    for metric in metrics:
        plt.figure(figsize=(10, 8))

        # Prepare heatmap data
        heatmap_data = []
        for perc, df in best_metrics.items():
            heatmap_data.append(df[metric].values)

        # Generate heatmap, xticklabels show every 20th label
        ax = sns.heatmap(heatmap_data, annot=False, cmap=cmap_choice,
                         xticklabels=df['known_associations'][::20],  # Show label every 20th tick
                         yticklabels=labels, cbar_kws={"shrink": 0.5})

        # Show x-ticks only at the bottom, remove ticks in the middle
        ax.set_xticks(np.arange(0, len(df['known_associations']), 20))
        ax.set_yticks(np.arange(len(labels)))
        ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)

        plt.title(f'Heatmap of {metric} for Different Percentages of Known Associations')
        plt.xlabel('Number of Known Associations')
        plt.ylabel('Percentages of Known Associations')
        plt.show()


# Data loading and IMC calculation
ms = pd.read_csv("C://Users//王雅唯//Desktop//要用到的//数据代码//11mirna融合相似.csv", header=None).values
ds = pd.read_csv("C://Users//王雅唯//Desktop//要用到的//数据代码//10疾病融合相似.csv", header=None).values
A = pd.read_csv('C://Users//王雅唯//Desktop//要用到的//数据代码//03现有已知关联.csv', header=None).values

r = 50
lambda_w = 1.0
lambda_h = 1.0
score_matrix, W_opt, H_opt = imc(A, ms, ds, r, lambda_w, lambda_h)

# Evaluate and select the best performance metrics
results = evaluate_precision_at_different_percentiles(score_matrix, A)
best_metrics = select_best_metrics(results)

# Plot heatmap
plot_heatmap(best_metrics)
