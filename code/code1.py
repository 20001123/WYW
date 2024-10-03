import torch
import torch.nn as nn
import pandas as pd
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# Define autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def imc(A, Sm, Sd, r, lambda_w=1.0, lambda_h=1.0):
    # m corresponds to the number of miRNAs, n to the number of diseases
    W = np.random.rand(64, r)
    H = np.random.rand(64, r)

    for k in range(1000):
        # Update rule for H (disease feature matrix)
        """
        To optimize the algorithm's performance, adjustments were made to the 
        update rules by fine-tuning the terms in the numerator and denominator 
        during the gradient descent steps, making them more suitable for efficient computation.
        """
        numerator = Sd.T @ A.T @ Sm @ W
        denominator = Sd.T @ Sd @ H @ W.T @ Sm.T @ Sm @ W + lambda_h * H + 1e-9  # No change in H's dimension during update
        H *= numerator / denominator

        # Update rule for W (miRNA feature matrix)
        numerator = Sm.T @ A @ Sd @ H
        denominator = Sm.T @ Sm @ W @ H.T @ Sd.T @ Sd @ H + lambda_w * W + 1e-9  # No change in W's dimension during update
        W *= numerator / denominator

    # Calculate the score matrix
    score = Sm @ W @ H.T @ Sd.T
    return score, W, H

# Loading similarity matrix
ms = pd.read_csv("...//data//05fusionMS.csv", header=None).values
ds = pd.read_csv("...//data//04fusionDS.csv", header=None).values

ms_tensor = torch.tensor(ms, dtype=torch.float32)
ds_tensor = torch.tensor(ds, dtype=torch.float32)

# Initialize
input_dim_ms = ms.shape[1]
encoding_dim = 64
autoencoder_ms = Autoencoder(input_dim_ms, encoding_dim)

# Define optimizer and loss function，加入L2正则化(weight_decay=1e-5)
optimizer_ms = torch.optim.Adam(autoencoder_ms.parameters(), lr=0.001,weight_decay=1e-5)
criterion = nn.MSELoss()

epochs = 100
set_seed(42)
for epoch in range(epochs):
    #Forward propagation
    reconstructed_ms = autoencoder_ms(ms_tensor)
    loss_ms = criterion(reconstructed_ms, ms_tensor)

    # Backpropagation and Optimization
    optimizer_ms.zero_grad()
    loss_ms.backward()
    optimizer_ms.step()

    print(f'Epoch {epoch + 1}, Loss: {loss_ms.item()}')

ms_updated = autoencoder_ms.encoder(ms_tensor).detach().numpy()
np.savetxt("...\\Sm_AE.csv", ms_updated, delimiter=",")

# Update miRNA similarity matrix
input_dim_ds = ds.shape[1]
autoencoder_ds = Autoencoder(input_dim_ds, encoding_dim)

optimizer_ds = torch.optim.Adam(autoencoder_ds.parameters(), lr=0.001,weight_decay=1e-5)

set_seed(42)
for epoch in range(epochs):
    reconstructed_ds = autoencoder_ds(ds_tensor)
    loss_ds = criterion(reconstructed_ds, ds_tensor)

    optimizer_ds.zero_grad()
    loss_ds.backward()
    optimizer_ds.step()

    print(f'Epoch {epoch + 1}, Loss: {loss_ds.item()}')

ds_updated = autoencoder_ds.encoder(ds_tensor).detach().numpy()
np.savetxt("...\\Sd_AE.csv", ds_updated, delimiter=",")

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import torch

A = pd.read_csv('...//03knownassociation.csv', header=None).values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fprs = {i: [] for i in range(kf.n_splits)}
tprs = {i: [] for i in range(kf.n_splits)}
mean_fpr = np.linspace(0, 1, 100)

# Cross-validation and training logic of the IMC model
ranks = [10, 50, 100, 120, 150, 180, 210, 250, 300]  # Range of ranks to try
lambda_ws = [0.2, 0.5, 0.8, 1]  # Range of lambda_w values to try
lambda_hs = [0.1, 0.3, 0.5, 0.7, 1]  # Range of lambda_h values to try

set_seed(42)
for r in ranks:
    for lambda_w in lambda_ws:
        for lambda_h in lambda_hs:
            for fold_index, (train, test) in enumerate(kf.split(A)):

                A_train = A.copy()
                A_train[test, :] = 0
                score_matrix, W_opt, H_opt = imc(A_train, ms_updated, ds_updated, r, lambda_w, lambda_h)

                # Make predictions for the test set indices
                pred = score_matrix[test, :]
                true = A[test, :]

                # Calculate ROC AUC only for the predicted test set
                fpr, tpr, thresholds = roc_curve(true.ravel(), pred.ravel())
                tprs[fold_index].append(np.interp(mean_fpr, fpr, tpr))
                tprs[fold_index][-1][0] = 0.0

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
plt.figure(figsize=(10, 8))
# Plot ROC curves for each fold and the average ROC curve.
for fold_index in range(kf.n_splits):
    mean_tpr = np.mean(tprs[fold_index], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # Plot ROC curves for all parameter combinations
    for param_tpr in tprs[fold_index]:
        plt.plot(mean_fpr, param_tpr, lw=1, alpha=0.3, color=colors[fold_index])
    # Plot the average ROC curve for each fold
    plt.plot(mean_fpr, mean_tpr, color=colors[fold_index], label=f'Fold{fold_index + 1}Mean ROC(AUC={mean_auc:.2f})',
             lw=2, alpha=0.8)

# Plot the overall average ROC curve for all folds
all_mean_tpr = np.mean([np.mean(tprs[i], axis=0) for i in range(kf.n_splits)], axis=0)
all_mean_tpr[-1] = 1.0
all_mean_auc = auc(mean_fpr, all_mean_tpr)
plt.plot(mean_fpr, all_mean_tpr, color='black', lw=2, alpha=0.8, label=f'Overall Mean ROC (AUC = {all_mean_auc:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()