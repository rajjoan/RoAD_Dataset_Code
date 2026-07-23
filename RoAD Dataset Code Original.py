from RoADDataset import Dataset

# Instantiate the Dataset
ds = Dataset(normalize=True)

# Access a subset
training_subset = ds.sets['training'] # Size 9

import numpy as np
from torch import tensor, float32

# Configuration
SEQ_LENGTH = 50
OVERLAP = False  # Set to True for overlapping, False for non-overlapping
step_size = 1 if OVERLAP else SEQ_LENGTH

trainX = []
trainY = []

for i, run in enumerate(training_subset):
    run_data = np.array(run)
    num_timestamps = run_data.shape[0]

    # Check if run is long enough
    if num_timestamps < SEQ_LENGTH:
        continue

    # Use range with a step_size to control overlap
    # We use (num_timestamps - SEQ_LENGTH + 1) to ensure the last possible window is included
    for start_idx in range(0, num_timestamps - SEQ_LENGTH + 1, step_size):
        sequence = run_data[start_idx : start_idx + SEQ_LENGTH]
        trainX.append(sequence)

        # Target is 0 (Non-anomalous)
        trainY.append(0)

# Convert to tensors
trainX = tensor(np.array(trainX), dtype=float32)
trainY = tensor(trainY, dtype=float32).view(-1, 1)

print(f"Mode: {'Overlapping' if OVERLAP else 'Non-overlapping'}")
print(f"Final Input Shape: {trainX.shape}")

from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    # NOTE: you can look into using dropout for LSTM if you want to... Only really useful if your layer_dim is > 1
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # Add dropout to this line if you want it
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Simplified: out is (batch, seq_len, hidden_dim)
        out, _ = self.lstm(x)

        # Predict the reconstruction for EVERY time step
        # This is called an AUTOENCODER.
        decoded = self.decoder(out)

        return decoded # shape: [batch, seq_len, 86]


from torch.utils.data import DataLoader, TensorDataset

# input_dim: 86 (matches your sample size)
# hidden_dim: 100 (size of the internal memory)
# layer_dim: 2 (stacked layers, 2-3 is the "sweet spot")

# 1. Prepare Data (Ensuring Y is [N, 1])
dataset = TensorDataset(trainX, trainY.view(-1, 1))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = LSTMModel(input_dim=86, hidden_dim=128, layer_dim=2)
criterion = nn.MSELoss()

# 0.01 is a bit high for LSTMs (which are prone to exploding gradients)
# 0.001 is usually the safer starting point for Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 30
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_X, _ in loader: # We don't need labels (Y) anymore
        # Forward pass
        outputs = model(batch_X)

        # Compare output to the original input (batch_X)
        loss = criterion(outputs, batch_X)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {epoch_loss/len(loader):.4f}')


DATASET = 'collision'

# Configuration matches training
step_size = 1 # Usually 1 for evaluation to get high-resolution detection

testX = []
testY = []

# Access the collision set from your Dataset object (assuming 'ds' is the instance)
collision_data = ds.sets[DATASET]

for recording in collision_data:
    # recording shape: (num_timestamps, 87) -> 86 features + 1 label
    num_timestamps = recording.shape[0]

    if num_timestamps < SEQ_LENGTH:
        continue

    for start_idx in range(0, num_timestamps - SEQ_LENGTH + 1, step_size):
        # Extract features (first 86 columns)
        seq_features = recording[start_idx : start_idx + SEQ_LENGTH, :86]

        # Extract label of the LAST timestamp in the window
        # (Alternatively, you could use np.any() if you want to flag if ANY point is an anomaly)
        seq_label = recording[start_idx + SEQ_LENGTH - 1, -1]

        testX.append(seq_features)
        testY.append(seq_label)

# Convert to Tensors
testX = torch.tensor(np.array(testX), dtype=torch.float32)
testY = torch.tensor(np.array(testY), dtype=torch.float32)

print(f"Collision Test Shape: {testX.shape}") # (N, 50, 86)


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

def plot_run_classification(model, run_data, threshold, seq_length=50):
    """
    Plots the reconstruction error for a single run, color-coded by
    prediction accuracy (TP, FP, TN, FN).
    """
    model.eval()

    # Prepare windows for this specific run
    num_timestamps = run_data.shape[0]
    features = run_data[:, :86]
    labels = run_data[:, 86]

    X_windows = []
    Y_labels = []

    for i in range(num_timestamps - seq_length + 1):
        X_windows.append(features[i : i + seq_length])
        Y_labels.append(labels[i + seq_length - 1])

    X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)

    # Get Model Predictions
    with torch.no_grad():
        reconstructed = model(X_tensor)
        # Calculate MSE for each sequence
        errors = torch.mean((reconstructed - X_tensor)**2, dim=(1, 2)).numpy()

    # Categorize each point based on threshold and ground truth
    Y_labels = np.array(Y_labels)
    categories = np.empty(len(errors), dtype=object)

    for i in range(len(errors)):
        if errors[i] > threshold:
            categories[i] = 'TP' if Y_labels[i] == 1 else 'FP'
        else:
            categories[i] = 'TN' if Y_labels[i] == 0 else 'FN'

    # Plotting Setup
    plt.figure(figsize=(15, 6))
    color_map = {'TP': 'red', 'FP': 'orange', 'TN': 'blue', 'FN': 'purple'}
    labels_map = {'TP': 'True Positive (Caught)', 'FP': 'False Positive (False Alarm)',
                  'TN': 'True Negative (Normal)', 'FN': 'False Negative (Miss)'}

    # Match window ends to timestamps for the X-axis
    t = np.arange(seq_length - 1, num_timestamps)

    for cat in ['TN', 'FP', 'FN', 'TP']:
        mask = (categories == cat)
        if np.any(mask):
            plt.scatter(t[mask], errors[mask], color=color_map[cat], label=labels_map[cat], s=12)

    plt.axhline(y=threshold, color='black', linestyle='--', alpha=0.5, label=f'Threshold ({threshold:.4f})')
    plt.title("Run Analysis: Anomaly Scores Categorized by Performance")
    plt.xlabel("Time Step")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.2)
    plt.show()

def print_anomaly_summary(true_labels, anomaly_scores, threshold):
    """
    Prints a formatted report of key classification metrics.
    """
    # Convert continuous scores to binary predictions
    predictions = (anomaly_scores > threshold).astype(int)

    # Calculate Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

    # Calculate Metrics
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    kappa = cohen_kappa_score(true_labels, predictions)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print("\n" + "="*40)
    print(f"{'ANOMALY DETECTION REPORT':^40}")
    print("="*40)
    print(f"Threshold:           {threshold:.6f}")
    print(f"Samples Evaluated:   {len(true_labels)}")
    print("-" * 40)
    print(f"True Positives (TP): {tp:>6}  (Correct Anomaly)")
    print(f"False Positives (FP):{fp:>6}  (False Alarm)")
    print(f"True Negatives (TN): {tn:>6}  (Correct Normal)")
    print(f"False Negatives (FN):{fn:>6}  (Missed Anomaly)")
    print("-" * 40)
    print(f"Accuracy:            {accuracy:.2%}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1-Score:            {f1:.4f}")
    print(f"Cohen's Kappa:       {kappa:.4f}")
    print("="*40 + "\n")


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, precision_recall_curve, average_precision_score

# ==========================================
# 1. GENERATE ANOMALY SCORES (RECONSTRUCTION ERRORS)
# ==========================================
model.eval()
with torch.no_grad():
    # Pass your collision test data through the model
    reconstruction = model(testX)

    # Calculate MSE for each sequence: (Input - Output)^2
    # This is the "Anomaly Score" or "Reconstruction Error"
    reconstruction_errors = torch.mean((reconstruction - testX)**2, dim=(1, 2)).numpy()

# Get ground truth labels for the ROC curve
true_labels = testY.numpy()

# ==========================================
# 2. CALCULATE OPTIMAL THRESHOLD (ROC)
# ==========================================
fpr, tpr, thresholds = roc_curve(true_labels, reconstruction_errors)
roc_auc = auc(fpr, tpr)

# Find the index of the threshold that maximizes (TPR - FPR)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx] # change

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Analysis')
plt.legend(loc="lower right")
plt.show()

# TODO: can also plot the PR Curve: https://www.geeksforgeeks.org/machine-learning/precision-recall-curve-ml/
# ==========================================
# PRECISION-RECALL CURVE
# ==========================================
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(true_labels, reconstruction_errors)
pr_auc = average_precision_score(true_labels, reconstruction_errors)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid()
plt.show()

# ==========================================
# 3. RUN EVALUATION & VISUALIZATION
# ==========================================
# Now we use the functions we defined previously
plot_run_classification(model, ds.sets[DATASET][0], optimal_threshold)

print_anomaly_summary(true_labels, reconstruction_errors, optimal_threshold)

##
datasets = ['collision', 'control', 'weight', 'velocity']
f1_scores = []

for d in datasets:
    data = ds.sets[d]

    tempX = []
    tempY = []

    for recording in data:
        num_timestamps = recording.shape[0]

        if num_timestamps < SEQ_LENGTH:
            continue

        for start_idx in range(0, num_timestamps - SEQ_LENGTH + 1):
            seq_features = recording[start_idx : start_idx + SEQ_LENGTH, :86]
            seq_label = recording[start_idx + SEQ_LENGTH - 1, -1]

            tempX.append(seq_features)
            tempY.append(seq_label)

    tempX = torch.tensor(np.array(tempX), dtype=torch.float32)
    tempY = np.array(tempY)

    with torch.no_grad():
        reconstruction = model(tempX)
        errors = torch.mean((reconstruction - tempX)**2, dim=(1, 2)).numpy()

    preds = (errors > optimal_threshold).astype(int)
    f1 = f1_score(tempY, preds, zero_division=0)
    f1_scores.append(f1)

plt.figure()
plt.bar(datasets, f1_scores)
plt.title("F1 Score Across Subdatasets")
plt.xlabel("Dataset")
plt.ylabel("F1 Score")
plt.show()