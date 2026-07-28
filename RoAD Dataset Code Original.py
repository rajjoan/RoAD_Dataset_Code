from RoADDataset import Dataset

import numpy as np
import torch
import torch.nn as nn
from torch import tensor, float32
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    cohen_kappa_score, roc_curve, auc, precision_recall_curve,
    average_precision_score
)

# ==========================================
# 0. CONFIG
# ==========================================
SEQ_LENGTH = 50
OVERLAP = True  # More overlapping windows -> more training sequences from the same 9 runs
step_size = 1 if OVERLAP else SEQ_LENGTH

INPUT_DIM = 86
HIDDEN_DIM = 128       # Try 64 as well once this is running -- see note in LSTMModel
LAYER_DIM = 3          # Modest bump from 2 -> 3 (not jumping straight to 8)
DROPOUT = 0.2          # Only applies between stacked layers (layer_dim > 1)
BATCH_SIZE = 32
LR = 0.001
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 15   # Stop if val loss doesn't improve for this many epochs
VAL_FRACTION = 0.15         # Fraction of normal training sequences held out for validation
THRESHOLD_PERCENTILE = 95   # Threshold = this percentile of VALIDATION reconstruction errors

# ==========================================
# 1. LOAD + WINDOW TRAINING DATA (ALL NORMAL)
# ==========================================
ds = Dataset(normalize=True)
training_subset = ds.sets['training']  # Size 9 -- assumed all-normal runs

# ==========================================
# DIAGNOSTIC 1: LABEL BALANCE PER SUBSET (run this first, before training)
# ==========================================
# If a subset is ~100% anomalous (or ~0%), a model that just flags
# "everything anomalous" will score perfectly on it -- that's not real
# detection skill, just a degenerate classifier meeting a degenerate label
# distribution. Worth knowing this BEFORE trusting any F1 score.
print("\n" + "=" * 50)
print("LABEL BALANCE CHECK (per subset, per-timestep)")
print("=" * 50)
for set_name in ['collision', 'control', 'weight', 'velocity']:
    total_timesteps = 0
    anomalous_timesteps = 0
    for recording in ds.sets[set_name]:
        labels = np.array(recording)[:, -1]
        total_timesteps += len(labels)
        anomalous_timesteps += int(np.sum(labels == 1))
    pct = 100 * anomalous_timesteps / total_timesteps if total_timesteps > 0 else float('nan')
    print(f"{set_name:>10}: {anomalous_timesteps:>7} / {total_timesteps:>7} timesteps anomalous ({pct:.1f}%)")
print("=" * 50 + "\n")


all_sequences = []

for i, run in enumerate(training_subset):
    run_data = np.array(run)
    num_timestamps = run_data.shape[0]

    if num_timestamps < SEQ_LENGTH:
        continue

    for start_idx in range(0, num_timestamps - SEQ_LENGTH + 1, step_size):
        sequence = run_data[start_idx: start_idx + SEQ_LENGTH]
        all_sequences.append(sequence)

all_sequences = np.array(all_sequences)
print(f"Mode: {'Overlapping' if OVERLAP else 'Non-overlapping'}")
print(f"Total normal sequences before train/val split: {all_sequences.shape}")

# ==========================================
# 2. TRAIN / VALIDATION SPLIT (both still 100% normal data)
# ==========================================
# The validation split here is ONLY used to (a) monitor for early stopping and
# (b) determine the anomaly threshold. It never touches the test (collision,
# control, weight, velocity) sets or their labels.
train_seqs, val_seqs = train_test_split(
    all_sequences, test_size=VAL_FRACTION, random_state=42
)

trainX = tensor(train_seqs, dtype=float32)
valX = tensor(val_seqs, dtype=float32)

print(f"Train sequences: {trainX.shape}, Validation sequences: {valX.shape}")


# ==========================================
# 3. MODEL: LSTM-BASED AUTOENCODER (bottlenecked, per instructor's point #4)
# ==========================================
class LSTMModel(nn.Module):
    """
    True bottlenecked sequence autoencoder using 2 LSTMs (encoder + decoder),
    matching the instructor's recommended architecture:

        Input -> LSTM Encoder -> Latent Vector -> Repeat 50x -> LSTM Decoder
        -> Reconstructed Sequence

    The encoder compresses the whole sequence into a single latent vector
    (its final hidden state). The decoder reconstructs the sequence from
    ONLY that vector -- it never sees the original input directly.
    """
    def __init__(self, input_dim, hidden_dim, layer_dim, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        enc_dropout = dropout if layer_dim > 1 else 0.0
        self.encoder = nn.LSTM(input_dim, hidden_dim, layer_dim,
                                batch_first=True, dropout=enc_dropout)
        self.decoder_rnn = nn.LSTM(hidden_dim, hidden_dim, layer_dim,
                                    batch_first=True, dropout=enc_dropout)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # LSTM returns (output, (h_n, c_n)) -- we only need h_n, the hidden state
        _, (h_n, _) = self.encoder(x)   # h_n: (layer_dim, batch, hidden_dim)
        context = h_n[-1]               # final layer's hidden state = latent vector

        # Repeat the latent vector across all time steps to seed the decoder
        decoder_input = context.unsqueeze(1).repeat(1, seq_len, 1)
        decoded_seq, _ = self.decoder_rnn(decoder_input)

        reconstruction = self.output_layer(decoded_seq)
        return reconstruction


# ==========================================
# 4. TRAINING WITH EARLY STOPPING (on validation loss)
# ==========================================
train_loader = DataLoader(TensorDataset(trainX), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(valX), batch_size=BATCH_SIZE, shuffle=False)

model = LSTMModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, layer_dim=LAYER_DIM, dropout=DROPOUT)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_loss = float('inf')
epochs_without_improvement = 0
best_state_dict = None

for epoch in range(MAX_EPOCHS):
    # --- Train ---
    model.train()
    train_loss = 0
    for (batch_X,) in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_X)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)

    # --- Validate ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (batch_X,) in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_X)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f'Epoch [{epoch + 1}/{MAX_EPOCHS}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    # --- Early stopping check ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1} (no val improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

# Restore the best-performing weights on validation data
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)


# ==========================================
# 5. DETERMINE THRESHOLD FROM VALIDATION RECONSTRUCTION ERRORS
#    (NOT from the test set -- this is the leakage fix)
# ==========================================
model.eval()
with torch.no_grad():
    val_reconstruction = model(valX)
    # SCORING FIX: use only the LAST timestep's error, matching how labels
    # are defined (seq_label = final timestep of the window), instead of
    # averaging over the whole 50-step window.
    val_errors = torch.mean((val_reconstruction[:, -1, :] - valX[:, -1, :]) ** 2, dim=1).numpy()

optimal_threshold = np.percentile(val_errors, THRESHOLD_PERCENTILE)
print(f"\nThreshold set from validation errors ({THRESHOLD_PERCENTILE}th percentile): {optimal_threshold:.6f}")
print(f"Validation error stats -- mean: {val_errors.mean():.6f}, std: {val_errors.std():.6f}, "
      f"max: {val_errors.max():.6f}")


# ==========================================
# 6. BUILD TEST SET (collision, as in original) -- untouched by training/thresholding
# ==========================================
DATASET = 'collision'
eval_step_size = 1  # High-resolution detection during evaluation

testX = []
testY = []

collision_data = ds.sets[DATASET]

for recording in collision_data:
    num_timestamps = recording.shape[0]

    if num_timestamps < SEQ_LENGTH:
        continue

    for start_idx in range(0, num_timestamps - SEQ_LENGTH + 1, eval_step_size):
        seq_features = recording[start_idx: start_idx + SEQ_LENGTH, :86]
        seq_label = recording[start_idx + SEQ_LENGTH - 1, -1]

        testX.append(seq_features)
        testY.append(seq_label)

testX = torch.tensor(np.array(testX), dtype=torch.float32)
testY = torch.tensor(np.array(testY), dtype=torch.float32)

print(f"Collision Test Shape: {testX.shape}")  # (N, 50, 86)


# ==========================================
# 7. EVALUATION HELPERS (unchanged from original)
# ==========================================
def plot_run_classification(model, run_data, threshold, seq_length=50):
    model.eval()

    num_timestamps = run_data.shape[0]
    features = run_data[:, :86]
    labels = run_data[:, 86]

    X_windows = []
    Y_labels = []

    for i in range(num_timestamps - seq_length + 1):
        X_windows.append(features[i: i + seq_length])
        Y_labels.append(labels[i + seq_length - 1])

    X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(X_tensor)
        # SCORING FIX: score only the last timestep, matching the label
        errors = torch.mean((reconstructed[:, -1, :] - X_tensor[:, -1, :]) ** 2, dim=1).numpy()

    Y_labels = np.array(Y_labels)
    categories = np.empty(len(errors), dtype=object)

    for i in range(len(errors)):
        if errors[i] > threshold:
            categories[i] = 'TP' if Y_labels[i] == 1 else 'FP'
        else:
            categories[i] = 'TN' if Y_labels[i] == 0 else 'FN'

    plt.figure(figsize=(15, 6))
    color_map = {'TP': 'red', 'FP': 'orange', 'TN': 'blue', 'FN': 'purple'}
    labels_map = {'TP': 'True Positive (Caught)', 'FP': 'False Positive (False Alarm)',
                  'TN': 'True Negative (Normal)', 'FN': 'False Negative (Miss)'}

    t = np.arange(seq_length - 1, num_timestamps)

    for cat in ['TN', 'FP', 'FN', 'TP']:
        mask = (categories == cat)
        if np.any(mask):
            plt.scatter(t[mask], errors[mask], color=color_map[cat], label=labels_map[cat], s=12)

    plt.axhline(y=threshold, color='black', linestyle='--', alpha=0.5, label=f'Threshold ({threshold:.4f})')
    plt.title("Run Analysis: Anomaly Scores Categorized by Performance (LSTM v5)")
    plt.xlabel("Time Step")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.2)
    plt.show()


def print_anomaly_summary(true_labels, anomaly_scores, threshold):
    predictions = (anomaly_scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    kappa = cohen_kappa_score(true_labels, predictions)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print("\n" + "=" * 40)
    print(f"{'ANOMALY DETECTION REPORT (LSTM v5)':^40}")
    print("=" * 40)
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
    print("=" * 40 + "\n")


# ==========================================
# 8. EVALUATE ON TEST SET USING THE VALIDATION-DERIVED THRESHOLD
#    (ROC/PR curves are still plotted for reference/diagnostics -- they
#    do NOT influence the threshold used for the reported metrics below)
# ==========================================
model.eval()
with torch.no_grad():
    reconstruction = model(testX)
    # SCORING FIX: use only the LAST timestep's error, matching the label
    reconstruction_errors = torch.mean((reconstruction[:, -1, :] - testX[:, -1, :]) ** 2, dim=1).numpy()

true_labels = testY.numpy()

# ==========================================
# DIAGNOSTIC 2: VALIDATION vs. TEST ERROR DISTRIBUTION OVERLAY
# ==========================================
# Splits the collision test errors by their OWN label, and overlays them
# against the validation (normal training) error distribution. If the
# test set's NORMAL-labeled windows already sit far to the right of the
# validation distribution, that's evidence of a real distribution shift
# between training conditions and test conditions -- not just a bad
# threshold choice.
test_normal_errors = reconstruction_errors[true_labels == 0]
test_anomalous_errors = reconstruction_errors[true_labels == 1]

plt.figure(figsize=(10, 6))
plt.hist(val_errors, bins=60, alpha=0.5, density=True, label='Validation (normal, held-out)', color='steelblue')
plt.hist(test_normal_errors, bins=60, alpha=0.5, density=True, label='Collision test -- NORMAL-labeled windows', color='green')
plt.hist(test_anomalous_errors, bins=60, alpha=0.5, density=True, label='Collision test -- ANOMALOUS-labeled windows', color='red')
plt.axvline(optimal_threshold, color='black', linestyle='--', label=f'Threshold ({optimal_threshold:.4f})')
plt.xlabel('Reconstruction Error (last timestep, MSE)')
plt.ylabel('Density')
plt.title('Validation vs. Test Error Distributions (LSTM v5)')
plt.legend()
plt.show()

print("\n" + "=" * 50)
print("ERROR DISTRIBUTION DIAGNOSTIC")
print("=" * 50)
print(f"Validation (normal) errors      -- mean: {val_errors.mean():.6f}, median: {np.median(val_errors):.6f}")
print(f"Test NORMAL-labeled errors      -- mean: {test_normal_errors.mean():.6f}, median: {np.median(test_normal_errors):.6f}")
print(f"Test ANOMALOUS-labeled errors   -- mean: {test_anomalous_errors.mean():.6f}, median: {np.median(test_anomalous_errors):.6f}")
print("=" * 50)
print("If 'Test NORMAL-labeled' mean/median is much higher than 'Validation' mean/median,")
print("that indicates a distribution shift between training and test conditions --")
print("the model reconstructs test-set behavior worse than training-set behavior,")
print("REGARDLESS of whether a given window is actually anomalous.")
print("=" * 50 + "\n")


fpr, tpr, thresholds = roc_curve(true_labels, reconstruction_errors)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Analysis (LSTM v5) -- for reference only, not used to set threshold')
plt.legend(loc="lower right")
plt.show()

precision_vals, recall_vals, pr_thresholds = precision_recall_curve(true_labels, reconstruction_errors)
pr_auc = average_precision_score(true_labels, reconstruction_errors)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (LSTM v5)')
plt.legend(loc='lower left')
plt.grid()
plt.show()

# Use the threshold from VALIDATION (step 5), not from the ROC curve above
plot_run_classification(model, ds.sets[DATASET][0], optimal_threshold)
print_anomaly_summary(true_labels, reconstruction_errors, optimal_threshold)


# ==========================================
# 9. F1 ACROSS SUBDATASETS (unchanged from original, uses validation-derived threshold)
# ==========================================
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
            seq_features = recording[start_idx: start_idx + SEQ_LENGTH, :86]
            seq_label = recording[start_idx + SEQ_LENGTH - 1, -1]

            tempX.append(seq_features)
            tempY.append(seq_label)

    tempX = torch.tensor(np.array(tempX), dtype=torch.float32)
    tempY = np.array(tempY)

    with torch.no_grad():
        reconstruction = model(tempX)
        # SCORING FIX: score only the last timestep, matching the label
        errors = torch.mean((reconstruction[:, -1, :] - tempX[:, -1, :]) ** 2, dim=1).numpy()

    preds = (errors > optimal_threshold).astype(int)
    f1 = f1_score(tempY, preds, zero_division=0)
    f1_scores.append(f1)

plt.figure()
plt.bar(datasets, f1_scores)
plt.title("F1 Score Across Subdatasets (LSTM v5)")
plt.xlabel("Dataset")
plt.ylabel("F1 Score")
plt.show()
