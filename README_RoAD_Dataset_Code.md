# Robotic Arm Anomaly Detection with an LSTM Autoencoder

Code for my preprint: *LSTM-Based Early Anomaly Detection for Industrial Robotic Arm Cybersecurity* (Research Archive of Rising Scholars, Oct 31, 2026). This is a preprint and has not been peer-reviewed.
DOI: [10.58445/x1e5yf88](https://doi.org/10.58445/x1e5yf88) · [Preprint page](https://research-archive.org/index.php/rars/preprint/view/4323/version/4489)

## What it does
Trains an LSTM autoencoder on normal robotic-arm sensor data from the public RoAD dataset and flags time windows it can't reconstruct well as possible anomalies.

## Results and limits
- AUROC 0.71 on the collision subset, slightly above the best previously reported unsupervised result.
- Precision is low, so many alerts are false alarms.
- The weight and velocity subsets are labeled anomalous throughout, so they can't be used to judge detection.

## What I built or learned
[ADD DETAIL: windowing, threshold choice, evaluation in Google Colab]

## How to run it
1. Get the RoAD dataset: [ADD DETAIL: link to the dataset's official source]
2. `pip install -r requirements.txt` [ADD DETAIL: create this file]
3. [ADD DETAIL: notebook or script name]

## Screenshot
![ROC curve](docs/roc_curve.png) <!-- [ADD DETAIL] -->

## Cite
Raj, J. (2026). LSTM-Based Early Anomaly Detection for Industrial Robotic Arm Cybersecurity [Preprint]. Research Archive of Rising Scholars. https://doi.org/10.58445/x1e5yf88

---
Made by [Joanna Raj](https://github.com/rajjoan) · [rajjoan.github.io](https://rajjoan.github.io)
