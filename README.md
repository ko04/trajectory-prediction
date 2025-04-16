# Trajectory Transformer for Change-Sensitive Prediction

This project implements a Transformer-based model to predict sensitive changes in trajectory data. It uses a sequence of value function scores (`vf_scores`) derived from learned representations (`init_hs`) and final outcomes (`final_reward`) to forecast future steps in a time series. The model is especially designed to emphasize and learn from changes in the trajectory, weighted dynamically using a sensitivity factor.

## 📂 File Overview

- **`change_sensitive.py`**: The main Python script that defines the dataset, Transformer model, training and validation loops, and visualization utilities.
- Model components include:
  - Positional Encoding
  - Transformer Encoder with causal masking
  - Weighted loss based on step-wise changes

## 🚀 Features

- Change-sensitive weighting during training for improved modeling of dynamic sequences
- Teacher-forcing and autoregressive prediction modes
- Visualization of prediction vs. ground-truth trajectory
- Transformer encoder with custom fusion of context vectors and target reward
- Plotting of training/validation loss and per-sample predictions

## How to run the file

Run the python file by calling:

```bash
python change_sensitive.py
