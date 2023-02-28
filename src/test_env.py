
import os 
import sys

import numpy as np
import torch

# Add the parent directory to the path
try:    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except: pass

# %cd src

from utils import plot_roc_curve

plot_roc_curve([0, 1], [0, 1])

def calculate_recall(y_true, y_pred):
    """Calculate recall given the true and predicted labels
    """
    epsilon = 1e-7
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall

y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 2, 1, 0, 0, 1])
calculate_recall(y_true, y_pred)

X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# if __name__ == '__main__':
#     print('hello world')