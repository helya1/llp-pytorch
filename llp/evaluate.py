"""Instance-level evaluation of a trained model."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


@torch.no_grad()
def evaluate(model, features: np.ndarray, y: np.ndarray, device=None) -> dict:
    """Instance-level accuracy and AUC on held-out data with true labels."""
    device = device or next(model.parameters()).device
    model.eval()
    prob = torch.sigmoid(model(torch.as_tensor(features, device=device))).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(y, prob > 0.5)),
        "auc": float(roc_auc_score(y, prob)),
    }
