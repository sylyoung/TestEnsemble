import numpy as np

def estimate_binary(binary_preds):
    """
    Estimates the class imbalance (b) and sensitivity for each classifier in a binary problem.
    Input: binary_preds (m, n) with entries 1 and -1.
    Output: b (class imbalance), v (eigenvector), mu (mean predictions)
    """
    m, n = binary_preds.shape

    mu = np.mean(binary_preds, axis=1)
    deviations = binary_preds - mu[:, np.newaxis]
    # Calculate the covariance matrix
    Q = np.dot(deviations, deviations.T) / (binary_preds.shape[1] - 1)
    # Principal eigenvector
    v = np.linalg.eig(Q)[1][:, 0]

    if v[0] < 0:
        v = -v

    # Compute third-order covariance tensor using einsum for efficiency
    centered = binary_preds - mu[:, None]
    T = np.einsum('ia,ja,ka->ijk', centered, centered, centered) / n

    # Extract triples where i < j < k to form equations T_ijk = alpha * v_i v_j v_k
    triples = [(i, j, k) for i in range(m) for j in range(i + 1, m) for k in range(j + 1, m)]
    t_values = np.array([T[i, j, k] for i, j, k in triples])
    v_values = np.array([v[i] * v[j] * v[k] for i, j, k in triples])

    # Solve for alpha using least squares
    if len(t_values) == 0:
        alpha = 0.0
    else:
        alpha = np.dot(t_values, v_values) / np.dot(v_values, v_values)

    # Compute class imbalance b from alpha
    alpha = np.clip(alpha, -1e8, 1e8)  # Avoid overflow
    b = -alpha / np.sqrt(4 + alpha ** 2) if alpha != 0 else 0.0
    b = np.clip(b, -1 + 1e-5, 1 - 1e-5)  # Ensure valid class probability

    return b, v, mu


def estimate_multiclass(preds, true):
    """
    Implements Section 5: Multi-class case via one-vs-all binary reductions.
    Input: preds (num_classifiers, num_samples) with class labels 0, ..., K-1.
    Output: p (class probabilities), confusion_diagonals (diagonal entries of confusion matrices)
    """
    m, n = preds.shape
    classes = np.unique(true)
    K = len(classes)
    p = np.zeros(K)
    confusion_diagonals = np.zeros((m, K))

    for k_idx, k in enumerate(classes):
        # Create binary predictions: 1 if class k, else -1
        binary_preds = np.where(preds == k, 1, -1)

        # Estimate binary parameters
        b, v, mu = estimate_binary(binary_preds)

        # Compute class probability p_k
        p_k = (b + 1) / 2
        p[k_idx] = p_k

        # Compute sensitivities (diagonal entries) using Eq. (5)
        sqrt_term = np.sqrt((1 - b) / (1 + b))
        psi_k = 0.5 * (1 + mu + v * sqrt_term)
        confusion_diagonals[:, k_idx] = np.clip(psi_k, 0, 1)  # Ensure valid probabilities

    # Normalize class probabilities to sum to 1
    p /= p.sum()

    return p, confusion_diagonals