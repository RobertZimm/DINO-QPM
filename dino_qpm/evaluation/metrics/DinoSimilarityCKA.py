import numpy as np
import torch
from dino_qpm.architectures.qpm_dino.similarity_functions import compute_similarity


def _sample_indices(n_total_samples: int, k: int) -> np.ndarray:
    """
    Helper function to generate sample indices for bootstrapping/sampling.

    Args:
        n_total_samples: Total number of samples available.
        k: Number of samples to draw.

    Returns:
        Array of indices to sample. If k >= n_total_samples, returns all indices.
        Otherwise, returns k randomly selected indices without replacement.
    """
    if k >= n_total_samples:
        return np.arange(n_total_samples)
    else:
        return np.random.choice(n_total_samples, k, replace=False)


def _center_columns(X: np.ndarray, ret_mean: bool = False) -> np.ndarray:
    """
    Centers the columns of a matrix X by subtracting the column-wise mean.

    Args:
        X: A numpy array of shape (n_samples, n_features).

    Returns:
        A numpy array of the same shape as X, with column means subtracted.
    """
    # Calculate mean across samples (axis 0)
    col_mean = X.mean(axis=0, keepdims=True)

    if ret_mean:
        return X - col_mean, col_mean

    return X - col_mean


def _compute_cka_from_samples(X_sample: np.ndarray, Y_sample: np.ndarray) -> float:
    """
    Internal helper function to compute CKA score from already sampled matrices.
    Uses the Primal CKA formulation.

    Args:
        X_sample: The first sampled representation matrix, shape (k, d1).
        Y_sample: The second sampled representation matrix, shape (k, d2).

    Returns:
        The CKA similarity score for this sample pair.
    """
    # Center the features (columns) of the sampled matrices
    X_c = _center_columns(X_sample)
    Y_c = _center_columns(Y_sample)

    # Calculate the covariance matrices (d x d)
    # Time complexity: O(k * d^2)
    c_xx = X_c.T @ X_c
    c_yy = Y_c.T @ Y_c
    c_xy = X_c.T @ Y_c  # Corrected cross-covariance calculation

    # Calculate the Frobenius norms of the covariance matrices
    c_xx_norm = np.linalg.norm(c_xx, 'fro')
    c_yy_norm = np.linalg.norm(c_yy, 'fro')
    c_xy_norm_sq = np.linalg.norm(c_xy, 'fro')**2

    # --- Final CKA Score ---
    denominator = c_xx_norm * c_yy_norm

    # Handle potential division by zero
    if denominator < 1e-10:
        return 1.0 if c_xy_norm_sq < 1e-10 else 0.0

    return c_xy_norm_sq / denominator


def compute_pearson_corr(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes the Pearson correlation coefficient between two vectors.

    Args:
        X: First vector (n,).
        Y: Second vector (n,).

    Returns:
        Pearson correlation coefficient (float).
    """
    X_mean = X.mean()
    Y_mean = Y.mean()

    numerator = np.sum((X - X_mean) * (Y - Y_mean))
    denominator = np.sqrt(np.sum((X - X_mean) ** 2)
                          * np.sum((Y - Y_mean) ** 2))

    if denominator < 1e-10:
        return 0.0

    return numerator / denominator


def compute_proto_consistency(X: np.ndarray, Y: np.ndarray, prototype_X: np.ndarray, prototype_Y: np.ndarray,
                              similarity_method: str = 'cosine', eps: float = 1e-10, gamma: float = 1e-3) -> float:
    """
    Computes pearson correlation between vectors of similarities between prototypes and their respective sample space.
    Prototype_x and prototype_y are themselves vectors of shape (d1,) and (d2,) respectively.
    Uses the specified similarity_method for computing similarities.
    """
    # Convert to torch tensors
    X_torch = torch.from_numpy(X).float()
    Y_torch = torch.from_numpy(Y).float()
    prototype_X_torch = torch.from_numpy(prototype_X).float()
    prototype_Y_torch = torch.from_numpy(prototype_Y).float()
    
    # Reshape for compute_similarity: (n_samples, 1, embed_dim)
    X_reshaped = X_torch.unsqueeze(1)
    Y_reshaped = Y_torch.unsqueeze(1)

    # Reshape prototypes: (1, embed_dim) -> no reshaping needed, already correct shape

    # Compute similarity vectors using the configured method
    # Result shape: (1, n_samples) -> we take [0] to get (n_samples,)
    sim_X = compute_similarity(X_reshaped, prototype_X_torch.unsqueeze(0),
                               similarity_method=similarity_method, gamma=gamma)[0].numpy()
    sim_Y = compute_similarity(Y_reshaped, prototype_Y_torch.unsqueeze(0),
                               similarity_method=similarity_method, gamma=gamma)[0].numpy()

    return compute_pearson_corr(sim_X, sim_Y)


def sampled_proto_consistency(X: np.ndarray, Y: np.ndarray, prototype_X: np.ndarray, prototype_Y: np.ndarray,
                              k: int, similarity_method: str = 'cosine', gamma: float = 1e-3) -> float:
    """
    Calculates the prototype consistency using a random sample of k data points.

    Args:
        X: The first representation matrix, shape (n_total_samples, d1).
        Y: The second representation matrix, shape (n_total_samples, d2).
        prototype_X: Prototype vector in X-space, shape (d1,).
        prototype_Y: Prototype vector in Y-space, shape (d2,).
        k: The number of samples to draw for the consistency estimation.
        similarity_method: Method to use for similarity computation ('cosine', 'log_l2', 'rbf')
        gamma: Gamma parameter for rbf method

    Returns:
        The prototype consistency score (Pearson correlation coefficient).
    """
    n_total_samples = X.shape[0]

    # --- 1. Sampling ---
    indices = _sample_indices(n_total_samples, k)
    X_sample = X[indices]
    Y_sample = Y[indices]

    # --- 2. Compute prototype consistency ---
    return compute_proto_consistency(X_sample, Y_sample, prototype_X, prototype_Y,
                                     similarity_method=similarity_method, gamma=gamma)


def bootstrapped_sampled_proto_consistency(X: np.ndarray, Y: np.ndarray, prototype_X: np.ndarray, prototype_Y: np.ndarray,
                                           k: int, num_runs: int = 10, similarity_method: str = 'cosine', gamma: float = 1e-3) -> tuple[float, float]:
    """
    Calculates the prototype consistency using bootstrapping over multiple runs.

    Repeats the sampling and consistency calculation `num_runs` times and returns
    the mean and standard deviation of the resulting consistency scores.

    Args:
        X: The first representation matrix, shape (n_total_samples, d1).
        Y: The second representation matrix, shape (n_total_samples, d2).
        prototype_X: Prototype vector in X-space, shape (d1,).
        prototype_Y: Prototype vector in Y-space, shape (d2,).
        k: The number of samples to draw in each run.
        num_runs: The number of bootstrap runs to perform.
        similarity_method: Method to use for similarity computation ('cosine', 'log_l2', 'rbf')
        gamma: Gamma parameter for rbf method

    Returns:
        A tuple containing:
            - mean_consistency (float): The average consistency score across all runs.
            - std_consistency (float): The standard deviation of consistency scores across all runs.
    """
    n_total_samples = X.shape[0]
    consistency_scores = []

    for _ in range(num_runs):
        # --- 1. Sampling ---
        indices = _sample_indices(n_total_samples, k)
        X_sample = X[indices]
        Y_sample = Y[indices]

        # --- 2. Compute consistency for this run ---
        consistency_run = compute_proto_consistency(
            X_sample, Y_sample, prototype_X, prototype_Y,
            similarity_method=similarity_method, gamma=gamma)
        consistency_scores.append(consistency_run)

    # --- 3. Calculate Mean and Std Dev ---
    mean_consistency = np.mean(consistency_scores).item()
    std_consistency = np.std(consistency_scores).item()

    return mean_consistency, std_consistency


def bootstrapped_sampled_linear_cka(X: np.ndarray, Y: np.ndarray, k: int, num_runs: int = 10) -> tuple[float, float]:
    """
    Calculates the Sampled Linear Centered Kernel Alignment (CKA) using
    bootstrapping over multiple runs.

    Repeats the sampling and CKA calculation `num_runs` times and returns
    the mean and standard deviation of the resulting CKA scores.

    Args:
        X: The first representation matrix, shape (n_total_samples, d1).
        Y: The second representation matrix, shape (n_total_samples, d2).
        k: The number of samples to draw in each run.
        num_runs: The number of bootstrap runs to perform.

    Returns:
        A tuple containing:
            - mean_cka (float): The average CKA score across all runs.
            - std_cka (float): The standard deviation of CKA scores across all runs.
    """
    n_total_samples = X.shape[0]
    cka_scores = []

    for _ in range(num_runs):
        # --- 1. Sampling ---
        indices = _sample_indices(n_total_samples, k)
        X_sample = X[indices]
        Y_sample = Y[indices]

        # --- 2. Compute CKA for this run ---
        cka_run = _compute_cka_from_samples(X_sample, Y_sample)
        cka_scores.append(cka_run)

    # --- 3. Calculate Mean and Std Dev ---
    mean_cka = np.mean(cka_scores).item()
    std_cka = np.std(cka_scores).item()

    return mean_cka, std_cka


# --- Keep the original single-run function for comparison or direct use ---
def sampled_linear_cka(X: np.ndarray, Y: np.ndarray, k: int) -> float:
    """
    Calculates the Sampled Linear Centered Kernel Alignment (CKA) between
    two representation matrices X and Y for a single run.

    (Same implementation details as before)

    Args:
        X: The first representation matrix, shape (n_total_samples, d1).
        Y: The second representation matrix, shape (n_total_samples, d2).
        k: The number of samples to draw for the CKA estimation.

    Returns:
        The CKA similarity score (a float between 0 and 1).
    """
    n_total_samples = X.shape[0]

    # --- 1. Sampling ---
    indices = _sample_indices(n_total_samples, k)
    X_sample = X[indices]
    Y_sample = Y[indices]

    # --- 2. Compute CKA ---
    return _compute_cka_from_samples(X_sample, Y_sample)


if __name__ == "__main__":
    # --- Demonstration ---
    N_TOTAL_SAMPLES = 100_000
    D1_FEATURES = 512
    D2_FEATURES = 768
    K_SAMPLES = 5_000
    NUM_BOOTSTRAP_RUNS = 20  # Number of times to repeat sampling

    print(f"Creating dummy data...")
    print(f"Total samples n = {N_TOTAL_SAMPLES}")
    print(f"Features d1 = {D1_FEATURES}, d2 = {D2_FEATURES}")
    print(f"Sample size k = {K_SAMPLES}")
    print(f"Bootstrap runs = {NUM_BOOTSTRAP_RUNS}")

    X = np.random.randn(N_TOTAL_SAMPLES, D1_FEATURES)

    # Case 1: Y is a noisy copy of X
    Y_similar = X[:, :D2_FEATURES] if D1_FEATURES > D2_FEATURES else X
    if D2_FEATURES > D1_FEATURES:
        Y_similar = np.hstack(
            [X, np.random.randn(N_TOTAL_SAMPLES, D2_FEATURES - D1_FEATURES)])
    Y_similar = Y_similar + np.random.randn(N_TOTAL_SAMPLES, D2_FEATURES) * 0.1

    # Case 2: Y is completely unrelated
    Y_random = np.random.randn(N_TOTAL_SAMPLES, D2_FEATURES)

    # 2. Run the CKA calculations with bootstrapping
    print("\nCalculating bootstrapped CKA...")

    # Test 1: Similarity with itself
    mean_cka_self, std_cka_self = bootstrapped_sampled_linear_cka(
        X, X, K_SAMPLES, NUM_BOOTSTRAP_RUNS
    )
    print(
        f"  CKA(X, X):         Mean = {mean_cka_self:.6f}, Std Dev = {std_cka_self:.6f}")

    # Test 2: Similarity with a noisy copy
    mean_cka_similar, std_cka_similar = bootstrapped_sampled_linear_cka(
        X, Y_similar, K_SAMPLES, NUM_BOOTSTRAP_RUNS
    )
    print(
        f"  CKA(X, Y_similar): Mean = {mean_cka_similar:.6f}, Std Dev = {std_cka_similar:.6f}")

    # Test 3: Similarity with random data
    mean_cka_random, std_cka_random = bootstrapped_sampled_linear_cka(
        X, Y_random, K_SAMPLES, NUM_BOOTSTRAP_RUNS
    )
    print(
        f"  CKA(X, Y_random):  Mean = {mean_cka_random:.6f}, Std Dev = {std_cka_random:.6f}")

    # Optional: Run the single version for comparison
    # print("\nSingle run for comparison:")
    # cka_self_single = sampled_linear_cka(X, X, K_SAMPLES)
    # print(f"  Single CKA(X, X): {cka_self_single:.6f}")

    # --- Test Prototype Consistency ---
    print("\n" + "="*60)
    print("TESTING PROTOTYPE CONSISTENCY")
    print("="*60)

    # Create prototype vectors
    # Prototype 1: A "concept" in X-space (e.g., average of first 100 samples)
    prototype_X = X[:100].mean(axis=0)

    # Case 1: Corresponding prototype in Y-space (should have high consistency)
    prototype_Y_similar = Y_similar[:100].mean(axis=0)

    # Case 2: Random prototype in Y-space (should have low consistency)
    prototype_Y_random = np.random.randn(D2_FEATURES)

    print(f"\nPrototype vectors created:")
    print(f"  prototype_X shape: {prototype_X.shape}")
    print(f"  prototype_Y_similar shape: {prototype_Y_similar.shape}")
    print(f"  prototype_Y_random shape: {prototype_Y_random.shape}")

    # Test 1: Full dataset consistency (baseline)
    print("\n--- Full Dataset Prototype Consistency ---")
    consistency_similar_full = compute_proto_consistency(
        X, Y_similar, prototype_X, prototype_Y_similar)
    consistency_random_full = compute_proto_consistency(
        X, Y_random, prototype_X, prototype_Y_random)
    print(
        f"  Consistency(X, Y_similar) with corresponding prototypes: {consistency_similar_full:.6f}")
    print(
        f"  Consistency(X, Y_random) with random prototype:         {consistency_random_full:.6f}")

    # Test 2: Sampled consistency (single run)
    print("\n--- Sampled Prototype Consistency (single run) ---")
    consistency_similar_sampled = sampled_proto_consistency(
        X, Y_similar, prototype_X, prototype_Y_similar, K_SAMPLES)
    consistency_random_sampled = sampled_proto_consistency(
        X, Y_random, prototype_X, prototype_Y_random, K_SAMPLES)
    print(
        f"  Sampled Consistency(X, Y_similar): {consistency_similar_sampled:.6f}")
    print(
        f"  Sampled Consistency(X, Y_random):  {consistency_random_sampled:.6f}")

    # Test 3: Bootstrapped sampled consistency
    print("\n--- Bootstrapped Sampled Prototype Consistency ---")
    mean_cons_similar, std_cons_similar = bootstrapped_sampled_proto_consistency(
        X, Y_similar, prototype_X, prototype_Y_similar, K_SAMPLES, NUM_BOOTSTRAP_RUNS
    )
    mean_cons_random, std_cons_random = bootstrapped_sampled_proto_consistency(
        X, Y_random, prototype_X, prototype_Y_random, K_SAMPLES, NUM_BOOTSTRAP_RUNS
    )
    print(
        f"  Consistency(X, Y_similar): Mean = {mean_cons_similar:.6f}, Std Dev = {std_cons_similar:.6f}")
    print(
        f"  Consistency(X, Y_random):  Mean = {mean_cons_random:.6f}, Std Dev = {std_cons_random:.6f}")

    # Test 4: Self-consistency (should be perfect = 1.0)
    print("\n--- Self-Consistency Test ---")
    mean_cons_self, std_cons_self = bootstrapped_sampled_proto_consistency(
        X, X, prototype_X, prototype_X, K_SAMPLES, NUM_BOOTSTRAP_RUNS
    )
    print(
        f"  Self-Consistency(X, X, same prototype): Mean = {mean_cons_self:.6f}, Std Dev = {std_cons_self:.6f}")
    print(f"  (Should be close to 1.0)")

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
