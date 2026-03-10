import numpy as np
from scipy import sparse

def _normalized_laplacian(W):
    """
    Computes the symmetric normalized Laplacian matrix.
    """
    # 1. Compute node degrees (sum of weights for each node)
    # W is symmetric here because of W.maximum(W.T) in the main function
    d = np.array(W.sum(axis=1)).flatten()
    
    # 2. Compute D^{-1/2}, handling isolated nodes (degree 0) safely
    d_inv_sqrt = np.zeros_like(d)
    d_inv_sqrt[d > 0] = 1.0 / np.sqrt(d[d > 0])
    
    # Create a sparse diagonal matrix for D^{-1/2}
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    
    # 3. Compute D^{-1/2} W D^{-1/2}
    normalized_W = D_inv_sqrt @ W @ D_inv_sqrt
    
    # 4. L = I - D^{-1/2} W D^{-1/2}
    I = sparse.eye(W.shape[0], format=W.format)
    L = I - normalized_W
    
    return L

def _sparse_rayleigh(L, F):
    """
    Computes the Rayleigh quotient for each column feature in F.
    """
    # Numerator: f^T L f for each column
    # L @ F does fast sparse-dense matrix multiplication resulting in shape (n_cells, n_features)
    # Multiplying by F element-wise and summing down the columns gives the dot product
    numerator = np.sum(F * (L @ F), axis=0)
    
    # Denominator: f^T f for each column
    denominator = np.sum(F**2, axis=0)
    
    # Compute quotient, guarding against division by zero 
    # (Though your main function's standardization step mostly prevents this)
    with np.errstate(divide='ignore', invalid='ignore'):
        rayleigh = numerator / denominator
        # If any features were all zeros, default their quotient to 0
        rayleigh = np.nan_to_num(rayleigh, nan=0.0)
        
    return rayleigh

def signal_smoothness(adjacency, features, normalize_features=True):
    """Measure how smoothly a set of features varies across a graph.

    Given a graph (as an adjacency matrix) and a matrix of node features,
    this function quantifies whether neighboring nodes tend to have similar
    feature values.  The core quantity is the **Rayleigh quotient** of each
    feature column on the normalized graph Laplacian:

        R(f) = f^T L f  /  f^T f

    where L = I - D^{-1/2} W D^{-1/2} is the symmetric normalized Laplacian,
    W is the (symmetrized) weight matrix derived from ``adjacency``, and D is
    the diagonal degree matrix.

    Interpretation
    \--------------
    * R(f) is bounded in [0, 2] for the normalized Laplacian.
    * R(f) = 0 means the feature is perfectly constant across every connected
      component (maximally smooth).
    * R(f) close to 2 means the feature oscillates maximally between neighbors
      (anti-correlated across edges).
    * For a random signal on a connected graph, the expected Rayleigh quotient
      is approximately 1 (the mean eigenvalue of L).

    A good niche-detection or clustering method should produce regions whose
    internal spatial graph yields low Rayleigh quotients for biologically
    relevant features (e.g. gene expression), because cells that belong to
    the same microenvironment share similar transcriptional programs.

    Parameters
    \----------
    adjacency : array-like or scipy.sparse matrix, shape (n_cells, n_cells)
        Adjacency (or weight) matrix of the graph.  Can be dense or sparse.
        It does not need to be symmetric; the function symmetrizes it
        internally.  Nonzero entries are treated as edge weights.
    features : array-like, shape (n_cells, n_features)
        Feature matrix whose columns are the signals to evaluate (e.g. gene
        expression values, one column per gene).
    normalize_features : bool, default True
        If True, each feature column is zero-centered and scaled to unit
        variance before computing the Rayleigh quotient.  This prevents
        features with large absolute values from dominating summary
        statistics and makes quotients comparable across features with
        different scales.

    Returns
    \-------
    dict with the following keys:

        ``per_feature`` : np.ndarray, shape (n_features,)
            Rayleigh quotient for each feature column.
        ``mean`` : float
            Mean Rayleigh quotient across all features.
        ``median`` : float
            Median Rayleigh quotient across all features.
        ``std`` : float
            Standard deviation of Rayleigh quotients across features.
        ``n_features`` : int
            Number of feature columns evaluated.
        ``n_cells`` : int
            Number of cells (nodes) in the graph.
        ``random_baseline`` : float
            Expected Rayleigh quotient of random Gaussian signals on the
            same graph (computed empirically).  Useful as a reference:
            values well below this indicate genuine smoothness, values
            near or above it indicate the features are no smoother than
            noise on this graph.

    Examples
    \--------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> # 4 nodes in a line: 0-1-2-3
    >>> row = [0, 1, 1, 2, 2, 3]
    >>> col = [1, 0, 2, 1, 3, 2]
    >>> A = csr_matrix((np.ones(6), (row, col)), shape=(4, 4))
    >>> # smooth signal: values increase along the chain
    >>> F = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> result = signal_smoothness(A, F)
    >>> result['mean'] < 0.5   # should be quite smooth
    True

    >>> # noisy signal: alternating high/low
    >>> F_noisy = np.array([[10.0], [0.0], [10.0], [0.0]])
    >>> result_noisy = signal_smoothness(A, F_noisy)
    >>> result_noisy['mean'] > result['mean']   # noisier => higher quotient
    True
    """
    from scipy.sparse import issparse, csr_matrix as sp_csr
    # from ._gsp import _normalized_laplacian
    import numpy as np

    F = np.asarray(features, dtype=np.float64)
    if F.ndim == 1:
        F = F[:, None]
    n, d = F.shape

    if issparse(adjacency):
        W = adjacency.copy()
    else:
        W = sp_csr(np.asarray(adjacency, dtype=np.float64))
    W = W.maximum(W.T)

    L = _normalized_laplacian(W)

    if normalize_features:
        mu = F.mean(axis=0, keepdims=True)
        sigma = F.std(axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        F = (F - mu) / sigma

    per_feat = _sparse_rayleigh(L.tocsr(), F)

    rng = np.random.RandomState(0)
    rand_F = rng.randn(n, max(d, 20))
    rand_rq = _sparse_rayleigh(L.tocsr(), rand_F)
    rand_baseline = float(np.mean(rand_rq))

    return {
        'per_feature': per_feat,
        'mean': float(np.mean(per_feat)),
        'median': float(np.median(per_feat)),
        'std': float(np.std(per_feat)),
        'n_features': d,
        'n_cells': n,
        'random_baseline': rand_baseline,
    }