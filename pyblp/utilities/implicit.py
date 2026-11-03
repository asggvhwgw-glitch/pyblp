"""Experimental matrix-free implicit-gradient solvers for ordinary RC logit.

This backend is opt-in and never changes the default dense linear solver.
A graph-aggregated preconditioner is attempted only for large, multi-RHS
problems with sufficiently homogeneous agent groups. It is *not* a
Graph-Newton share-inversion implementation.
"""
from typing import Optional, Tuple
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def _graph_inverse(probabilities: np.ndarray, weights: np.ndarray, shares: np.ndarray,
                   minimum_ratio: float = 0.35):
    """Return an inverse coarse-graph preconditioner, or None.

    With grouped agent types, M = D - B B.T where B contains weighted
    group-average probabilities. Woodbury reduces each apply to K columns.
    This is a preconditioner, not a replacement for the true Hessian.
    """
    J, I = probabilities.shape
    if I < 8:
        return None
    outside = 1 - probabilities.sum(axis=0)
    if np.any(outside <= 1e-10):
        return None
    k = min(8, I)
    sampled = np.unique(np.linspace(0, J - 1, min(48, J), dtype=int))
    features = probabilities[sampled].T
    center_ids = [int(np.argmax(np.einsum('ij,ij->i', features, features)))]
    distances = np.full(I, np.inf)
    for _ in range(k - 1):
        squared = np.sum((features - features[center_ids[-1]])**2, axis=1)
        distances = np.minimum(distances, squared)
        center_ids.append(int(np.argmax(distances)))
    centers = features[center_ids]
    groups = np.argmin(np.sum((features[:, None, :] - centers[None])**2, axis=2), axis=1)

    columns = []
    for g in range(k):
        selected = groups == g
        if not np.any(selected):
            continue
        mass = weights[selected].sum()
        if mass <= 0:
            continue
        mean = probabilities[:, selected] @ weights[selected] / mass
        mean_outside = outside[selected] @ weights[selected] / mass
        if np.any(probabilities[:, selected] < minimum_ratio * mean[:, None]):
            return None
        if np.any(outside[selected] < minimum_ratio * mean_outside):
            return None
        columns.append(np.sqrt(mass) * mean)
    if not columns:
        return None
    B = np.column_stack(columns)
    core = np.eye(B.shape[1]) - (B.T / shares) @ B
    try:
        chol = np.linalg.cholesky(core)
    except np.linalg.LinAlgError:
        return None

    def inverse(v):
        diagonal_part = v / shares
        projection = B.T @ diagonal_part
        correction = np.linalg.solve(chol.T, np.linalg.solve(chol, projection))
        return diagonal_part + B @ correction / shares

    return inverse


def solve_implicit_derivatives(
    probabilities: np.ndarray,
    weights: np.ndarray,
    observed_shares: np.ndarray,
    right_hand_side: np.ndarray,
    mode: str = 'auto',
    rtol: float = 1e-10,
) -> Tuple[Optional[np.ndarray], str]:
    """Solve (D - P diag(w) P.T) X = rhs without forming a J by J matrix.

    Return (None, diagnostic) to request a dense fallback when assumptions or
    a posteriori accuracy checks fail.
    """
    if mode not in {'auto', 'cg', 'hybrid'}:
        raise ValueError("mode must be 'auto', 'cg', or 'hybrid'.")
    P = np.asarray(probabilities)
    w = np.asarray(weights).reshape(-1)
    s = np.asarray(observed_shares).reshape(-1)
    rhs = np.asarray(right_hand_side)
    J, I = P.shape
    if rhs.ndim != 2 or rhs.shape[0] != J or w.size != I or s.size != J:
        return None, 'incompatible shapes'
    if rhs.shape[1] == 0:
        return np.empty_like(rhs), 'zero right-hand sides'
    if mode == 'auto' and J < 256:
        return None, 'small market: dense solver preferred'
    if (P.dtype != np.dtype('float64') or w.dtype != np.dtype('float64')
            or rhs.dtype != np.dtype('float64')):
        return None, 'dtype not supported by experimental backend'
    if (not np.isfinite(P).all() or not np.isfinite(rhs).all()
            or not np.isfinite(w).all() or not np.isfinite(s).all()
            or np.any(w < 0) or not np.any(w > 0)
            or np.any(s <= 0) or s.sum() >= 1):
        return None, 'signed weights, nonfinite data, or invalid shares'
    if abs(w.sum() - 1.0) > 1e-8:
        return None, 'integration weights not normalized'
    if np.max(np.abs(P @ w - s)) > 1e-8:
        return None, 'inversion has not converged to the observed shares'

    def matvec(v):
        return s * v - P @ (w * (P.T @ v))

    A = LinearOperator((J, J), matvec=matvec, dtype=np.float64)
    outside = 1 - s.sum()
    if outside <= 0:
        return None, 'outside share not positive'

    def basic_inverse(v):
        return v / s + v.sum() / outside

    inverse = basic_inverse
    selected = 'matrix-free CG with outside-share preconditioner'
    if mode in {'auto', 'hybrid'} and J >= 384 and rhs.shape[1] >= 8:
        graph = _graph_inverse(P, w, s)
        if graph is not None:
            inverse = graph
            selected = 'matrix-free CG with grouped-graph preconditioner'
    M = LinearOperator((J, J), matvec=inverse, dtype=np.float64)
    columns = []
    maxiter = max(100, min(4 * J, 5000))
    for j in range(rhs.shape[1]):
        b = rhs[:, j]
        if not np.any(b):
            columns.append(np.zeros(J))
            continue
        try:
            solution, info = cg(A, b, M=M, atol=0.0, rtol=rtol, maxiter=maxiter)
        except TypeError:
            # SciPy versions predating the rtol/atol split.
            solution, info = cg(A, b, M=M, tol=rtol, maxiter=maxiter)
        if info != 0 or not np.isfinite(solution).all():
            return None, selected + ': CG failure'
        true_error = np.max(np.abs(matvec(solution) - b))
        if true_error > 5 * rtol * max(np.max(np.abs(b)), 1e-12):
            return None, selected + ': a posteriori residual too large'
        columns.append(solution)
    return np.column_stack(columns), selected
