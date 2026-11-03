"""Regression checks for blockwise IV and direct moment reductions."""
import numpy as np
import pytest
from scipy.linalg import block_diag
from pyblp.utilities.statistics import IV, compute_gmm_moments_mean, compute_gmm_moments_jacobian_mean


@pytest.mark.parametrize('split', [27, 50, 65])
@pytest.mark.parametrize('convert', [True, False])
def test_block_iv_matches_dense_reference(split, convert):
    rng = np.random.default_rng(170)
    N = 90
    Z_list = [rng.normal(size=(40, 3)), rng.normal(size=(50, 2))]
    X = rng.normal(size=(N, 3))
    X_list = [X[:split], X[split:]]
    y_list = [rng.normal(size=(40, 1)), rng.normal(size=(50, 1))]
    j_list = [rng.normal(size=(40, 4)), rng.normal(size=(50, 4))]
    A = rng.normal(size=(5, 5))
    W = A.T @ A + np.eye(5)
    Z = block_diag(*Z_list)
    cov = np.linalg.pinv(X.T @ Z @ W @ Z.T @ X)
    proj = cov @ X.T @ Z @ W @ Z.T
    y = np.vstack(y_list)
    old_beta = proj @ y
    old_u = y - X @ old_beta
    old_grad = np.vstack(j_list) - X @ (proj @ np.vstack(j_list))
    implementation = IV(X_list, Z_list, W)
    beta, us, grads = implementation.estimate(X_list, Z_list, W, y_list, j_list, convert)
    np.testing.assert_allclose(beta, old_beta, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(np.vstack(us), old_u, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(
        np.vstack(grads), old_grad if convert else np.vstack(j_list), rtol=3e-11, atol=3e-11,
    )
    fixed_beta, fixed_u, _ = implementation.estimate(
        X_list, Z_list, W, y_list, j_list, False, fixed_parameters=old_beta,
    )
    np.testing.assert_allclose(fixed_beta, old_beta, rtol=0, atol=0)
    np.testing.assert_allclose(np.vstack(fixed_u), old_u, rtol=3e-11, atol=3e-11)


@pytest.mark.parametrize('n,k,p', [(50, 8, 4), (1000, 30, 12), (500, 0, 2)])
def test_direct_gmm_moment_means(n, k, p):
    rng = np.random.default_rng(n + k + p)
    u = rng.normal(size=(n, 1))
    Z = rng.normal(size=(n, k))
    jac = rng.normal(size=(n, p))
    moment = compute_gmm_moments_mean([u], [Z])
    grad = compute_gmm_moments_jacobian_mean([jac], [Z])
    np.testing.assert_allclose(moment, (u * Z).mean(axis=0)[:, None], rtol=5e-13, atol=5e-13)
    np.testing.assert_allclose(
        grad, (jac[:, None, :] * Z[:, :, None]).mean(axis=0), rtol=5e-13, atol=5e-13,
    )
