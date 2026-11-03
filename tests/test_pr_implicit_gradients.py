"""Experimental implicit-gradient kernels and an end-to-end opt-in smoke test."""
import numpy as np
import pytest
import pyblp
from pyblp.utilities.implicit import solve_implicit_derivatives


@pytest.mark.parametrize('J,I,K', [(48, 90, 1), (128, 200, 6), (512, 128, 12)])
def test_matrixfree_matches_dense(J, I, K):
    rng = np.random.default_rng(J + I + K)
    w = np.full(I, 1 / I)
    u = rng.normal(loc=-5, scale=0.7, size=(J, I))
    u -= np.max(u, axis=0, keepdims=True)
    p = np.exp(u)
    p /= np.exp(-np.max(rng.normal(loc=-5, scale=0.7, size=(J, I)), axis=0)) + p.sum(axis=0)
    # Re-normalize with a controlled outside alternative; this keeps H SPD.
    p *= 0.75 / p.sum(axis=0, keepdims=True)
    shares = p @ w
    rhs = rng.normal(size=(J, K))
    expected = np.linalg.solve(np.diag(shares) - (p * w) @ p.T, rhs)
    computed, status = solve_implicit_derivatives(p, w, shares, rhs, mode='hybrid', rtol=1e-10)
    assert computed is not None, status
    np.testing.assert_allclose(computed, expected, rtol=5e-7, atol=5e-8)


def test_signed_quadrature_falls_back():
    p = np.full((10, 5), 0.01)
    w = np.full(5, 0.2)
    w[0] = -0.2
    result, diagnostic = solve_implicit_derivatives(
        p, w, np.full(10, 0.01), np.ones((10, 2)), mode='cg',
    )
    assert result is None
    assert 'signed' in diagnostic


def test_opt_in_problem_solve_gradient_agrees(monkeypatch):
    rng = np.random.default_rng(2288)
    markets, J, I = 4, 15, 45
    N = markets * J
    ids = np.repeat(np.arange(markets), J)
    prices = rng.uniform(0.6, 2.1, size=N)
    sigma = 0.35
    nodes = rng.normal(size=markets * I)
    delta = -3.0 - prices
    shares = np.empty(N)
    for t in range(markets):
        rows = slice(t * J, (t + 1) * J)
        cols = slice(t * I, (t + 1) * I)
        e = np.exp(delta[rows, None] + sigma * prices[rows, None] * nodes[cols])
        shares[rows] = np.mean(e / (1 + e.sum(axis=0)), axis=1)
    problem = pyblp.Problem(
        [pyblp.Formulation('1 + prices'), pyblp.Formulation('0 + prices')],
        {'market_ids': ids, 'prices': prices, 'shares': shares, 'demand_instruments0': prices**2},
        agent_data={'market_ids': np.repeat(np.arange(markets), I),
                    'nodes': nodes[:, None], 'weights': np.full(markets * I, 1 / I)},
    )
    kwargs = dict(
        sigma=np.array([[sigma]]), method='1s',
        optimization=pyblp.Optimization('return'),
        initial_update=False, error_behavior='raise',
    )
    with monkeypatch.context() as patcher:
        patcher.setattr(pyblp.options, 'implicit_gradient_solver', 'dense')
        baseline = problem.solve(**kwargs)
    with monkeypatch.context() as patcher:
        patcher.setattr(pyblp.options, 'implicit_gradient_solver', 'cg')
        improved = problem.solve(**kwargs)
    np.testing.assert_allclose(improved.delta, baseline.delta, atol=5e-9, rtol=0)
    np.testing.assert_allclose(improved.objective, baseline.objective, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(improved.gradient, baseline.gradient, atol=5e-7, rtol=5e-7)
