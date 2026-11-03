"""Regression test for simulation-error moments and fixed linear parameters."""
import numpy as np
import pyblp


def test_simulation_covariance_keeps_linear_coefficients_fixed(monkeypatch):
    from pyblp.economies.problem import ProblemEconomy

    rng = np.random.default_rng(90210)
    markets, products_per_market, agents_per_market, repetitions = 8, 4, 30, 12
    N = markets * products_per_market
    ids = np.repeat(np.arange(markets), products_per_market)
    prices = rng.uniform(0.3, 2.2, N)
    design = np.column_stack([np.ones(N), prices])
    delta_true = design @ np.array([-1.0, -1.2]) + 0.15 * rng.standard_normal(N)
    sigma = 0.7

    def draw():
        return {
            'market_ids': np.repeat(np.arange(markets), agents_per_market),
            'nodes': rng.normal(size=(markets * agents_per_market, 1)),
            'weights': np.full(markets * agents_per_market, 1 / agents_per_market),
        }

    base = draw()
    resamples = [draw() for _ in range(repetitions)]
    shares = np.empty(N)
    for t in range(markets):
        products = slice(t * products_per_market, (t + 1) * products_per_market)
        agents = slice(t * agents_per_market, (t + 1) * agents_per_market)
        utility = delta_true[products, None] + sigma * prices[products, None] * base['nodes'][agents].T
        exp_utility = np.exp(utility)
        probabilities = exp_utility / (1 + exp_utility.sum(axis=0))
        shares[products] = probabilities.mean(axis=1)

    problem = pyblp.Problem(
        [pyblp.Formulation('1 + prices'), pyblp.Formulation('0 + prices')],
        {'market_ids': ids, 'prices': prices, 'shares': shares, 'demand_instruments0': prices},
        agent_data=base,
    )
    observed = []
    original = ProblemEconomy._compute_progress

    def record(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if kwargs.get('agents_override') is not None:
            observed.append((result.beta.copy(), result.mean_g.copy()))
        return result

    monkeypatch.setattr(ProblemEconomy, '_compute_progress', record)
    results = problem.solve(
        sigma=np.array([[sigma]]),
        sigma_bounds=(np.array([[sigma]]), np.array([[sigma]])),
        method='1s',
        initial_update=False,
        error_behavior='raise',
        iteration=pyblp.Iteration('squarem', {'atol': 1e-13}),
        resample_agent_data=lambda i: resamples[i] if i < len(resamples) else None,
    )
    assert len(observed) == repetitions
    for beta, _ in observed:
        np.testing.assert_allclose(beta, results.beta, rtol=0, atol=0)
    moments = np.array([mean_g.ravel() for _, mean_g in observed])
    expected = problem.N * np.atleast_2d(np.cov(moments, rowvar=False))
    np.testing.assert_allclose(results.simulation_covariances, expected, rtol=2e-10, atol=2e-12)
    assert np.linalg.norm(results.simulation_covariances) > 0
